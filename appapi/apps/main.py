import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from tqdm import tqdm
from torch import autocast
import tokenizer
from PIL import Image
from flask import Flask, request, jsonify
import io
import os

# Initialize your custom pipeline
model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

class InstructPix2PixPipelineCustom:
    def __init__(self, vae, tokenizer, text_encoder, unet, scheduler, image_processor):
        self.vae = vae
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.unet = unet
        self.scheduler = scheduler
        self.image_processor = image_processor
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def get_text_embeds(self, text):
        """returns embeddings for the given `text`"""

        # tokenize the text
        text_input = self.tokenizer(text,
                                    padding='max_length',
                                    #max_length=tokenizer.model_max_length,
                                    max_length=77,
                                    truncation=True,
                                    return_tensors='pt')
        # embed the text
        with torch.no_grad():
            text_embeds = self.text_encoder(text_input.input_ids.to(self.device))[0]
        return text_embeds

    def get_prompt_embeds(self, prompt, prompt_negative=None):
        """returns prompt embeddings based on classifier free guidance"""

        if isinstance(prompt, str):
            prompt = [prompt]

        if prompt_negative is None:
            prompt_negative = ['']
        elif isinstance(prompt_negative, str):
            prompt_negative = [prompt_negative]

        # get conditional prompt embeddings
        cond_embeds = self.get_text_embeds(prompt)
        # get unconditional prompt embeddings
        uncond_embeds = self.get_text_embeds(prompt_negative)

        # instructpix2pix takes conditional embeds first, followed by unconditional embeds twice
        # this is different from other diffusion pipelines
        prompt_embeds = torch.cat([cond_embeds, uncond_embeds, uncond_embeds])
        return prompt_embeds

    def transform_image(self, image):
        """transform image from pytorch tensor to PIL format"""
        image = self.image_processor.postprocess(image, output_type='pil')
        return image

    def get_image_latents(self, image):
        """get image latents to be used with classifier free guidance"""

        # get conditional image embeds
        image = image.to(self.device)
        image_latents_cond = self.vae.encode(image).latent_dist.mode()

        # get unconditional image embeds
        image_latents_uncond = torch.zeros_like(image_latents_cond)
        image_latents = torch.cat([image_latents_cond, image_latents_cond, image_latents_uncond])

        return image_latents

    def get_initial_latents(self, height, width, num_channels_latents, batch_size):
        """returns noise latent tensor of relevant shape scaled by the scheduler"""

        image_latents = torch.randn((batch_size, num_channels_latents, height, width))
        image_latents = image_latents.to(self.device)

        # scale the initial noise by the standard deviation required by the scheduler
        image_latents = image_latents * self.scheduler.init_noise_sigma
        return image_latents


    def denoise_latents(self, prompt_embeds, image_latents, timesteps, latents, guidance_scale, image_guidance_scale):
        """denoises latents from noisy latent to a meaningful latent as conditioned by image_latents"""

        # use autocast for automatic mixed precision (AMP) inference
        with autocast('cuda'):
            for i,  t in tqdm(enumerate(timesteps)):
                # duplicate image latents *thrice* to do classifier free guidance
                latent_model_input = torch.cat([latents] * 3)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                latent_model_input = torch.cat([latent_model_input, image_latents], dim=1)


                # predict noise residuals
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input, t,
                        encoder_hidden_states=prompt_embeds)['sample']

                # separate predictions into conditional (on text), conditional (on image) and unconditional outputs
                noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
                # perform guidance
                noise_pred = (
                    noise_pred_uncond
                    + guidance_scale * (noise_pred_text - noise_pred_image)
                    + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
                    )

                # remove the noise from the current sample i.e. go from x_t to x_{t-1}
                latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']

        return latents

    def __call__(self, prompt, image, prompt_negative=None, num_inference_steps=20, guidance_scale=10, image_guidance_scale=2):
        """generates new image based on the `prompt` and the `image`"""

        # encode input prompt
        prompt_embeds = self.get_prompt_embeds(prompt, prompt_negative)

        # preprocess image
        image = self.image_processor.preprocess(image)

        # prepare image latents
        image = image.half()
        image_latents = self.get_image_latents(image)

        # prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        height_latents, width_latents = image_latents.shape[-2:]

        # prepare the initial image in the latent space (noise on which we will do reverse diffusion)
        num_channels_latents = self.vae.config.latent_channels
        batch_size = prompt_embeds.shape[0] // 2
        latents = self.get_initial_latents(height_latents, width_latents, num_channels_latents, batch_size)

        # denoise latents
        latents = self.denoise_latents(prompt_embeds,
                                       image_latents,
                                       timesteps,
                                       latents,
                                       guidance_scale,
                                       image_guidance_scale)

        # decode latents to get the image into pixel space
        latents = latents.to(torch.float16) # change dtype of latents since
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

        # convert to PIL Image format
        image = image.detach() # detach to remove any computed gradients
        image = self.transform_image(image)

        return image

vae = pipe.vae
tokenizer = pipe.tokenizer
text_encoder = pipe.text_encoder
unet = pipe.unet
scheduler = pipe.scheduler
image_processor = pipe.image_processor

custom_pipe = InstructPix2PixPipelineCustom(vae, tokenizer, text_encoder, unet, scheduler, image_processor)
# Create a Flask app
app = Flask(__name__)
# Define a Flask route
@app.route('/api', methods=['POST'])
def index():
    try:
        # Get the prompt and image URL from the request
        prompt = request.json["prompt"]
        image_url = request.json["image"]

        # Load the original image from the URL
        response = requests.get(image_url)
        image_data = Image.open(io.BytesIO(response.content))

        # Print the prompt
        print(prompt)

        # Save the original image
        image_data.save("https://drive.google.com/file/d/1KnVLGH6j-ytxmpnS4m8CNP7_7CYSpiTu/view?usp=drive_link")

        # Load the original image
        original_image_path = 'https://drive.google.com/file/d/1KnVLGH6j-ytxmpnS4m8CNP7_7CYSpiTu/view?usp=drive_link'  # Path to your original image
        original_image = Image.open(original_image_path).convert("RGB")
        # Generate a modified image using the custom pipeline
        modified_image = custom_pipe(prompt, original_image, num_inference_steps=20)[0]
        modified_image.show()
            # Save the modified image
        modified_image.save("/content/drive/MyDrive/genai-cx/modified_image.jpg")
        modified_image_url="https://drive.google.com/file/d/1GoNEp5b_pznNP4rgPy8tz0Bl0fQ6aUeF/view?usp=sharing"
        return jsonify({"image": modified_image_url }),200

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)
