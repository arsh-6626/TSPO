import logging
import cv2
import numpy as np
import torch
from PIL import Image

from src.config import DEVICE, DTYPE
from src.utils.helpers import IMG_T512, alpha_schedule

logger = logging.getLogger("train_tspo")


class Inpainter:
    def __init__(self, inpainter_pth=None):
        from diffusers import DDIMScheduler, StableDiffusionInpaintPipeline

        logger.info("Loading SD inpainting pipeline …")
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=DTYPE,
            safety_checker=None,
            requires_safety_checker=False,
        )
        self.pipe.safety_checker    = None
        self.pipe.feature_extractor = None
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to(str(DEVICE))
        self.pipe.enable_attention_slicing()

        if inpainter_pth:
            ckpt = torch.load(inpainter_pth, map_location="cpu")
            for k in ("unet", "model", "state_dict", "model_state_dict"):
                if isinstance(ckpt, dict) and k in ckpt:
                    ckpt = ckpt[k]
                    break
            self.pipe.unet.load_state_dict(ckpt, strict=False)

    def generate_candidates(self, image, mask, prompt, knobs,
                             height=512, width=512, steps=20):
        neg   = "blurry, watermark, text, naked, violence, blood, gore"
        img_r = image.convert("RGB").resize((width, height))
        out   = []

        for knob in knobs:
            pm    = self._proc_mask(mask, knob.mask_dilation, knob.mask_feather, height, width)
            pil_m = Image.fromarray((pm * 255).astype(np.uint8)).convert("L")
            ni    = self._jitter(img_r, knob.noise_jitter, knob.seed_offset)
            gen   = torch.Generator(str(DEVICE)).manual_seed(42 + knob.seed_offset)

            with torch.autocast(str(DEVICE), enabled=(str(DEVICE) == "cuda")):
                result = self.pipe(
                    prompt=prompt, negative_prompt=neg,
                    image=ni, mask_image=pil_m,
                    guidance_scale=knob.cfg_scale,
                    num_inference_steps=steps,
                    generator=gen, height=height, width=width,
                ).images[0]
            out.append(result)
        return out

    def short_ddim_inversion(self, winner_rgb, d_steps, t_norm):
        sched = self.pipe.scheduler
        vae   = self.pipe.vae
        img_t = IMG_T512(winner_rgb.convert("RGB")).unsqueeze(0).to(DEVICE).to(DTYPE)

        with torch.no_grad():
            z0 = vae.encode(img_t).latent_dist.mean * vae.config.scaling_factor

        sched.set_timesteps(d_steps)
        target_t = int((1.0 - t_norm) * 500)
        z = z0.clone()

        for t in reversed(sched.timesteps[:d_steps]):
            if t.item() < target_t:
                break
            with torch.no_grad():
                ab = sched.alphas_cumprod[t].to(DEVICE)
                z  = (ab ** 0.5) * z + ((1 - ab) ** 0.5) * torch.randn_like(z)
        return z.float()

    def blend_latents(self, z_ctrl, z_edit, mask, t_norm):
        α       = alpha_schedule(t_norm)
        feather = cv2.GaussianBlur(mask.astype(np.float32), (15, 15), 5)
        m64     = cv2.resize(feather, (64, 64), cv2.INTER_LINEAR)
        m_t     = torch.from_numpy(m64).to(DEVICE).float().unsqueeze(0).unsqueeze(0)
        return (1.0 - α * m_t) * z_ctrl + α * m_t * z_edit

    @staticmethod
    def _proc_mask(mask, dil, feath, h, w):
        m = cv2.resize(mask.astype(np.uint8), (w, h), cv2.INTER_NEAREST)
        if dil > 0:
            k = max(3, int(dil * 30) | 1)
            m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        if feath > 0:
            f = max(3, int(feath * 31) | 1)
            m = (cv2.GaussianBlur(m.astype(np.float32), (f, f), 0) > 0.5).astype(np.uint8)
        return m

    @staticmethod
    def _jitter(image, jitter, seed):
        if jitter < 1e-4:
            return image
        rng = np.random.RandomState(seed)
        img = np.array(image).astype(np.float32)
        return Image.fromarray(
            (img + rng.randn(*img.shape) * jitter * 20).clip(0, 255).astype(np.uint8)
        )
