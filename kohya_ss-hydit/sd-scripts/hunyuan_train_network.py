import warnings

warnings.filterwarnings("ignore")
import os
from PIL import Image
import time
import json
import re
from typing import (
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)
import numpy as np
import argparse
from k_diffusion.external import DiscreteVDDPMDenoiser
from k_diffusion.sampling import sample_euler_ancestral, get_sigmas_exponential
import torch
from diffusers import DDPMScheduler
from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from library import (
    hunyuan_models,
    hunyuan_utils,
    sdxl_model_util,
    sdxl_train_util,
    train_util,
)
import train_network
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


def load_scheduler_sigmas(beta_start=0.00085, beta_end=0.018, num_train_timesteps=1000):
    betas = (
        torch.linspace(
            beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32
        )
        ** 2
    )
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    sigmas = np.array(((1 - alphas_cumprod) / alphas_cumprod) ** 0.5)
    sigmas = np.concatenate([sigmas[::-1], [0.0]]).astype(np.float32)
    sigmas = torch.from_numpy(sigmas)
    return alphas_cumprod, sigmas


def line_to_prompt_dict(line: str) -> dict:
    # subset of gen_img_diffusers
    prompt_args = line.split(" --")
    prompt_dict = {}
    prompt_dict["prompt"] = prompt_args[0]

    for parg in prompt_args:
        try:
            m = re.match(r"w (\d+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["width"] = int(m.group(1))
                continue

            m = re.match(r"h (\d+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["height"] = int(m.group(1))
                continue

            m = re.match(r"d (\d+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["seed"] = int(m.group(1))
                continue

            m = re.match(r"s (\d+)", parg, re.IGNORECASE)
            if m:  # steps
                prompt_dict["sample_steps"] = max(1, min(1000, int(m.group(1))))
                continue

            m = re.match(r"l ([\d\.]+)", parg, re.IGNORECASE)
            if m:  # scale
                prompt_dict["scale"] = float(m.group(1))
                continue

            m = re.match(r"n (.+)", parg, re.IGNORECASE)
            if m:  # negative prompt
                prompt_dict["negative_prompt"] = m.group(1)
                continue

            m = re.match(r"ss (.+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["sample_sampler"] = m.group(1)
                continue

            m = re.match(r"cn (.+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["controlnet_image"] = m.group(1)
                continue

        except ValueError as ex:
            logger.error(f"Exception in parsing / 解析エラー: {parg}")
            logger.error(ex)

    return prompt_dict

class HunYuanNetworkTrainer(train_network.NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.vae_scale_factor = sdxl_model_util.VAE_SCALE_FACTOR
        self.is_sdxl = True

    def assert_extra_args(self, args, train_dataset_group):
        super().assert_extra_args(args, train_dataset_group)
        # sdxl_train_util.verify_sdxl_training_args(args)

        if args.cache_text_encoder_outputs:
            assert (
                train_dataset_group.is_text_encoder_output_cacheable()
            ), "when caching Text Encoder output, either caption_dropout_rate, shuffle_caption, token_warmup_step or caption_tag_dropout_rate cannot be used / Text Encoderの出力をキャッシュするときはcaption_dropout_rate, shuffle_caption, token_warmup_step, caption_tag_dropout_rateは使えません"

        assert (
            args.network_train_unet_only or not args.cache_text_encoder_outputs
        ), "network for Text Encoder cannot be trained with caching Text Encoder outputs / Text Encoderの出力をキャッシュしながらText Encoderのネットワークを学習することはできません"

        train_dataset_group.verify_bucket_reso_steps(16)

    def load_target_model(self, args, weight_dtype, accelerator):
        use_extra_cond = getattr(args, "use_extra_cond", False)
        (
            load_stable_diffusion_format,
            text_encoder1,
            text_encoder2,
            vae,
            unet,
            logit_scale,
            ckpt_info,
        ) = hunyuan_utils.load_target_model(
            args,
            accelerator,
            hunyuan_models.MODEL_VERSION_HUNYUAN_V1_1,
            weight_dtype,
            use_extra_cond=use_extra_cond,
        )

        self.load_stable_diffusion_format = load_stable_diffusion_format
        self.logit_scale = logit_scale
        self.ckpt_info = ckpt_info

        return (
            hunyuan_models.MODEL_VERSION_HUNYUAN_V1_1,
            [text_encoder1, text_encoder2],
            vae,
            unet,
        )

    def load_tokenizer(self, args):
        tokenizer = hunyuan_utils.load_tokenizers()
        return tokenizer

    def load_noise_scheduler(self, args):
        return DDPMScheduler(
            beta_start=0.00085,
            beta_end=args.beta_end,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            clip_sample=False,
            steps_offset=1,
        )

    def is_text_encoder_outputs_cached(self, args):
        return args.cache_text_encoder_outputs

    def cache_text_encoder_outputs_if_needed(
        self,
        args,
        accelerator,
        unet,
        vae,
        tokenizers,
        text_encoders,
        dataset: train_util.DatasetGroup,
        weight_dtype,
    ):
        if args.cache_text_encoder_outputs:
            raise NotImplementedError
        else:
            # Text Encoderから毎回出力を取得するので、GPUに乗せておく
            text_encoders[0].to(accelerator.device, dtype=weight_dtype)
            text_encoders[1].to(accelerator.device, dtype=weight_dtype)

    def get_text_cond(
        self, args, accelerator, batch, tokenizers, text_encoders, weight_dtype
    ):
        if (
            "text_encoder_outputs1_list" not in batch
            or batch["text_encoder_outputs1_list"] is None
        ):
            input_ids1 = batch["input_ids"]
            input_ids2 = batch["input_ids2"]
            logger.debug("input_ids1", input_ids1.shape)
            logger.debug("input_ids2", input_ids2.shape)
            with torch.enable_grad():
                input_ids1 = input_ids1.to(accelerator.device)
                input_ids2 = input_ids2.to(accelerator.device)
                encoder_hidden_states1, mask1, encoder_hidden_states2, mask2 = (
                    hunyuan_utils.hunyuan_get_hidden_states(
                        args.max_token_length,
                        input_ids1,
                        input_ids2,
                        tokenizers[0],
                        tokenizers[1],
                        text_encoders[0],
                        text_encoders[1],
                        None if not args.full_fp16 else weight_dtype,
                        accelerator=accelerator,
                    )
                )
                logger.debug("encoder_hidden_states1", encoder_hidden_states1.shape)
                logger.debug("encoder_hidden_states2", encoder_hidden_states2.shape)
        else:
            raise NotImplementedError
        return encoder_hidden_states1, mask1, encoder_hidden_states2, mask2

    def call_unet(
        self,
        args,
        accelerator,
        unet,
        noisy_latents,
        timesteps,
        text_conds,
        batch,
        weight_dtype,
    ):
        noisy_latents = noisy_latents.to(
            weight_dtype
        )  # TODO check why noisy_latents is not weight_dtype

        # get size embeddings
        orig_size = batch["original_sizes_hw"]  # B, 2
        crop_size = batch["crop_top_lefts"]  # B, 2
        target_size = batch["target_sizes_hw"]  # B, 2
        B, C, H, W = noisy_latents.shape

        style = torch.as_tensor([0] * B, device=accelerator.device)
        image_meta_size = torch.concat(
            [
                orig_size,
                target_size,
                # Not following SDXL but following HunYuan's Implementation
                # TODO examine if this is correct
                torch.zeros_like(target_size),
            ]
        )
        freqs_cis_img = hunyuan_utils.calc_rope(H * 8, W * 8, 2, 88)

        # concat embeddings
        encoder_hidden_states1, mask1, encoder_hidden_states2, mask2 = text_conds
        noise_pred = unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states1,
            text_embedding_mask=mask1,
            encoder_hidden_states_t5=encoder_hidden_states2,
            text_embedding_mask_t5=mask2,
            image_meta_size=image_meta_size,
            style=style,
            cos_cis_img=freqs_cis_img[0],
            sin_cis_img=freqs_cis_img[1],
        )
        # TODO Handle learned sigma correctly
        return noise_pred.chunk(2, dim=1)[0]

    def sample_images(
        self,
        accelerator,
        args,
        epoch,
        global_step,
        device,
        vae,
        tokenizer,
        text_encoder,
        unet,
    ):
        save_dir = args.output_dir + "/sample"
        os.makedirs(save_dir, exist_ok=True)
        steps = global_step
        if steps == 0:
            if not args.sample_at_first:
                return
        else:
            if args.sample_every_n_steps is None and args.sample_every_n_epochs is None:
                return
            if args.sample_every_n_epochs is not None:
                # sample_every_n_steps は無視する
                if epoch is None or epoch % args.sample_every_n_epochs != 0:
                    return
            else:
                if (
                    steps % args.sample_every_n_steps != 0 or epoch is not None
                ):  # steps is not divisible or end of epoch
                    return
        # logger.warning("Sampling images not supported yet.")
        # read prompts
        if args.sample_prompts.endswith(".txt"):
            with open(args.sample_prompts, "r", encoding="utf-8") as f:
                lines = f.readlines()
            prompts = [
                line.strip() for line in lines if len(line.strip()) > 0 and line[0] != "#"
            ]
        elif args.sample_prompts.endswith(".toml"):
            with open(args.sample_prompts, "r", encoding="utf-8") as f:
                data = toml.load(f)
            prompts = [
                dict(**data["prompt"], **subset) for subset in data["prompt"]["subset"]
            ]
        elif args.sample_prompts.endswith(".json"):
            with open(args.sample_prompts, "r", encoding="utf-8") as f:
                prompts = json.load(f)
        for i in range(len(prompts)):
            prompt_dict = prompts[i]
            if isinstance(prompt_dict, str):
                prompt_dict = line_to_prompt_dict(prompt_dict)
                prompts[i] = prompt_dict
            assert isinstance(prompt_dict, dict)
            # Adds an enumerator to the dict based on prompt position. Used later to name image files. Also cleanup of extra data in original prompt dict.
            prompt_dict["enum"] = i
            prompt_dict.pop("subset", None)
        CLIP_TOKENS = 75 * 2 + 2
        ATTN_MODE = "xformers"
        DEVICE = "cuda"
        DTYPE = torch.float16
        BETA_END = 0.018
        USE_EXTRA_COND = False
        with torch.inference_mode(True), torch.no_grad():
            for prompt_dict in prompts:
                assert isinstance(prompt_dict, dict)
                negative_prompt: str = prompt_dict.get("negative_prompt", "")
                sample_steps = prompt_dict.get("sample_steps", 30)
                width = prompt_dict.get("width", 512)
                height = prompt_dict.get("height", 512)
                scale = prompt_dict.get("scale", 7.5)
                seed = prompt_dict.get("seed")
                controlnet_image = prompt_dict.get("controlnet_image")
                prompt: str = prompt_dict.get("prompt", "")
                sampler_name: str = prompt_dict.get("sample_sampler", args.sample_sampler)
                alphas, sigmas = load_scheduler_sigmas(beta_end=BETA_END)
                (
                    denoiser,
                    patch_size,
                    head_dim,
                    clip_tokenizer,
                    clip_encoder,
                    mt5_embedder,
                ) = unet, 2, 88, tokenizer[0], text_encoder[0], text_encoder[1]

                with torch.autocast("cuda"):
                    clip_h, clip_m, mt5_h, mt5_m = hunyuan_utils.get_cond(
                        prompt,
                        mt5_embedder,
                        clip_tokenizer,
                        clip_encoder,
                        # Should be same as original implementation with max_length_clip=77
                        # Support 75*n + 2
                        max_length_clip=CLIP_TOKENS,
                    )
                    neg_clip_h, neg_clip_m, neg_mt5_h, neg_mt5_m = hunyuan_utils.get_cond(
                        negative_prompt,
                        mt5_embedder,
                        clip_tokenizer,
                        clip_encoder,
                        max_length_clip=CLIP_TOKENS,
                    )
                    clip_h = torch.concat([clip_h, neg_clip_h], dim=0)
                    clip_m = torch.concat([clip_m, neg_clip_m], dim=0)
                    mt5_h = torch.concat([mt5_h, neg_mt5_h], dim=0)
                    mt5_m = torch.concat([mt5_m, neg_mt5_m], dim=0)
                    torch.cuda.empty_cache()

                style = torch.as_tensor([0] * 2, device=DEVICE)
                # src hw, dst hw, 0, 0
                size_cond = [height, width, height, width, 0, 0]
                image_meta_size = torch.as_tensor([size_cond] * 2, device=DEVICE)
                freqs_cis_img = hunyuan_utils.calc_rope(height, width, patch_size, head_dim)

                denoiser_wrapper = DiscreteVDDPMDenoiser(
                    # A quick patch for learn_sigma
                    lambda *args, **kwargs: denoiser(*args, **kwargs).chunk(2, dim=1)[0],
                    alphas,
                    False,
                ).to(DEVICE)

                def cfg_denoise_func(x, sigma):
                    cond, uncond = denoiser_wrapper(
                        x.repeat(2, 1, 1, 1),
                        sigma.repeat(2),
                        encoder_hidden_states=clip_h,
                        text_embedding_mask=clip_m,
                        encoder_hidden_states_t5=mt5_h,
                        text_embedding_mask_t5=mt5_m,
                        image_meta_size=image_meta_size,
                        style=style,
                        cos_cis_img=freqs_cis_img[0],
                        sin_cis_img=freqs_cis_img[1],
                    ).chunk(2, dim=0)
                    return uncond + (cond - uncond) * scale

                sigmas = denoiser_wrapper.get_sigmas(sample_steps).to(DEVICE)
                sigmas = get_sigmas_exponential(
                    sample_steps, denoiser_wrapper.sigma_min, denoiser_wrapper.sigma_max, DEVICE
                )
                x1 = torch.randn(1, 4, height // 8, width // 8, dtype=torch.float16, device=DEVICE)

                with torch.autocast("cuda"):
                    sample = sample_euler_ancestral(
                        cfg_denoise_func,
                        x1 * sigmas[0],
                        sigmas,
                    )
                    torch.cuda.empty_cache()
                    with torch.no_grad():
                        latent = sample / 0.13025
                        vae.to(DEVICE)
                        image = vae.decode(latent).sample
                        image = (image / 2 + 0.5).clamp(0, 1)
                        image = image.permute(0, 2, 3, 1).cpu().numpy()
                        image = (image * 255).round().astype(np.uint8)
                        image = [Image.fromarray(im) for im in image]
                        for im in image:
                            # im.save(f"test_1600_{VERSION}_lora.png")
                            ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
                            num_suffix = f"e{epoch:06d}" if epoch is not None else f"{steps:06d}"
                            seed_suffix = "" if seed is None else f"_{seed}"
                            i: int = prompt_dict["enum"]
                            img_filename = f"{'' if args.output_name is None else args.output_name + '_'}{num_suffix}_{i:02d}_{ts_str}{seed_suffix}.png"
                            im.save(os.path.join(save_dir, img_filename))

def setup_parser() -> argparse.ArgumentParser:
    parser = train_network.setup_parser()
    sdxl_train_util.add_sdxl_training_arguments(parser)
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    trainer = HunYuanNetworkTrainer()
    trainer.train(args)
