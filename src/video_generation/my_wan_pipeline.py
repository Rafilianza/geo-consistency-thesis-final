"""
my_wan_pipeline.py — Subclass of WanPipeline with research hooks.

Provides controlled interventions in the denoising loop for Path A research:
generating video + semantic segmentation jointly, where SAM-derived signals
can modify the denoising trajectory.

Hooks exposed (all optional, default = standard WanPipeline behavior):
  - decode_at_steps: list of step indices where intermediate latents should be
    decoded to pixels for inspection/segmentation
  - step_intervention: callable invoked AFTER noise_pred is computed but BEFORE
    scheduler.step. Can modify noise_pred or latents.
  - cfg_intervention: callable that replaces the standard CFG formula

Returns the standard WanPipelineOutput plus extra fields (intermediate_decodes,
intermediate_latents) when those hooks are enabled.
"""

from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field

import torch

from diffusers import WanPipeline
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput


@dataclass
class MyWanPipelineOutput(WanPipelineOutput):
    """Extends WanPipelineOutput with optional research-mode outputs."""
    intermediate_decodes: Dict[int, Any] = field(default_factory=dict)  # step_idx -> decoded video tensor
    intermediate_latents: Dict[int, torch.Tensor] = field(default_factory=dict)  # step_idx -> latent snapshot
    final_latent: Optional[torch.Tensor] = None  # latent after last denoising step, before VAE decode


class MyWanPipeline(WanPipeline):
    """
    Research subclass of WanPipeline.

    Identical to WanPipeline when no hooks are provided. With hooks, exposes
    inspection and intervention points in the denoising loop.

    Usage:
        pipe = MyWanPipeline.from_pretrained(...)
        result = pipe(
            prompt="...",
            decode_at_steps=[40, 45, 49],          # for Path B: inspect mid-trajectory
            step_intervention=my_sam_callback,      # for Path A: SAM-guided generation
        )
        result.frames           # final video (same as WanPipeline)
        result.intermediate_decodes[40]   # video tensor at step 40 (if decode_at_steps used)
        result.intermediate_latents[40]   # latent at step 40 (if decode_at_steps used)
    """

    def _decode_latents_to_video(self, latents):
        """Decode latents to video tensor using the same logic as the parent's final decode."""
        latents = latents.to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
            1, self.vae.config.z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)
        latents_normalized = latents / latents_std + latents_mean
        video = self.vae.decode(latents_normalized, return_dict=False)[0]
        return video

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        guidance_scale_2: Optional[float] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        # --- Subclass research hooks ---
        decode_at_steps: Optional[List[int]] = None,
        capture_latent_steps: Optional[List[int]] = None,
        step_intervention: Optional[Callable] = None,
        cfg_intervention: Optional[Callable] = None,
        verbose: bool = True,
    ):
        # --- Input checks (mirror parent) ---
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        self.check_inputs(
            prompt, negative_prompt, height, width,
            prompt_embeds, negative_prompt_embeds,
            callback_on_step_end_tensor_inputs, guidance_scale_2,
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        if self.config.boundary_ratio is not None and guidance_scale_2 is None:
            guidance_scale_2 = guidance_scale

        self._guidance_scale = guidance_scale
        self._guidance_scale_2 = guidance_scale_2
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device
        decode_at_steps = set(decode_at_steps or [])
        capture_latent_steps = set(capture_latent_steps or [])
        intermediate_decodes = {}
        intermediate_latents = {}

        # --- Logging Stage A ---
        if verbose:
            print(f"[MyWan STAGE=A init]", flush=True)
            print(f"  vae.dtype             = {self.vae.dtype}", flush=True)
            print(f"  transformer.dtype     = {self.transformer.dtype}", flush=True)
            print(f"  text_encoder.dtype    = {self.text_encoder.dtype}", flush=True)
            print(f"  boundary_ratio        = {self.config.boundary_ratio}", flush=True)
            print(f"  expand_timesteps      = {self.config.expand_timesteps}", flush=True)
            print(f"  decode_at_steps       = {sorted(decode_at_steps) if decode_at_steps else 'none'}", flush=True)
            print(f"  step_intervention     = {step_intervention is not None}", flush=True)
            print(f"  cfg_intervention      = {cfg_intervention is not None}", flush=True)

        # --- Batch size ---
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # --- Encode prompts ---
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        transformer_dtype = self.transformer.dtype if self.transformer is not None else self.transformer_2.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        if verbose:
            print(f"[MyWan STAGE=C text_encoded]", flush=True)
            print(f"  prompt_embeds.shape   = {tuple(prompt_embeds.shape)}", flush=True)
            print(f"  prompt_embeds.dtype   = {prompt_embeds.dtype}", flush=True)

        # --- Timesteps ---
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        if verbose:
            print(f"[MyWan STAGE=D scheduler]", flush=True)
            print(f"  scheduler_class       = {type(self.scheduler).__name__}", flush=True)
            print(f"  num_inference_steps   = {len(timesteps)}", flush=True)
            print(f"  flow_shift            = {getattr(self.scheduler.config, 'flow_shift', 'N/A')}", flush=True)
            print(f"  timesteps_first5      = {timesteps[:5].tolist()}", flush=True)

        # --- Prepare latents ---
        num_channels_latents = (
            self.transformer.config.in_channels
            if self.transformer is not None
            else self.transformer_2.config.in_channels
        )
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height, width, num_frames,
            torch.float32, device, generator, latents,
        )
        mask = torch.ones(latents.shape, dtype=torch.float32, device=device)

        if verbose:
            print(f"[MyWan STAGE=E initial_latents]", flush=True)
            print(f"  latents.shape         = {tuple(latents.shape)}", flush=True)
            print(f"  latents.dtype         = {latents.dtype}", flush=True)
            print(f"  latents.mean          = {latents.float().mean().item():.6f}", flush=True)
            print(f"  latents.std           = {latents.float().std().item():.6f}", flush=True)

        # --- Denoising loop ---
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        if self.config.boundary_ratio is not None:
            boundary_timestep = self.config.boundary_ratio * self.scheduler.config.num_train_timesteps
        else:
            boundary_timestep = None

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                self._current_timestep = t

                # Select transformer (handles single + dual-expert)
                if boundary_timestep is None or t >= boundary_timestep:
                    current_model = self.transformer
                    current_guidance_scale = guidance_scale
                else:
                    current_model = self.transformer_2
                    current_guidance_scale = guidance_scale_2

                latent_model_input = latents.to(transformer_dtype)

                if self.config.expand_timesteps:
                    temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
                    timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
                else:
                    timestep = t.expand(latents.shape[0])

                # Conditional forward
                with current_model.cache_context("cond"):
                    noise_pred = current_model(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]

                # Unconditional forward (for CFG)
                if self.do_classifier_free_guidance:
                    with current_model.cache_context("uncond"):
                        noise_uncond = current_model(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=negative_prompt_embeds,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]

                    # CFG combination — overridable
                    if cfg_intervention is not None:
                        noise_pred = cfg_intervention(
                            step_idx=i, t=t,
                            noise_cond=noise_pred, noise_uncond=noise_uncond,
                            guidance_scale=current_guidance_scale,
                        )
                    else:
                        noise_pred = noise_uncond + current_guidance_scale * (noise_pred - noise_uncond)

                # === HOOK: step intervention (Path A) ===
                # Called AFTER noise_pred is finalized, BEFORE scheduler.step.
                # Can modify noise_pred or latents (e.g., based on SAM output).
                if step_intervention is not None:
                    intervention_out = step_intervention(
                        step_idx=i, t=t,
                        latents=latents, noise_pred=noise_pred,
                        prompt_embeds=prompt_embeds,
                    )
                    if intervention_out is not None:
                        latents = intervention_out.get("latents", latents)
                        noise_pred = intervention_out.get("noise_pred", noise_pred)

                # Scheduler step
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                # === HOOK: decode at this step (Path B foundation) ===
                if i in decode_at_steps:
                    if verbose:
                        print(f"[MyWan STAGE=F decode_at_step={i}]", flush=True)
                    intermediate_latents[i] = latents.detach().clone().cpu()
                    decoded = self._decode_latents_to_video(latents)
                    intermediate_decodes[i] = decoded.detach().cpu()
                # == Capture RAW Latent
                if i in capture_latent_steps:
                    intermediate_latents[i] = latents.detach().clone().cpu()
                
                # Standard callback
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        self._current_timestep = None

        # Capture the final latent before the VAE decode (Goal 2 foundation)
        final_latent = latents.detach().clone().cpu()
        if verbose:
            print(f"[MyWan STAGE=G final_latent captured]", flush=True)
            print(f"  final_latent.shape = {tuple(final_latent.shape)}", flush=True)
            print(f"  final_latent.dtype = {final_latent.dtype}", flush=True)

        # Final decode (same as parent)
        if not output_type == "latent":
            if verbose:
                print(f"[MyWan STAGE=H final_decode]", flush=True)
                print(f"  latents.dtype before cast = {latents.dtype}", flush=True)
                print(f"  vae.dtype                 = {self.vae.dtype}", flush=True)
            video = self._decode_latents_to_video(latents)
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return MyWanPipelineOutput(
            frames=video,
            intermediate_decodes=intermediate_decodes,
            intermediate_latents=intermediate_latents,
            final_latent=final_latent,
        )