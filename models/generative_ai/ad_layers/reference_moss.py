"""Run the MOSS-Image-Edit reference workflow without ComfyUI."""

from math import sqrt

from PIL import Image

NEGATIVE_PROMPT = "nsfw, worst quality, bad anatomy, watermark, jpeg artifact"
STEPS, CFG, SHIFT, SKIPPED_LAYER = 30, 4.0, 5.0, 22


def reference_size(width, height):
    """Scale to one megapixel with the workflow's 16-pixel alignment."""
    scale = 1024 / sqrt(width * height)
    return tuple(max(16, round(side * scale / 16) * 16) for side in (width, height))


def normal_sigmas(steps=STEPS):
    import torch

    def shift(time):
        return SHIFT * time / (1 + (SHIFT - 1) * time)

    times = torch.linspace(1, shift(1 / 1000), steps)
    return torch.cat((shift(times), torch.zeros(1)))


def generate_batch(jobs, snapshot, adapter, device="cuda"):
    """Yield ads in job order. A job is a pair of a prompt and a RGB PIL image."""
    import gc
    import json

    import torch
    from moss_edit import components, lora, processing
    from moss_edit.model import MossEditModel
    from transformers import AutoModel, AutoTokenizer

    def free_memory():
        gc.collect()
        torch.cuda.empty_cache()

    model_config = json.loads((snapshot / "transformer/config.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(snapshot / "tokenizer", local_files_only=True)
    encoder = AutoModel.from_pretrained(
        snapshot / "text_encoder", dtype=torch.bfloat16, device_map={"": device}, local_files_only=True,
    ).requires_grad_(False).eval()
    gpu = torch.device(device)
    text = components.TextComponents(encoder, tokenizer, gpu, gpu, metadata={"text_dim": model_config["text_dim"]})
    with torch.no_grad():
        negative = components.encode_prompts(text, NEGATIVE_PROMPT, negative_prompts=None, device="cpu").positive
        positives = [
            components.encode_prompts(text, prompt, negative_prompts=None, device="cpu").positive
            for prompt, _ in jobs
        ]
    del encoder, text
    free_memory()

    vae_class = components.load_moss_module(snapshot).AutoencoderKLFlux2
    vae = vae_class.from_pretrained(
        snapshot / "vae", torch_dtype=torch.float32, local_files_only=True,
    ).requires_grad_(False).eval().to(device)
    references = []
    with torch.no_grad():
        for _, source in jobs:
            reference = source.resize(reference_size(*source.size), Image.Resampling.LANCZOS)
            latent = processing.encode_image(vae, processing.image_to_tensor(reference), sample_posterior=False)
            references.append(latent.cpu())
    del latent
    vae.to("cpu")
    free_memory()

    model = components.load_transformer(snapshot, device=device, dtype=torch.bfloat16)
    options = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    first = next(iter(options["modules"].values()))
    lora.inject_lora(model, targets=options["targets"], rank=first["rank"], alpha=first["alpha"],
                     dropout=first["dropout"], param_dtype=getattr(torch, first["param_dtype"]))
    lora.load_adapter(model, adapter)
    model.requires_grad_(False).eval()

    class LayerGuidanceModel(MossEditModel):
        skip_layer = False

        def _body_block(self, layer, body, *args, **kwargs):
            if self.skip_layer and layer is model.layers[SKIPPED_LAYER]:
                return body
            return super()._body_block(layer, body, *args, **kwargs)

    editor = LayerGuidanceModel(model).eval()
    schedule = normal_sigmas().to(device)
    negative = negative.to(device)
    vae.to(device)
    for positive, reference in zip(positives, references, strict=True):
        with torch.no_grad():
            reference = reference.to(device)
            positive = positive.to(device)
            cache = editor.prepare_cache([reference])
            seed = torch.Generator("cpu").manual_seed(1)
            sample = torch.randn(reference.shape, generator=seed, dtype=torch.float32).to(device)
            for sigma, next_sigma in zip(schedule[:-1], schedule[1:]):
                editor.skip_layer = False
                cond = editor(sample, sigma.reshape(1), positive.embeds, [reference], positive.attention_mask, cache)
                editor.skip_layer = True
                uncond = editor(sample, sigma.reshape(1), negative.embeds, [reference], negative.attention_mask, cache)
                sample.add_(uncond.float() + CFG * (cond.float() - uncond.float()), alpha=float(next_sigma - sigma))
            del cache
            if not torch.isfinite(sample).all():
                raise RuntimeError("MOSS-Image-Edit returned non-finite latents")
            pixels = processing.decode_latents(vae, sample)[0].cpu().clamp(-1, 1)
            pixels = ((pixels + 1) * 127.5).round().byte().permute(1, 2, 0).numpy()
        yield Image.fromarray(pixels)
