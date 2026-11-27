"""Generate ads using the official FLUX.3 [klein] reference pipeline."""


def generate_batch(jobs, weights):
    from flux3.pipeline import MODELS, Pipeline, prepare_reference

    pipeline = Pipeline(
        {"distilled": str(weights / "variants/fp8r")},
        str(weights / "text_encoder"),
        str(weights / "ae.safetensors"),
    )
    for prompt, source in jobs:
        reference, _ = prepare_reference(source)
        yield pipeline.generate(
            "distilled", prompt, [reference], 1024, 1024,
            steps=MODELS["distilled"]["steps"], guidance=None, seed=1,
        )
