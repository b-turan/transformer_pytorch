import transformers


def count_parameters(model):
    """Counts the learnable parameters of a given model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def epoch_time(start_time, end_time):
    """Returns elapsed time in minutes and seconds."""
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def build_model(model_name, IS_PRETRAINED, device):
    """
    Returns T5 model (pretrained or randomly initialized)
    """
    if IS_PRETRAINED:
        # TODO: check alternative loading with AutoModel.from_pretrained()
        model = transformers.T5ForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="auto"
        ).to(device)
    else:
        config = transformers.AutoConfig.from_pretrained(
            model_name
        )  # see transformers/issues/14674
        model = transformers.T5ForConditionalGeneration(config).to(device)
    return model
