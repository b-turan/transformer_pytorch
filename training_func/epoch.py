import numpy as np
import torch as th
from tqdm import tqdm


def train_epoch(model, train_dataloader, optimizer, lr_scheduler, CLIP, device):
    '''
    Trains model on the entire dataset for one epoch.
    ------------------------------------
    model (nn.model): Torch model
    train_dataloader (torch.dataloader): Dataloader
    optimizer (th.optim): Optimizer
    lr_scheduler (transformers.get_scheduler): learning rate scheduler
    CLIP (int): Gradient Clipping    
    device (torch.device): cuda or cpu
    ------------------------------------
    returns average epoch loss
    '''
    model.train()
    epoch_loss = 0
    for batch in tqdm(train_dataloader):
        batch = batch.to(device)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        th.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        epoch_loss += loss.item()
    return epoch_loss / len(train_dataloader)


def validation_epoch(model, dataloader, metric, tokenizer, device):
    '''
    Evaluates model on the entire dataset.
    ------------------------------------
    model (nn.model): Torch model
    dataloader (torch.dataloader): Dataloader
    metric (Datasets.metric): SacreBleu Metric
    tokenizer (transformer.tokenizer): Tokenizer
    device (torch.device): cuda or cpu
    ------------------------------------
    returns average epoch validation loss
    '''
    model.eval()
    for batch in tqdm(dataloader):
        batch = batch.to(device)
        with th.no_grad():
            generated_tokens = model.generate(batch["input_ids"], 
                                            attention_mask=batch["attention_mask"],
                                            max_length=128
            )
        labels = batch["labels"]
        decoded_preds, decoded_labels = postprocess(generated_tokens, labels, tokenizer)
        metric.add_batch(predictions=decoded_preds, references=decoded_labels)
    results = metric.compute()
    return results


def postprocess(predictions, labels, tokenizer):
    '''
    Postprecessor to generate translations by the model.
    ------------------------------------
    predictions (torch.tensor): Model's prediction
    labels (torch.tensor): Targets
    tokenizer (transformer.tokenizer): Tokenizer
    ------------------------------------
    returns decoded translations and decoded labels
    '''
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]
    return decoded_preds, decoded_labels
