import numpy as np
import sh
import torch
from accelerate import Accelerator
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    T5ForConditionalGeneration,
    get_scheduler,
)

#### See Course on translation: https://huggingface.co/course/chapter7/4?fw=pt ####

# remove and recreate logs folder for development purposes
sh.rm("-r", "-f", "runs")
sh.mkdir("runs")

# raw_datasets = load_dataset("kde4", lang1="en", lang2="de")
raw_datasets = load_dataset("wmt16", "de-en")  # {train, validation, test}
split_datasets = raw_datasets["train"].train_test_split(train_size=0.3, seed=20)
split_datasets = split_datasets["train"].train_test_split(train_size=0.9, seed=20)
split_datasets["validation"] = split_datasets.pop("test")

# model_checkpoint = "Helsinki-NLP/opus-mt-en-fr"
model_checkpoint = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, return_tensors="pt")
config = AutoConfig.from_pretrained(model_checkpoint)  # see transformers/issues/14674
model = T5ForConditionalGeneration(config)
# model = AutoModelForSeq2SeqLM.from_config(config)

writer = SummaryWriter()

max_input_length = 64
max_target_length = 64


def preprocess_function(examples):
    """
    TODO(b-turan): Add prefix for t5-model.
    """
    inputs = [ex["en"] for ex in examples["translation"]]
    targets = [ex["de"] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Set up the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def initialize_weights(m):
    """
    Modifies weight initialization,
    https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/.
    ------------------------------------
    Returns initialized model 'm'
    """
    if hasattr(m, "weight") and m.weight.dim() > 1:
        torch.nn.init.xavier_uniform_(m.weight.data)


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # In case the model returns more than the prediction logits
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100s in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}


def postprocess(predictions, labels):
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


tokenized_datasets = split_datasets.map(
    preprocess_function, batched=True, remove_columns=split_datasets["train"].column_names
)

# model.apply(initialize_weights)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
metric = load_metric("sacrebleu")

tokenized_datasets.set_format("torch")
train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=16,
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], collate_fn=data_collator, batch_size=16
)

optimizer = AdamW(model.parameters(), lr=2e-5)  # TODO: replace with torch.optim.AdamW
# optimizer = th.optim.Adam(model.parameters(), lr=LEARNING_RATE)

accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

num_train_epochs = 30
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_train_epochs):
    # Training
    model.train()
    for batch in train_dataloader:
        # see https://huggingface.co/docs/accelerate/accelerator
        # TODO (when using accelerator): Use clipgrad_norm() instead of torch.nn.utils.clip_grad_norm_
        # TODO (when using accelerator): and clipgrad_value() instead of torch.nn.utils.clip_grad_value
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # Evaluation
    model.eval()
    for batch in tqdm(eval_dataloader):
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=64,
            )
        labels = batch["labels"]

        # Necessary to pad predictions and labels for being gathered
        generated_tokens = accelerator.pad_across_processes(
            generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
        )
        labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

        predictions_gathered = accelerator.gather(generated_tokens)
        labels_gathered = accelerator.gather(labels)

        decoded_preds, decoded_labels = postprocess(predictions_gathered, labels_gathered)
        metric.add_batch(predictions=decoded_preds, references=decoded_labels)

    results = metric.compute()
    print(f"epoch {epoch}, BLEU score: {results['score']:.2f}")
    writer.add_scalar("SacreBLEU/valid", results["score"], epoch)
    writer.flush
