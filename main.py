import math
import time

import datasets
import sh
import torch as th
import transformers
from torch.utils.tensorboard import SummaryWriter

from pre_processor.pre_processor import get_dataloader, tokenize_datasets
from training_func.epoch import train_epoch, validation_epoch
from utils import arg_parser, utils

device = th.device("cuda" if th.cuda.is_available() else "cpu")
print("Running on Device:", device)

# remove and recreate logs folder for development purposes
sh.rm("-r", "-f", "runs")
sh.mkdir("runs")


def main():
    parser = arg_parser.create_parser()
    args = parser.parse_args()
    # initialize tensorboard
    writer = SummaryWriter()

    # trainer specific args
    IS_TRAINING = args.train
    N_EPOCHS = args.epochs
    N_SAMPLES = args.n_samples
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    NUM_WARMUP_STEPS = args.num_warmup_steps
    CLIP = args.clip
    NUM_WORKERS = args.num_workers
    # MOMENTUM = args.momentum
    # model specific args
    MODEL = args.model
    IS_PRETRAINED = args.is_pretrained
    MAX_INPUT_LENGTH = args.max_input_length
    MAX_TARGET_LENGTH = args.max_target_length
    # program specific args
    # SEED = args.seed
    DEBUG = args.debug

    # initialize {is_pretrained} T5-Tokenizer and T5-Model
    tokenizer = transformers.T5Tokenizer.from_pretrained(MODEL)
    model = utils.build_model(MODEL, IS_PRETRAINED, device)

    # data pre-processing / tokenization
    tokenized_datasets = tokenize_datasets(
        tokenizer=tokenizer,
        n_samples=N_SAMPLES,
        max_input_length=MAX_INPUT_LENGTH,
        max_target_length=MAX_TARGET_LENGTH,
        debug=DEBUG,
    )
    train_dataloader, validation_dataloader = get_dataloader(
        tokenizer, model, tokenized_datasets, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )
    optimizer = th.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    num_training_steps = N_EPOCHS * len(train_dataloader)
    lr_scheduler = transformers.get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=NUM_WARMUP_STEPS,
        num_training_steps=num_training_steps,
    )

    # original paper of SacreBLEU by Matt Post: https://arxiv.org/pdf/1804.08771.pdf
    # additional material: # https://www.youtube.com/watch?v=M05L1DhFqcw
    metric = datasets.load_metric("sacrebleu")

    print(
        20 * "---"
        + f"The model has {utils.count_parameters(model):,} trainable parameters "
        + 20 * "---"
    )

    # Training/Validation
    if IS_TRAINING:
        print(40 * "-" + " Start Training " + 40 * "-")
        for epoch in range(N_EPOCHS):
            start_time = time.time()
            # train loop
            train_loss = train_epoch(model, train_dataloader, optimizer, lr_scheduler, CLIP, device)
            end_time = time.time()
            epoch_mins, epoch_secs = utils.epoch_time(start_time, end_time)

            print(f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s")
            print(f"\t Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}")

            # validation loop (sacrebleu score)
            bleu_results = validation_epoch(model, validation_dataloader, metric, tokenizer, device)
            print(f"epoch {epoch}, SacreBLEU score: {bleu_results['score']:.2f}")

            # logging
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("SacreBLEU/valid", bleu_results["score"], epoch)
        writer.flush()
    else:
        # TODO: finish implementation
        # validation loop
        bleu_results = validation_epoch(model, validation_dataloader, metric, tokenizer, device)
        print(f"epoch {epoch}, SacreBLEU score: {bleu_results['score']:.2f}")


if __name__ == "__main__":
    main()
