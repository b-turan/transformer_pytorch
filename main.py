import math
import time

import datasets
import torch as th
import transformers
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from pre_processor.pre_processor import _prepare_ds, get_dataloader
from training_func.epoch import get_bleu_score, train_epoch, validation_epoch
from utils import arg_parser, utils

device = th.device('cuda' if th.cuda.is_available() else 'cpu')
print("Running on Device:", device)


def main():
    parser = arg_parser.create_parser()
    args = parser.parse_args()
    # initialize tensorboard
    writer = SummaryWriter() 
    
    # trainer specific arguments
    IS_TRAINING = args.train
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    NUM_WARMUP_STEPS = args.num_warmup_steps
    N_EPOCHS = args.epochs
    CLIP = args.clip
    NUM_WORKERS = args.num_workers
    MOMENTUM = args.momentum
    TRAINING_SAMPLES = args.training_samples
    # model specific arguments
    MODEL = args.model
    SEQ_LENGTH = args.seq_length
    IS_PRETRAINED = args.is_pretrained

    # initialize pretrained tokenizer
    tokenizer = transformers.T5Tokenizer.from_pretrained(MODEL) 
    # data pre-processing
    train_ds, validation_ds, test_ds =_prepare_ds(tokenizer, number_of_training_samples=TRAINING_SAMPLES, 
                                                seq_length=SEQ_LENGTH)
    train_dataloader, validation_dataloader, _ = get_dataloader(train_ds, validation_ds, test_ds, 
                                                                batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    # Model Initialization
    model = utils.build_model(MODEL, IS_PRETRAINED, device)
    optimizer = th.optim.Adam(model.parameters(), lr = LEARNING_RATE)
    num_training_steps = N_EPOCHS * len(train_dataloader)
    lr_scheduler = transformers.get_scheduler(name="linear", optimizer=optimizer, 
                                              num_warmup_steps=NUM_WARMUP_STEPS, 
                                              num_training_steps=num_training_steps)
    
    print(40*'-' + 'Model got initialized' + 40*'-')
    print(f'\t The model has {utils.count_parameters(model):,} trainable parameters')

    # Training/Validation
    if IS_TRAINING:
        print(40*'-'+'Start Training'+40*'-')
        for epoch in range(N_EPOCHS):
            start_time = time.time()

            train_loss = train_epoch(model, train_dataloader, optimizer, lr_scheduler, CLIP, tokenizer, device)
            valid_loss = validation_epoch(model, validation_dataloader, device)
            
            end_time = time.time()
            epoch_mins, epoch_secs = utils.epoch_time(start_time, end_time)
            
            print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\t Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

            # Bleu Score
            print(40*'-' + 'Calculate Bleu Score ' + 40*'-')
            sacre_bleu_score = get_bleu_score(model, validation_dataloader, tokenizer, device)
            print(f'\t SacreBleu Score: {sacre_bleu_score:.3f}')

            # logging
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/valid", valid_loss, epoch)
            writer.add_scalar("BLEU Score", sacre_bleu_score, epoch)
        writer.flush()
    else:
        valid_loss = validation_epoch(model, validation_dataloader, tokenizer, device)
        # Bleu Score
        print(40*'-' + 'Calculate Bleu Score ' + 40*'-')
        sacre_bleu_score = get_bleu_score(model, validation_dataloader, tokenizer, device)
        print(f'\t SacreBleu Score: {sacre_bleu_score:.3f}')


if __name__ == '__main__':
    main()
