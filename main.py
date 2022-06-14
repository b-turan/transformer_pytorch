import math
import time

import datasets
import torch as th
import transformers
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from pre_processor.pre_processor import _prepare_ds, get_dataloader
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
    model = build_model(MODEL, IS_PRETRAINED)
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

            train_loss = train_epoch(model, train_dataloader, optimizer, lr_scheduler, CLIP)
            valid_loss = validation_epoch(model, validation_dataloader)
            
            end_time = time.time()
            epoch_mins, epoch_secs = utils.epoch_time(start_time, end_time)
            
            print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\t Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

            # Bleu Score
            print(40*'-' + 'Calculate Bleu Score ' + 40*'-')
            sacre_bleu_score = get_bleu_score(model, validation_dataloader, tokenizer)
            print(f'\t SacreBleu Score: {sacre_bleu_score:.3f}')

            # logging
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/valid", valid_loss, epoch)
            writer.add_scalar("BLEU Score", sacre_bleu_score, epoch)
        writer.flush()
    else:
        valid_loss = validation_epoch(model, validation_dataloader, tokenizer)
        # Bleu Score
        print(40*'-' + 'Calculate Bleu Score ' + 40*'-')
        sacre_bleu_score = get_bleu_score(model, validation_dataloader, tokenizer)
        print(f'\t SacreBleu Score: {sacre_bleu_score:.3f}')


def build_model(model_name, IS_PRETRAINED):
    '''
    Returns T5 model (pretrained or randomly initialized)
    '''
    if IS_PRETRAINED:
        # TODO: check alternative loading with AutoModel.from_pretrained()
        model = transformers.T5ForConditionalGeneration.from_pretrained(model_name, 
                                                                        torch_dtype="auto"
                                                                        ).to(device)
    else:
        config = transformers.AutoConfig.from_pretrained(model_name) # see transformers/issues/14674
        model = transformers.T5ForConditionalGeneration(config).to(device)
    return model


def train_epoch(model, train_dataloader, optimizer, lr_scheduler, CLIP):
    '''
    Trains model on the entire dataset for one epoch.
    ------------------------------------
    model (nn.model): Torch model
    train_dataloader (torch.dataloader): Dataloader
    optimizer (th.optim): Optimizer
    CLIP (int): Gradient Clipping
    ------------------------------------
    returns average epoch loss
    '''
    model.train()
    epoch_loss = 0
    for batch in tqdm(train_dataloader):
            src_ids = batch['src_ids'].to(device)
            trg_ids = batch['trg_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            loss = model(input_ids=src_ids, attention_mask=attention_mask, labels=trg_ids).loss     
            loss.backward()
            th.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            print(lr_scheduler.get_last_lr())
    return epoch_loss / len(train_dataloader)


def validation_epoch(model, dataloader):
    '''
    Evaluates model on the entire dataset.
    ------------------------------------
    model (nn.model): Torch model
    dataloader (torch.dataloader): Dataloader
    tokenizer (transformer.tokenizer): Tokenizer
    ------------------------------------
    returns average epoch validation loss
    '''
    model.eval()
    epoch_loss = 0
    with th.no_grad():
        for i, batch in enumerate(dataloader):
            src_ids = batch['src_ids'].to(device)
            trg_ids = batch['trg_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            loss = model(input_ids=src_ids, attention_mask=attention_mask, labels=trg_ids).loss     
            epoch_loss += loss.item()
    return epoch_loss / len(dataloader)


def get_bleu_score(model, dataloader, tokenizer):
    '''
    Calculates sacrebleu score of given model on entire dataset.
    ------------------------------------
    model (nn.model): Torch model
    dataloader (torch.dataloader): Dataloader
    tokenizer (transformer.tokenizer): Tokenizer
    ------------------------------------
    Returns sacre_bleu_score
    '''
    # evaluate bleu score  
    # TODO: can tokenizer be used with cuda?
    model.eval()
    sacre_bleu_score = 0
    for batch in tqdm(dataloader):
        src_ids = batch['src_ids'].to(device)
        trg_ids = batch['trg_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        pred_seq = model.generate(src_ids,
                                attention_mask = attention_mask,
                                # do_sample=True, 
                                # top_p=0.84, 
                                # top_k=100, 
                                # max_length=SEQ_LENGTH
                                ) # encoded translation of src sentences
        trg_decoded = tokenizer.batch_decode(trg_ids, skip_special_tokens=True) # decoded trg sentences 
        pred_seq_decoded = tokenizer.batch_decode(pred_seq, skip_special_tokens=True) # decoded output translation 
        
        # original paper of SacreBLEU by Matt Post:
        # https://arxiv.org/pdf/1804.08771.pdf
        # additional material: 
        # hugging face on (sacre)BLEU score https://www.youtube.com/watch?v=M05L1DhFqcw
        sacre_bleu = datasets.load_metric('sacrebleu') # 'bleu' as input also possible
        pred_list = [[sentence] for sentence in pred_seq_decoded]
        trg_list = [[sentence] for sentence in trg_decoded]
        sacre_bleu_score += sacre_bleu.compute(predictions=pred_list, references=trg_list)['score']
    return sacre_bleu_score/len(dataloader)


if __name__ == '__main__':
    main()
