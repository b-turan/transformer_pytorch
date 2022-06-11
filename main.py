import math
import time

import datasets
import sh
import torch as th
import transformers
from absl import app, flags
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from pre_processor.pre_processor import _prepare_ds, get_dataloader
from utils import arg_parser, utils

parser = arg_parser.create_parser()
args = parser.parse_args()
writer = SummaryWriter()

# trainer specific arguments
IS_TRAINING = args.train
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
N_EPOCHS = args.epochs
CLIP = args.clip
NUM_WORKERS = args.num_workers
MOMENTUM = args.momentum
TRAINING_SAMPLES = args.training_samples
# model specific arguments
MODEL = args.model
SEQ_LENGTH = args.seq_length
IS_PRETRAINED = args.is_pretrained


# remove and recreate logs folder for development purposes
sh.rm('-r', '-f', 'logs/')
sh.mkdir('logs')

device = th.device('cuda' if th.cuda.is_available() else 'cpu')
print("Running on Device:", device)


def build_model(tokenizer):
    '''
    Returns T5 model (pretrained or randomly initialized)
    '''
    if IS_PRETRAINED:
        model = transformers.T5ForConditionalGeneration.from_pretrained(MODEL, torch_dtype="auto")       
    else:
        start_token_id = tokenizer.convert_tokens_to_ids(['<pad>'])[0] # see transformers/issues/16571
        config = transformers.T5Config(vocab_size=tokenizer.vocab_size, decoder_start_token_id=start_token_id)
        model = transformers.T5ForConditionalGeneration(config)
    return model


def train_epoch(model, train_dataloader, optimizer, CLIP):
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
            src_ids = batch['src_ids']
            trg_ids = batch['trg_ids']
            attention_mask = batch['attention_mask']
            optimizer.zero_grad()
            loss = model(input_ids=src_ids, attention_mask=attention_mask, labels=trg_ids).loss     
            loss.backward()
            th.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()
            epoch_loss += loss.item()
    return epoch_loss / len(train_dataloader)

def validation_epoch(model, dataloader, tokenizer):
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
            src_ids = batch['src_ids']
            trg_ids = batch['trg_ids']
            attention_mask = batch['attention_mask']
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
    model.eval()
    sacre_bleu_score = 0
    for batch in tqdm(dataloader):
        src_ids = batch['src_ids']
        trg_ids = batch['trg_ids']
        pred_seq = model.generate(src_ids
                                # do_sample=True, 
                                # top_p=0.84, 
                                # top_k=100, 
                                # max_length=FLAGS.seq_length
                                ) # encoded translation of src sentences
        trg_decoded = tokenizer.batch_decode(trg_ids, skip_special_tokens=True) # decoded trg sentences 
        pred_seq_decoded = tokenizer.batch_decode(pred_seq, skip_special_tokens=True) # decoded output translation 

        # hugging face on bleu score: https://www.youtube.com/watch?v=M05L1DhFqcw
        sacre_bleu = datasets.load_metric('sacrebleu') # 'bleu' also possible
        pred_list = [[sentence] for sentence in pred_seq_decoded]
        trg_list = [[sentence] for sentence in trg_decoded]
        sacre_bleu_score += sacre_bleu.compute(predictions=pred_list, references=trg_list)['score']
    return sacre_bleu_score/len(dataloader)

if __name__ == '__main__':
    tokenizer = transformers.T5Tokenizer.from_pretrained(MODEL)
    # Data Preprocessing
    train_ds, validation_ds, test_ds =_prepare_ds(tokenizer, number_of_training_samples=TRAINING_SAMPLES, seq_length=SEQ_LENGTH)
    train_dataloader, validation_dataloader, test_dataloader = get_dataloader(train_ds, validation_ds, test_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    # Model Initialization
    model = build_model(tokenizer)
    optimizer = th.optim.Adam(model.parameters(), lr = LEARNING_RATE)
    print(40*'-' + 'Calculate Number of Model Parameters' + 40*'-')
    print(f'The model has {utils.count_parameters(model):,} trainable parameters')
    # Training/Validation
    if IS_TRAINING:
        print(40*'-'+'Start Training'+40*'-')
        for epoch in range(N_EPOCHS):
            start_time = time.time()

            train_loss = train_epoch(model, train_dataloader, optimizer, CLIP)
            valid_loss = validation_epoch(model, validation_dataloader, tokenizer)
            
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

    else:
        valid_loss = validation_epoch(model, validation_dataloader, tokenizer)
        # Bleu Score
        print(40*'-' + 'Calculate Bleu Score ' + 40*'-')
        sacre_bleu_score = get_bleu_score(model, validation_dataloader, tokenizer)
        print(f'\t SacreBleu Score: {sacre_bleu_score:.3f}')
