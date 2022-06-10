import functools

import datasets
import sh
import torch as th
import transformers
from absl import app, flags
from torch.utils.tensorboard import SummaryWriter

from pre_processor.pre_processor import (_prepare_ds, _tokenize,
                                         convert_for_tokenizer, get_dataloader)
from utils import arg_parser

parser = arg_parser.create_parser()
args = parser.parse_args()
writer = SummaryWriter()

# trainer specific arguments
TRAIN = args.train
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
    # model initialization
    if IS_PRETRAINED:
        model = transformers.T5ForConditionalGeneration.from_pretrained(MODEL, torch_dtype="auto")       
    else:
        start_token_id = tokenizer.convert_tokens_to_ids(['<pad>'])[0] # see transformers/issues/16571
        config = transformers.T5Config(vocab_size=tokenizer.vocab_size, decoder_start_token_id=start_token_id)
        model = transformers.T5ForConditionalGeneration(config)
    return model


def train_epoch(model, train_dataloader, optimizer, CLIP):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(train_dataloader):
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

def validation_epoch(model, validation_dataloader, tokenizer):
    # evaluate bleu score
    model.eval()
    sacre_bleu_score = 0
    for i, batch in enumerate(validation_dataloader):
        src_ids = batch['src_ids']
        trg_ids = batch['trg_ids']
        pred_seq = model.generate(src_ids)

        trg_decoded = tokenizer.batch_decode(trg_ids, skip_special_tokens=True) # decoded trg sentences 
        pred_seq_decoded = tokenizer.batch_decode(pred_seq, skip_special_tokens=True) # decoded output translation 

        # hugging face on bleu score: https://www.youtube.com/watch?v=M05L1DhFqcw
        sacre_bleu = datasets.load_metric('sacrebleu') # 'bleu' also possible

        pred_list = [[sentence] for sentence in pred_seq_decoded]
        trg_list = [[sentence] for sentence in trg_decoded]
        sacre_bleu_score = sacre_bleu.compute(predictions=pred_list, references=trg_list)
        print('Batch BLEU Score: ', sacre_bleu_score)

def run_training():
    '''
    Training loop through all epochs. 
    '''
    for epoch in range(N_EPOCHS):
        train_loss = train_epoch(model, train_dataloader, optimizer, CLIP)


def evaluate_model():
    validation_epoch(model, validation_dataloader, tokenizer)

if __name__ == '__main__':
    tokenizer = transformers.T5Tokenizer.from_pretrained(MODEL)
    # Data Preprocessing
    train_ds, validation_ds, test_ds =_prepare_ds(tokenizer, number_of_training_samples=TRAINING_SAMPLES, seq_length=SEQ_LENGTH)
    train_dataloader, validation_dataloader, test_dataloader = get_dataloader(train_ds, validation_ds, test_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    model = build_model(tokenizer)
    optimizer = th.optim.Adam(model.parameters(), lr = LEARNING_RATE)
    criterion = th.nn.CrossEntropyLoss()

    # start training
    if TRAIN:
        run_training()
        evaluate_model()
    else:
        evaluate_model()
