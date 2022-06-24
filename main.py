import math
import time

import datasets
import torch as th
import transformers
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import Trainer, TrainingArguments

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
        # TODO: remove train and validation routine and add HF Trainer
        training_args = TrainingArguments(output_dir='./',
                                        num_train_epochs=N_EPOCHS,
                                        learning_rate=LEARNING_RATE,
                                        per_device_train_batch_size=BATCH_SIZE,
                                        per_device_eval_batch_size=BATCH_SIZE,
                                        weight_decay=0.01,
                                        evaluation_strategy="epoch",
                                        disable_tqdm=False,
                                        log_level="error",
                                        max_grad_norm=CLIP)

        trainer = Trainer(model=model, args=training_args,
                        train_dataset=train_ds,
                        eval_dataset=validation_ds,
                        tokenizer=tokenizer)
        trainer.train()


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
            input_ids = batch['input_ids'].to(device)
            trg_ids = batch['trg_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=trg_ids).loss     
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
            input_ids = batch['input_ids'].to(device)
            trg_ids = batch['trg_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=trg_ids).loss     
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
        input_ids = batch['input_ids'].to(device)
        trg_ids = batch['trg_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        pred_seq = model.generate(input_ids,
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
