import datasets
import torch as th
from tqdm import tqdm


def train_epoch(model, train_dataloader, optimizer, lr_scheduler, CLIP, tokenizer, device):
    '''
    Trains model on the entire dataset for one epoch.
    ------------------------------------
    model (nn.model): Torch model
    train_dataloader (torch.dataloader): Dataloader
    optimizer (th.optim): Optimizer
    CLIP (int): Gradient Clipping    
    device (torch.device): cuda or cpu
    ------------------------------------
    returns average epoch loss
    '''
    model.train()
    epoch_loss = 0
    for batch in tqdm(train_dataloader):
            src_ids = batch['src_ids'].to(device)
            trg_ids = batch['trg_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # trg_ids = th.tensor(trg_ids)

            trg_ids[trg_ids == tokenizer.pad_token_id] = -100
            loss = model(input_ids=src_ids, attention_mask=attention_mask, labels=trg_ids).loss     
            loss.backward()
            th.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
    return epoch_loss / len(train_dataloader)


def validation_epoch(model, dataloader, device):
    '''
    Evaluates model on the entire dataset.
    ------------------------------------
    model (nn.model): Torch model
    dataloader (torch.dataloader): Dataloader
    tokenizer (transformer.tokenizer): Tokenizer
    device (torch.device): cuda or cpu
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


def get_bleu_score(model, dataloader, tokenizer, device):
    '''
    Calculates sacrebleu score of given model on entire dataset.
    ------------------------------------
    model (nn.model): Torch model
    dataloader (torch.dataloader): Dataloader
    tokenizer (transformer.tokenizer): Tokenizer
    device (torch.device): cuda or cpu
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
                                # attention_mask = attention_mask,
                                # do_sample=True, 
                                # top_p=0.84, 
                                # top_k=100, 
                                # max_length=32
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

