import datasets
import functools
import torch as th

def convert_for_tokenizer(ds):
    ''' Converts dataset to required format for tokenization '''
    new_ds = {} # re-init dictionary
    new_ds['de'] = [translation['de'] for translation in ds['translation']] # collect src sentences
    new_ds['en'] = [translation['en'] for translation in ds['translation']] # collect trg sentences
    new_ds = datasets.Dataset.from_dict(new_ds)
    return new_ds


def _tokenize(x, tokenizer, seq_length):
    prefix = "translate English to German: "
    src_encoding = tokenizer.batch_encode_plus(
            [prefix + sentence for sentence in x['en']], 
            max_length=seq_length, 
            padding="longest",
            truncation=True,
            )
    x['src_ids'] = src_encoding.input_ids
    x['attention_mask'] = src_encoding.attention_mask
    x['trg_ids'] = tokenizer.batch_encode_plus(
            x['de'], 
            max_length=seq_length, 
            padding="longest",
            truncation=True,
            )['input_ids']
    return x
    

def _prepare_ds(tokenizer, number_of_training_samples, seq_length):
        # available wmt16 language pairs: ['cs-en', 'de-en', 'fi-en', 'ro-en', 'ru-en', 'tr-en']
        ds = datasets.load_dataset('wmt16', 'de-en') # {train, validation, test}
        print(f"Train Dataset is cut to {number_of_training_samples} samples for development purposes! Remove cutting for full training.")
        train_ds, validation_ds, test_ds = ds['train'][:number_of_training_samples], ds['validation'], ds['test']
        train_ds, validation_ds, test_ds = map(convert_for_tokenizer, (train_ds, validation_ds, test_ds))
        # add tokenized columns to dataset
        train_ds = train_ds.map(functools.partial(_tokenize, tokenizer=tokenizer, seq_length=seq_length), batched=True)
        validation_ds = validation_ds.map(functools.partial(_tokenize, tokenizer=tokenizer, seq_length=seq_length), batched=True)
        test_ds = test_ds.map(functools.partial(_tokenize, tokenizer=tokenizer, seq_length=seq_length), batched=True)
        # convert columns to torch tensors
        train_ds.set_format(type='torch', columns=['src_ids', 'trg_ids', 'attention_mask'])
        validation_ds.set_format(type='torch', columns=['src_ids', 'trg_ids', 'attention_mask'])
        test_ds.set_format(type='torch', columns=['src_ids', 'trg_ids', 'attention_mask'])    
        return train_ds, validation_ds, test_ds 

def get_dataloader(train_ds, validation_ds, test_ds, batch_size, num_workers):
        train_dataloader = th.utils.data.DataLoader(
                train_ds,
                batch_size=batch_size,
                drop_last=True,
                shuffle=True,
                num_workers=num_workers
        )
        validation_dataloader = th.utils.data.DataLoader(
                validation_ds,
                batch_size=batch_size,
                drop_last=False,
                shuffle=True,
                num_workers=num_workers
        )
        validation_dataloader = th.utils.data.DataLoader(
                test_ds,
                batch_size=batch_size,
                drop_last=False,
                shuffle=True,
                num_workers=num_workers
        )
        return train_dataloader, validation_dataloader, test_ds

