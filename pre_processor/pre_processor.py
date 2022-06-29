import functools

import datasets
import torch as th
import transformers


def preprocess_function(examples, tokenizer, max_input_length, max_target_length):
    prefix = "translate English to German: "
    inputs = [prefix + ex["en"] for ex in examples["translation"]]
    targets = [ex["de"] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Set up the tokenizer for targets
    with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def tokenize_datasets(tokenizer, n_samples, max_input_length, max_target_length, debug):
    '''
    Tokenizes WMT16 dataset for dataloader. 
    Available WMT16 language pairs: ['cs-en', 'de-en', 'fi-en', 'ro-en', 'ru-en', 'tr-en']
    ------------------------------------------------------------------------------------------------------------
    tokenizer (transformers.tokenizer): Pretrained Tokenizer
    n_samples (int): Number of Samples for Training and Validation (in debug mode)
    max_input_length (int): Maximal length of tokens per sentence in the input sequence
    max_target_length (int): Maximal length of tokens per sentence in the target sequence
    debug (boolean): Debugging mode
    ------------------------------------------------------------------------------------------------------------
    returns tokenized dataset as dataset dict {train, validation, test}
    ------------------------------------------------------------------------------------------------------------
    TODO(b-turan): Generalize for other Datasets, 
    TODO(b-turan): Attention: tokenization of full dataset takes long time
    '''    
    print(40*'-' + ' ... Loading Datasets ... ' + 40*'-')
    split_ds = datasets.load_dataset('wmt16', 'de-en') # {train, validation, test}
    
    if debug:
            # reduce dataset for debug purposes
            split_ds = datasets.Dataset.from_dict(split_ds['train'][:n_samples]).train_test_split(test_size=0.1)
            split_ds['validation'] = split_ds.pop('test')
    
    print(40*'-' + ' ... Tokenizing Datasets ... ' + 40*'-')
    tokenized_datasets = split_ds.map(
            functools.partial(
                    preprocess_function, 
                    tokenizer=tokenizer, 
                    max_input_length=max_input_length, 
                    max_target_length=max_target_length), 
                    batched=True, remove_columns=split_ds["train"].column_names
    )
    return tokenized_datasets


def get_dataloader(tokenizer, model, tokenized_datasets, batch_size, num_workers):
    ''' Returns train, validation '''
    data_collator = transformers.DataCollatorForSeq2Seq(tokenizer, model=model)
    train_dataloader = th.utils.data.DataLoader(
            tokenized_datasets['train'],
            batch_size=batch_size,
            drop_last=True,
            shuffle=True,
            collate_fn=data_collator,
            num_workers=num_workers
    )
    validation_dataloader = th.utils.data.DataLoader(
            tokenized_datasets['validation'],
            batch_size=batch_size,
            collate_fn=data_collator,
            num_workers=num_workers
    )
    return train_dataloader, validation_dataloader

