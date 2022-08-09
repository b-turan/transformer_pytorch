# transformer_pytorch

## Masterplan
1. Train transformer model on translation tasks for fixed number of epochs
2. Apply pruning algorithms
3. Analyze BLEU score before and after pruning



## To Do
* [x] Check Neural [Machine Translation Tutorial](https://huggingface.co/course/chapter7/4?fw=pt) by HuggingFace
	- [x] Integrate t5-small (66M parameters) instead of MarianMT 
	- [x] Integrate WMT16 (en2de) instead of KDE4 (en2fr) dataset
	- [x] running on small fraction of original WMT16 dataset due to hardware limitations
	- [x] Settings
		- train_size=0.3*0.9 of original training size, 
		- valid_size=0.3*0.1 of original training_size
		- max_input_length = 64
		- max_target_length = 64
		- batch_size=16
		- num_train_epochs = 30
		- generated tokens max_length=64
		- pad_index=-100
		- lr=2e-5
	- [x] use accelerator 
	- [x] run 30 epochs on wmt16 and log sacrebleu scores on tensorboard -> sacrebleu score > 21
	- [ ] check performance on full dataset utilizing GPU-Cluster

* [x] Synchronize own implementation with tutorial 
	- [x] integrate tokenize_as_target method
	- [x] use accelerator 
	- [x] pad_index=-100
	- [x] sychronize with above settings
	- [x] run for 30 epochs -> sacrebleu score > 21

* [ ] If above is successful: Prepare Training for 4x NVIDIA A100 80GB GPUs
    - [ ] Find paper which pretrains on WMT16
    - [ ] Define appropriate training, evaluation and test dataset
    - [ ] Define appropriate training routine, i.e., warmup, learning_rate, lr_scheduler, batch_size, epochs etc.
