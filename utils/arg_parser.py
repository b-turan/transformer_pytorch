import argparse


# fmt: off
def create_parser():
    """
    Arguments devided into Trainer, Model and Program specific arguments.
    """
    parser = argparse.ArgumentParser(description="Transformer Model for Translation")
    # Trainer args  (gpus, epochs etc.)
    parser.add_argument("-g", "--gpus", type=int, metavar="", help="Number of GPUS, (None for CPU)", default=1)
    parser.add_argument("--batch_size", type=int, metavar="", help="Batch Size", default=16)
    parser.add_argument("-lr", "--learning_rate", type=float, metavar="", help="Initial Learning Rate", default=2e-5)
    parser.add_argument("--num_warmup_steps", type=int, metavar="", help="Number of Warmup Steps", default= 0)
    parser.add_argument("-e", "--epochs", type=int, metavar="", help="Number of Epochs", default= 30)
    parser.add_argument("--n_samples", type=int, metavar="", help="Number of Samples", default= 20000)
    parser.add_argument("--momentum", type=float, metavar="", help="Momentum", default= .9)
    parser.add_argument("--clip", type=int, metavar="", help="Gradient Clipping", default= 1)
    parser.add_argument("--num_workers", type=int, metavar="", help="num_workers", default= 8)
    parser.add_argument('--train', action=argparse.BooleanOptionalAction)
    # Model specific arguments
    parser.add_argument('--is_pretrained', action=argparse.BooleanOptionalAction)
    parser.add_argument("--model", type=str, metavar="", help="Choice of Dataset", default="t5-small")
    parser.add_argument("--max_input_length", type=int, metavar="", help="Maximal Number of Tokens per Sentence in Input Sequences", default=64)
    parser.add_argument("--max_target_length", type=int, metavar="", help="Maximal Number of Tokens per Sentence in Target Sequences and generated tokens", default=64)
    # Program arguments (data_path, save_dir, etc.)
    parser.add_argument("--seed", type=int, metavar="", help="Seed Choice", default=42)
    parser.add_argument('--debug', action=argparse.BooleanOptionalAction)

    return parser
# fmt: on
