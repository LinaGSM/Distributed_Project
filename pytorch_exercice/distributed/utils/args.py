import argparse

def parse_args(args_list=None):
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--batch_size",
        type=int,
        help='batch size used, default is 128.',
        default=128
    )

    if args_list:
        args = parser.parse_args(args_list)
    else:
        args = parser.parse_args()

    return args