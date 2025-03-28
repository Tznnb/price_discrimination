# main.py
import argparse
import sys

from examples.example_1_1 import run_example_1_1
from examples.example_1_2 import run_example_1_2


def main():
    parser = argparse.ArgumentParser(description='Price Discrimination Experiments')
    parser.add_argument('--example', type=int, choices=[1, 2],
                        help='Run example from paper (1 or 2)')
    parser.add_argument('--no-plots', action='store_true',
                        help='Disable plots')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output')

    args = parser.parse_args()

    if args.example == 1:
        run_example_1_1(show_plots=not args.no_plots)
    elif args.example == 2:
        run_example_1_2(show_plots=not args.no_plots, debug=args.debug)
    else:
        print("Please specify which example to run: --example 1 or --example 2")
        sys.exit(1)


if __name__ == "__main__":
    main()