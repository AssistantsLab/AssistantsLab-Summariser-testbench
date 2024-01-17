import argparse
import evaluation

valid_args = [
    "--test_bleu",
    "--test_rouge",
    "--output_to_file",
    "--simple",
    "--models"
]


def parse_args():
    parser = argparse.ArgumentParser(description="Description of your script.")
    parser.add_argument("--test_bleu", action="store_true", help="Generate BLEU test scores")
    parser.add_argument("--test_rouge", action="store_true", help="Generate Rouge-L test scores")
    parser.add_argument("--output_to_file", action="store_true", help="Output results to a .csv file")
    parser.add_argument("--simple", action="store_true", help="Start a simple test with all models and both BLEU and Rouge-L scores")
    parser.add_argument("--models", nargs="+", help="Specify one or more models")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not any(vars(args).values()):
        print("No arguments provided! This file is meant to be run with arguments.")
        print("Valid arguments:")
        print("--test_bleu (will generate BLEU test scores)")
        print("--test_rouge (will generate Rouge-L test scores)")
        print("--output_to_file (will output the results to a .csv file)")
        print("--simple (Starts a simple test with all models and both BLEU and Rouge-L scores)")
        print("--models (Specify one or more models)")
        exit(1)

    for arg in valid_args:
        if getattr(args, arg[2:]):
            models = args.models if args.models else [
                "facebook/bart-large-cnn",
                "google/pegasus-xsum",
                "knkarthick/MEETING-SUMMARY-BART-LARGE-XSUM-SAMSUM-DIALOGSUM-AMI",
                "sshleifer/distilbart-cnn-12-6",
                "slauw87/bart_summarisation",
            ]

            if args.simple:
                args.test_bleu = True
                args.test_rouge = True
                args.output_to_file = False

            evaluation_class = evaluation.Evaluation(
                test_bleu=args.test_bleu,
                test_rouge=args.test_rouge,
                output_to_file=args.output_to_file,
                models=models
            )

            evaluation_class.test_models()
            break
    else:
        print("No valid arguments provided!")
        exit(1)
