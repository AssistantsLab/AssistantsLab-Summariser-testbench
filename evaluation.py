import torch
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import utils
import statistics


def summarization_pipeline(model_name_or_path, max_length=50, min_length=1, length_penalty=2.0, num_beams=4):
    """
    Initializes a summarization pipeline using the specified pre-trained model for sequence-to-sequence language modeling.

    Parameters:
    - model_name_or_path (str): The name or path of the pre-trained model.
    - max_length (int): The maximum length of the generated summaries (default is 50).
    - min_length (int): The minimum length of the generated summaries (default is 1).
    - length_penalty (float): Length penalty for the generated summaries (default is 2.0).
    - num_beams (int): Number of beams used for beam search during summarization (default is 4).

    Returns:
    - summarizer: A summarization pipeline configured with the specified parameters.
    """
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    summarizer = pipeline(
        task="summarization",
        model=model,
        tokenizer=tokenizer,
        framework="pt",
        max_length=max_length,
        min_length=min_length,
        length_penalty=length_penalty,
        num_beams=num_beams
    )

    return summarizer


class Evaluation:

    def __init__(self, test_bleu: bool, test_rouge: bool, output_to_file: bool, models: list):
        if not test_bleu and not test_rouge:
            print('None of the testing metrics are enabled!')
            exit(0)

        self.test_bleu = test_bleu
        self.test_rouge = test_rouge
        self.output_to_file = output_to_file
        self.models = models

        # initialising empty dicts to ensure they exist
        self.generated_samples = {}
        self.bleu_scores = {}
        self.rouge_scores = {}

    def test_models(self):
        # generating samples
        for model in self.models:
            # init empty dicts to ensure we can just append later
            self.generated_samples[model] = []
            self.bleu_scores[model] = []
            self.rouge_scores[model] = []

            # generating samples
            print('Generating samples for model: ' + model)
            self.generate_samples(model)

            # calculating bleu and/or rouge
            if self.test_bleu:
                print('Generating BLEU scores for model: ' + model)
                self.calculate_bleu(model)
            if self.test_rouge:
                print('Generating Rouge scores for model: ' + model)
                self.calculate_rouge(model)

        print("\n-==+==--")
        print("\nFinished generating samples and evaluation scores")
        print("\n-==+==--\n")

        # Compute combined scores and sort models
        combined_scores = {}
        for model in self.models:
            rouge_scores = self.rouge_scores[model]
            bleu_scores = self.bleu_scores[model]
            combined_scores[model] = [(rouge * 0.6 + bleu * 0.4) for rouge, bleu in zip(rouge_scores, bleu_scores)]

        sorted_models = sorted(self.models, key=lambda model: statistics.mean(combined_scores[model]), reverse=True)

        if self.output_to_file:
            raise NotImplementedError
        else:
            for model in sorted_models:
                print("\n" + model + " scores:")

                if self.test_bleu and self.test_rouge:
                    print('ID | BLEU | ROUGE | Combined Score')
                    for index, (bleu, rouge, combined) in enumerate(
                            zip(self.bleu_scores[model], self.rouge_scores[model], combined_scores[model])):
                        print(f'{index} | {bleu} | {rouge} | {combined}')
                    print("\nAverage bleu: " + str(statistics.mean(self.bleu_scores[model])))
                    print("Average rouge: " + str(statistics.mean(self.rouge_scores[model])))
                    print("Average combined score: " + str(statistics.mean(combined_scores[model])))

                elif self.test_bleu:
                    print('ID | BLEU')
                    for index, (bleu) in enumerate(self.bleu_scores[model]):
                        print(f'{index} | {bleu}')
                    print("\nAverage bleu: " + str(statistics.mean(self.bleu_scores[model])))

                elif self.test_rouge:
                    print('ID | ROUGE')
                    for index, (rouge) in enumerate(self.rouge_scores[model]):
                        print(f'{index} | {rouge}')
                    print("\nAverage rouge: " + str(statistics.mean(self.rouge_scores[model])))

    def generate_samples(self, model):
        # get test values
        test_values = utils.get_values()

        # get summariser
        summariser = summarization_pipeline(
            model_name_or_path=model,
            max_length=50,
            min_length=1,
            length_penalty=2.0,
            num_beams=4
        )

        # iterate over samples
        for i in range(len(test_values)):
            # Ensuring we dont generate a summary if the length of the text is more than the model can handle
            if torch.numel(summariser.tokenizer(test_values[i][0], return_tensors="pt").input_ids) > \
                    summariser.tokenizer.model_max_length:
                continue

            summarised_text = summariser(test_values[i][0])[0]['summary_text']
            reference_text = test_values[i][1]

            print('  Text: ' + test_values[i][0])
            print('  Summarised: ' + summarised_text)
            print('  Sample: ' + str(i + 1) + " / " + str(len(test_values)))

            self.generated_samples[model].append((summarised_text, reference_text))

    def calculate_bleu(self, model):
        samples = self.generated_samples[model]

        for i in range(len(samples)):
            bleu_score = sentence_bleu(
                [samples[i][1].split()], samples[i][0].split(),
                smoothing_function=SmoothingFunction().method1
            )

            print('  sample: ' + str(i + 1) + " / " + str(len(samples)))

            self.bleu_scores[model].append(bleu_score)

    def calculate_rouge(self, model):
        samples = self.generated_samples[model]

        for i in range(len(samples)):
            rouge = Rouge()
            scores = rouge.get_scores(samples[i][0], samples[i][1])
            rouge_score = scores[0]['rouge-l']['f']

            print('  sample: ' + str(i + 1) + " / " + str(len(samples)))

            self.rouge_scores[model].append(rouge_score)
