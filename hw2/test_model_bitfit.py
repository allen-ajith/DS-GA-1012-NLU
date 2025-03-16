"""
Code for Problem 1 of HW 2.
"""
import pickle
import numpy as np

import evaluate
from datasets import load_dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, \
    Trainer, TrainingArguments

from train_model import preprocess_dataset

def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)


def init_tester(directory: str) -> Trainer:
    """
    Prolem 2b: Implement this function.

    Creates a Trainer object that will be used to test a fine-tuned
    model on the IMDb test set. The Trainer should fulfill the criteria
    listed in the problem set.

    :param directory: The directory where the model being tested is
        saved
    :return: A Trainer used for testing
    """

    model = BertForSequenceClassification.from_pretrained(directory)

    training_args = TrainingArguments(
        output_dir="test_results",
        per_device_eval_batch_size=8,
        do_train=False,
        do_eval=True,
        evaluation_strategy="no",
    )

    return Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics
    )


if __name__ == "__main__":  # Use this script to test your model
    model_name = "prajjwal1/bert-tiny"

    # Load IMDb dataset
    imdb = load_dataset("imdb")
    del imdb["train"]
    del imdb["unsupervised"]

    # Preprocess the dataset for the tester
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    imdb["test"] = preprocess_dataset(imdb["test"], tokenizer)

    # Set up tester
    tester = init_tester("path_to_your_best_model")

    # Test
    results = tester.predict(imdb["test"])
    with open("test_results_with_bitfit.p", "wb") as f:
        pickle.dump(results, f)
