from huggingface_hub import login
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate
import pandas as pd


login()

label2id = {'left': 0, 'center': 1, 'right': 2}

dataset = load_dataset("csv", data_files="allsides_balanced_news_headlines-texts.csv")

#qbias_df = pd.DataFrame(dataset)
#print(qbias_df)

dataset = dataset.rename_column("bias_rating", "label")
dataset_filtered = dataset.filter(lambda example: example['text'] is not None and example['text'] != "")
#dataset_labeled = dataset_filtered['train'].align_labels_with_mapping(label2id, "label")

dataset_split = dataset_filtered['train'].train_test_split(test_size=0.25)

#print(dataset_split['train'].num_rows)
#print(dataset_split['test'].num_rows)

def map_label(example):
    if example['label'] == 'right':
        example['label'] = 2
    elif example['label'] == 'center':
        example['label'] = 1
    elif example['label'] == 'left':
        example['label'] = 0
    return example

dataset_mapped = dataset_split.map(map_label)

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

dataset_mapped = dataset_mapped.map(tokenize, batched=True)


small_train = dataset_mapped["train"].shuffle(seed=42).select(range(100))
small_eval = dataset_mapped["test"].shuffle(seed=42).select(range(100))


model = AutoModelForSequenceClassification.from_pretrained("bucketresearch/politicalBiasBERT", num_labels=3)

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # convert the logits to their predicted class
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    eval_strategy="epoch",
    push_to_hub=True,
    output_dir="qbias_model",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_mapped["train"],
    eval_dataset=dataset_mapped["test"],
    compute_metrics=compute_metrics,
)
trainer.train()
trainer.evaluate()

trainer.push_to_hub("Halfbendy/QbiasBERT")