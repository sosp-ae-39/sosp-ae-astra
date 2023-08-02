# Datasets

## Alpaca dataset

`alpaca_opt_text_completion.pkl` contains the tokenized version of the [Alpaca dataset](https://github.com/tatsu-lab/stanford_alpaca).


## ShareGPT dataset

`sharegpt_opt_text_completion_length.pkl` is a dataset collected from [ShareGPT](https://sharegpt.com) in March 2023.
While the original dataset includes multi-round conversations, we transformed it into a text completion dataset by extracting the first round of each conversation, i.e., the user's first input and ChatGPT's first response.
We tokenized the input and output pairs using the OPT tokenizer.

To protect against potential licensing concerns, the dataset only contains the length distributions of the tokenized inputs and outputs, but not their specific values.
In the experiment with this dataset, all token ids are set to 0.
