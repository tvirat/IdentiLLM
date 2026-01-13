"""
Your file should contain a function `predict_all` that takes
in the name of a CSV file, and returns a list of predictions.

Your `pred.py` script can use different methods to process the input
data, but the format of the input it takes and the output your script produces should be the same.

Here's an example of how your script may be used in our test file:

    from example_pred import predict_all
    predict_all("example_test_set.csv")
"""

import numpy as np
import pandas as pd
import re

VOCAB = ['about', 'ai', 'also', 'an', 'and', 'another', 'answer', 'answers',
         'are', 'as', 'ask', 'asked', 'asking', 'at', 'be', 'because',
         'but', 'by', 'can', 'check', 'code', 'coding', 'complex',
         'concepts', 'correct', 'data', 'do', 'don', 'explain', 'for',
         'from', 'gave', 'give', 'gives', 'good', 'google', 'have', 'help',
         'ideas', 'if', 'in', 'information', 'is', 'it', 'its', 'just',
         'like', 'look', 'make', 'math', 'me', 'model', 'more', 'my',
         'myself', 'never', 'not', 'of', 'often', 'on', 'one', 'or',
         'other', 'problem', 'problems', 'question', 'questions',
         'response', 'responses', 'same', 'search', 'see', 'so', 'some',
         'sometimes', 'sources', 'suboptimal', 'tasks', 'that', 'the',
         'this', 'through', 'to', 'up', 'use', 'used', 'using', 'usually',
         'verify', 'very', 'was', 'what', 'when', 'which', 'will', 'with',
         'work', 'would', 'writing', 'wrong']

FEATURES = ['academic_scale',
            'Explaining complex concepts simply',
            'Drafting professional text ',
            'suboptimal_scale',
            'to',
            'Converting content between formats ',
            'verify_scale',
            'Writing or debugging code',
            'the',
            'concepts',
            'it',
            'and',
            'sub_Math computations',
            'of',
            'another',
            'model',
            'ask',
            'was',
            'Writing or editing essays/reports',
            'for',
            'google',
            'never',
            'coding',
            'sub_Writing or debugging code',
            'wrong',
            'ref_scale',
            'that',
            'usually',
            'answer',
            'my',
            'code',
            'math',
            'if',
            'or',
            'me',
            'search',
            'used',
            'as',
            'which',
            'on',
            'in',
            'gave',
            'questions',
            'about',
            'question',
            'Brainstorming or generating creative ideas',
            'this',
            'work',
            'would',
            'help']

ID_COL = "student_id"
TARGET = "label"

TEXT_COLS = [
    "In your own words, what kinds of tasks would you use this model for?",
    "Which types of tasks do you feel this model handles best? (Select all that apply.)",
    "For which types of tasks do you feel this model tends to give suboptimal responses? (Select all that apply.)",
    "Think of one task where this model gave you a suboptimal response. What did the response look like, and why did you find it suboptimal?",
    "When you verify a response from this model, how do you usually go about it?"
]

LIKERT_COLS = [
    "How likely are you to use this model for academic tasks?",
    "Based on your experience, how often has this model given you a response that felt suboptimal?",
    "How often do you expect this model to provide responses with references or supporting evidence?",
    "How often do you verify this model's responses?"
]

CHOICES = ['Brainstorming or generating creative ideas', 'Converting content between formats ',
           'Data processing or analysis', 'Drafting professional text ', 'Explaining complex concepts simply',
           'Math computations', 'Writing or debugging code', 'Writing or editing essays/reports']


def reformat_rename(df):
    # Rename columns
    new_names = ["student_id", "tasks_open", "academic_scale", "task_types",
                 "suboptimal_scale","suboptimal_types",
                 "suboptimal_open", "ref_scale", "verify_scale","verify_open"]
    df.columns = new_names

    # Remove parantheses in multiple select options. This is to prepare for the next splitting step
    df['task_types'] = df['task_types'].str.replace(r'\([^)]*\)', '', regex=True)
    df['suboptimal_types'] = df['suboptimal_types'].str.replace(r'\([^)]*\)', '', regex=True)

    # Split task_types into binary variables
    binary_df = df['task_types'].str.get_dummies(sep=',')
    df = pd.concat([df, binary_df], axis=1)
    # Split suboptimal_types into binary variables
    binary_df = df['suboptimal_types'].str.get_dummies(sep=',')
    df = pd.concat([df, binary_df], axis=1)
    df = df.drop(columns=['task_types', 'suboptimal_types'])

    df = rename_duplicate_columns(df)
    df = df.convert_dtypes()
    return df


def rename_duplicate_columns(df):
    # Create a copy to avoid modifying the original
    df_renamed = df.copy()

    # Dictionary to track occurrences of each column name
    seen_name = []
    new_columns = []

    for col in df.columns:
        if col not in seen_name:
            seen_name.append(col)
            new_columns.append(col)  # Keep first occurrence as is
        else:
            new_columns.append(f"sub_{col}")

    df_renamed.columns = new_columns
    return df_renamed


def get_X(df):
    # Normalize missing tokens
    # Convert non-breaking spaces to normal spaces, blank-only cells → NaN
    df.replace({u"\u00A0": " "}, regex=True, inplace=True)
    df.replace(r"^\s*$", np.nan, regex=True, inplace=True)

    MISSING_TOKENS = {
        "NA", "N/A", "na", "n/a",
        "None", "none",
        "null", "NULL",
        "Prefer not to say"
    }

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].replace(list(MISSING_TOKENS), np.nan)

    # Convert Likert scales to just numbers
    LIKERT_REGEX = re.compile(r"^\s*(\d+)\s*—?.*$")

    for c in LIKERT_COLS:
        if c in df.columns:
            # extract the number; invalid/missing stay NaN
            df[c] = df[c].astype(str).str.extract(LIKERT_REGEX)[0].astype(float)

    # Replace missing Likert values with column median
    medians = df[LIKERT_COLS].median(numeric_only=True)
    df[LIKERT_COLS] = df[LIKERT_COLS].fillna(medians)

    # Fill missing text with "no_response"
    for c in TEXT_COLS:
        if c in df.columns:
            df[c] = df[c].fillna("NA")

    df = reformat_rename(df.drop(columns=["label"]))
    df = make_bow(df, VOCAB)

    # We need to esnure that all required columns exist first
    for feature in FEATURES:
        if feature not in df.columns:
            df[feature] = 0
  
    df = df[FEATURES]
    return df.to_numpy()


def softmax(z):
    """
    Compute the softmax of vector z, or row-wise for a matrix z.
    For numerical stability, subtract the maximum logit value from each
    row prior to exponentiation (see above).

    Parameters:
        `z` - a numpy array of shape (K,) or (N, K)

    Returns: a numpy array with the same shape as `z`, with the softmax
        activation applied to each row of `z`
    """
    if len(z.shape) == 1:
        z = z - np.max(z)
        return np.exp(z) / np.sum(np.exp(z))

    z = z - np.max(z, axis=1, keepdims=True)
    row_sums = np.sum(np.exp(z), axis=1, keepdims=True)
    return np.exp(z) / row_sums


def make_bow(df, vocab):
    """
    Produce the bag-of-word representation of the data, along with a vector
    of labels. You *may* use loops to iterate over `data`. However, your code
    should not take more than O(len(data) * len(vocab)) to run.

    Parameters:
        `data`: a list of `(review, label)` pairs, like those produced from
                `list(csv.reader(open("trainvalid.csv")))`
        `vocab`: a list consisting of all unique words in the vocabulary

    Returns:
        `X`: A data matrix of bag-of-word features. This data matrix should be
             a numpy array with shape [len(data), len(vocab)].
             Moreover, `X[i,j] == 1` if the review in `data[i]` contains the
             word `vocab[j]`, and `X[i,j] == 0` otherwise.
    """
    df_text = df[['tasks_open', 'verify_open', 'suboptimal_open']].fillna('').agg(' '.join, axis=1)
    df_numeric = df.drop(columns=['tasks_open', 'verify_open', 'suboptimal_open'])

    bow = np.zeros([len(df), len(vocab)])
    # fill in the appropriate values of X and t
    for i, line in enumerate(df_text):
        for j in range(len(vocab)):
            if vocab[j] in line:
                bow[i, j] = 1
    df_bow = pd.DataFrame(bow, columns=VOCAB, index=df.index)
    df_final = pd.concat([df_numeric, df_bow], axis=1)
    return df_final


def predict_all(filename):
    # Load and clean dataset
    df = pd.read_csv(filename, keep_default_na=True, skipinitialspace=True)
    X = get_X(df)

    # Load the parameters
    params = np.load('mlp_params.npy', allow_pickle=True).item()
    weights = params['weights']
    intercepts = params['intercepts']

    # Extract weights and biases
    W1, b1 = weights[0], intercepts[0]
    W2, b2 = weights[1], intercepts[1]
    W3, b3 = weights[2], intercepts[2]

    # Forward pass through the network
    # Layer 1 (16 units)
    z1 = X.dot(W1) + b1
    a1 = np.maximum(0, z1)  # ReLU activation

    # Layer 2 (16 units)
    z2 = a1.dot(W2) + b2
    a2 = np.maximum(0, z2)  # ReLU activation

    # Output layer (3 units)
    logits = a2.dot(W3) + b3

    # Get predictions (class with highest score)
    predictions = np.argmax(logits, axis=1)

    class_mapping = {
        0: "ChatGPT",
        1: "Claude",
        2: "Gemini"
    }

    # Convert to class names
    class_predictions = [class_mapping[pred] for pred in predictions]

    return class_predictions
