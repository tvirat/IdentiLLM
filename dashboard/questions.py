# Define question structure
TASK_OPTIONS = [
    "Writing or debugging code",
    "Math computations",
    "Explaining complex concepts simply",
    "Drafting professional text (e.g., emails, résumés)",
    "Data processing or analysis",
    "Brainstorming or generating creative ideas",
    "Writing or editing essays/reports",
    "Converting content between formats (e.g., LaTeX)"
]

LIKERT_OPTIONS = [
    "1 – Never",
    "2 – Rarely",
    "3 – Sometimes",
    "4 – Often",
    "5 – Very often"
]

LIKELIHOOD_OPTIONS = [
    "1 – Very unlikely",
    "2 – Unlikely",
    "3 – Neutral / Unsure",
    "4 – Likely",
    "5 – Very likely"
]

QUESTIONS = [
    {
        "number": 1,
        "text": "In your own words, what kinds of tasks would you use this model for?",
        "type": "open_ended",
        "mandatory": False,
        "key": "In your own words, what kinds of tasks would you use this model for?"
    },
    {
        "number": 2,
        "text": "How likely are you to use this model for academic tasks?",
        "type": "likert_likelihood",
        "mandatory": True,
        "key": "How likely are you to use this model for academic tasks?"
    },
    {
        "number": 3,
        "text": "Which types of tasks do you feel this model handles best? (Select all that apply.)",
        "type": "multiple_choice",
        "mandatory": True,
        "options": TASK_OPTIONS,
        "key": "Which types of tasks do you feel this model handles best? (Select all that apply.)"
    },
    {
        "number": 4,
        "text": "Based on your experience, how often has this model given you a response that felt suboptimal?",
        "type": "likert",
        "mandatory": True,
        "key": "Based on your experience, how often has this model given you a response that felt suboptimal?"
    },
    {
        "number": 5,
        "text": "For which types of tasks do you feel this model tends to give suboptimal responses? (Select all that apply.)",
        "type": "multiple_choice",
        "mandatory": True,
        "options": TASK_OPTIONS,
        "key": "For which types of tasks do you feel this model tends to give suboptimal responses? (Select all that apply.)"
    },
    {
        "number": 6,
        "text": "Think of one task where this model gave you a suboptimal response. What did the response look like, and why did you find it suboptimal?",
        "type": "open_ended",
        "mandatory": False,
        "key": "Think of one task where this model gave you a suboptimal response. What did the response look like, and why did you find it suboptimal?"
    },
    {
        "number": 7,
        "text": "How often do you expect this model to provide responses with references or supporting evidence?",
        "type": "likert",
        "mandatory": True,
        "key": "How often do you expect this model to provide responses with references or supporting evidence?"
    },
    {
        "number": 8,
        "text": "How often do you verify this model's responses?",
        "type": "likert",
        "mandatory": True,
        "key": "How often do you verify this model's responses?"
    },
    {
        "number": 9,
        "text": "When you verify a response from this model, how do you usually go about it?",
        "type": "open_ended",
        "mandatory": False,
        "key": "When you verify a response from this model, how do you usually go about it?"
    }
]