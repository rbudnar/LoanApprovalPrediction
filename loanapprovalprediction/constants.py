TARGET = "loan_status"
CONTINUOUS_VARIABLES = [
    "person_age",
    "person_income",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]
CATEGORICAL_VARIABLES = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]

FINAL_FEATURES= [
    'loan_grade', 
    'loan_percent_income', 
    'person_home_ownership',
    'person_income', 
    'loan_intent', 
    'loan_int_rate',
    'loan_amnt'
]