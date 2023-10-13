# ###############################################################################
#
#  The MIT License (MIT)
#  Copyright (c) 2023 Philippe Ostiguy
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
#  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
#  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#  DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#  OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
#  OR OTHER DEALINGS IN THE SOFTWARE.
###############################################################################

import os

INPUT_PATH = "resources/input"
WORK_DIRECTORY = os.getcwd()
MODELS_PATH = "models"

MODEL_OPTIMIZATION_PATH = 'models/hyperparameter_optimization_optuna'
RAW_PATH = os.path.join(INPUT_PATH, "raw")
PREPROCESSED_PATH = os.path.join(INPUT_PATH, "preprocessed")
ENGINEERED_PATH = os.path.join(INPUT_PATH, "engineered")
PICKLE = "pickle"
TRANSFORMATION_PATH = os.path.join(INPUT_PATH, PICKLE)
MODEL_DATA_PATH = os.path.join(INPUT_PATH, "model_data")
PATHS_TO_CREATE = [
    INPUT_PATH,
    WORK_DIRECTORY,
    MODELS_PATH,
    RAW_PATH,
    PREPROCESSED_PATH,
    ENGINEERED_PATH,
    TRANSFORMATION_PATH,
    MODEL_DATA_PATH,
    MODEL_OPTIMIZATION_PATH
]

# Data attributes
RAW_ATTRIBUTES = ["date", "value"]
MODEL_PHASES = ["train", "predict"]

# Data engineering
ENGINEERED_DATA_TO_REMOVE = 2


DATASETS = ['train', 'predict', 'test']