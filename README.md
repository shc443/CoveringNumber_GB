# How DNNs Break the Curse of Dimensionality: Compositionality and Symmetry Learning

## Introduction

This research project explores how Deep Neural Networks(DNN) can learn composition of functions with bounded F-1 Norm, referenced in [How DNNs Break the Curse of Dimensionality: Compositionality and Symmetry Learning]

This page includes practical implementations in Python with associated experimental results. 
These notebooks contain the code, data, and visualizations used in our study, allowing for reproducibility and further exploration.

## Contents

- `notebooks/`: Directory containing Jupyter notebooks with code and experiments.
- `data/`: Directory with both syntheic & real-world datasets used in our experiments
- `README.md`: This file, providing an overview and instructions.

## Installation

To run the code in this project, you will need to have Python and Jupyter installed. You can set up a virtual environment and install the required packages using the following commands:

```sh
# Clone the repository
git clone https://github.com/shc443/CoveringNumber_GB
cd CoveringNumber_GB

# Install required packages
pip install -r requirements.txt

#Download syntheic data from Author's Google Drive (ignore if you want to start this project from the scratch)
gdown --fuzzy https://drive.google.com/file/d/1XAuHtRWdUGs5SqDfLttqHSUILmFsgn6b/view?usp=drive_link
mkdir sampled_kernel/
unzip sampled_kernel.zip -d data/synthetic/

#Download real-world dataset (WESAD as pkl)

#Download result files(test_error & GB files as pkl)

