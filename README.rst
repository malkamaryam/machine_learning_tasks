Diabetes 30-Day Readmission Prediction
========================================

A machine learning pipeline to predict whether a diabetic patient will be
readmitted to the hospital within 30 days of discharge, using the UCI
"Diabetes 130-US hospitals" dataset.

Overview
--------

This project covers the full pipeline from raw clinical data to model
explainability:

- Data cleaning and missing value handling
- Feature engineering (diagnosis grouping by ICD-9 category, admission/discharge
  recoding, drug dosage encoding)
- One-hot and ordinal encoding of categorical clinical features
- Feature selection using chi-square and point-biserial correlation tests
- Model training and comparison: Logistic Regression, Random Forest, and XGBoost
- Model explainability using SHAP (summary plots, waterfall plots, feature
  importance comparison across models)

Project Structure
------------------

::

    machine_learning_tasks/
    ├── biome_analytics/      # Core package: preprocessing and helper functions
    ├── data/                 # Raw dataset (not tracked in git)
    ├── notebooks/            # Exploratory Jupyter notebook
    ├── docs/                 # Documentation
    ├── tests/                # Unit tests
    ├── setup.py
    ├── requirements.txt
    └── LICENSE

Installation
------------

.. code-block:: bash

    git clone https://github.com/malkamaryam/machine_learning_tasks.git
    cd machine_learning_tasks
    pip install -r requirements.txt

Dataset
-------

This project uses the `Diabetes 130-US hospitals for years 1999-2008
<https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008>`_
dataset. Place ``diabetic_data.csv`` and ``IDS_mapping.csv`` in the ``data/``
folder before running the pipeline.

Usage
-----

.. code-block:: bash

    python biome_analytics/core.py

Running Tests
-------------

.. code-block:: bash

    pytest tests/

Results
-------

Model performance is compared using ROC-AUC and PR-AUC, with SHAP values used
to identify and visualize the top clinical risk factors for early readmission.

License
-------

This project is licensed under the MIT License. See ``LICENSE`` for details.

Author
------

Malka Maryam — BS Bioinformatics, NUST Islamabad
