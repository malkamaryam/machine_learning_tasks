Machine Learning Tasks
=======================

A portfolio of two independent machine learning pipelines.

--------------------------------------------------------------------

Task 1: Diabetes 30-Day Readmission Prediction
------------------------------------------------

A machine learning pipeline to predict whether a diabetic patient will be
readmitted to the hospital within 30 days of discharge, using the UCI
"Diabetes 130-US hospitals" dataset.

**Overview**

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

**Installation & Usage**

.. code-block:: bash

    git clone https://github.com/malkamaryam/machine_learning_tasks.git
    cd machine_learning_tasks
    pip install -r requirements.txt

Place ``diabetic_data.csv`` and ``IDS_mapping.csv`` in ``data/`` before running:

.. code-block:: bash

    python biome_analytics/core.py

--------------------------------------------------------------------

Task 2: Hospital Value-Performance Benchmark
-----------------------------------------------

A quality-vs-cost benchmarking pipeline for U.S. hospitals, using CMS's
public Provider Data Catalog datasets (Hospital General Information, HCAHPS
patient experience, unplanned hospital visits/readmissions, and Medicare
Spending Per Beneficiary).

**Overview**

This project builds a full pipeline from raw CMS data to an interactive
dashboard:

- Ingestion and joining of 4 CMS hospital datasets on Facility ID
- Feature engineering: composite quality/cost value scores using z-scores,
  with documented assumptions on missing-data handling and metric weighting
- Peer grouping via K-means clustering (hospital type, ownership, emergency
  services), with group count chosen via the elbow method
- Outlier detection within each peer group (2 standard deviation threshold)
- Baseline ML model (Gradient Boosting) predicting CMS's official hospital
  star rating, iterated across 3 versions (R\u00b2 improved from 0.157 to 0.512)
- Explainability via SHAP values, giving a hospital-specific breakdown of
  which metrics drive its predicted rating
- Interactive Streamlit + Plotly dashboard: filter by State, Hospital Type,
  or Peer Group; view a hospital's value score, outlier status, SHAP
  breakdown, and a peer-group radar chart comparison

Full documentation of assumptions, decisions, bugs found/fixed, and
findings is in ``docs/hospital_value_benchmark.md``.

**Installation & Usage**

.. code-block:: bash

    git clone https://github.com/malkamaryam/machine_learning_tasks.git
    cd machine_learning_tasks
    pip install -r requirements.txt

Place the CMS Provider Data Catalog hospital CSV files in
``data/hospital_value_benchmark/`` before running the pipeline in order:

.. code-block:: bash

    python hospital_value_benchmark/ingestion.py
    python hospital_value_benchmark/features.py
    python hospital_value_benchmark/peer_groups.py
    python hospital_value_benchmark/outliers.py
    python hospital_value_benchmark/model.py
    python hospital_value_benchmark/explainability.py

Then launch the dashboard:

.. code-block:: bash

    python -m streamlit run hospital_value_benchmark/dashboard.py

--------------------------------------------------------------------

Project Structure
--------------------

.. code-block::

    machine_learning_tasks/
    \u251c\u2500\u2500 biome_analytics/            # Task 1 package
    \u251c\u2500\u2500 hospital_value_benchmark/   # Task 2 package
    \u251c\u2500\u2500 data/                       # Raw datasets (not tracked in git)
    \u251c\u2500\u2500 notebooks/                  # Exploratory notebooks
    \u251c\u2500\u2500 docs/                       # Documentation for both tasks
    \u251c\u2500\u2500 tests/                      # Unit tests for both tasks
    \u251c\u2500\u2500 setup.py
    \u251c\u2500\u2500 requirements.txt
    \u2514\u2500\u2500 LICENSE

Running Tests
----------------

.. code-block:: bash

    pytest tests/

License
----------

This project is licensed under the MIT License. See ``LICENSE`` for details.

Author
---------

Malka Maryam \u2014 BS Bioinformatics, NUST Islamabad
