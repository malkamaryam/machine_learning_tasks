from setuptools import setup, find_packages

setup(
    name="machine_learning_tasks",
    version="0.2.0",
    description=(
        "Machine learning task portfolio: "
        "(1) Diabetes 30-day readmission prediction (biome_analytics), "
        "(2) Hospital value-performance benchmark (hospital_value_benchmark)"
    ),
    author="Malka Maryam",
    packages=find_packages(),
    install_requires=[
        # shared / Task 1 (biome_analytics)
        "pandas",
        "numpy",
        "scikit-learn",
        "scipy",
        "xgboost",
        "matplotlib",
        "shap",
        # Task 2 (hospital_value_benchmark) additions
        "joblib",
        "streamlit",
        "plotly",
    ],
)
