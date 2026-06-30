from setuptools import setup, find_packages

setup(
    name="biome_analytics",
    version="0.1.0",
    description="Diabetes 30-day readmission prediction pipeline",
    author="Malka Maryam",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "scipy",
        "xgboost",
        "matplotlib",
        "shap",
    ],
)
