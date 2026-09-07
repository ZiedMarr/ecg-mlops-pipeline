from setuptools import setup, find_packages

setup(
    name="DL_BSA_F",
    version="0.1.0",
    packages=find_packages(),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "hydra-core",
        "hydra-submitit-launcher",
        "wandb",
        "numpy",
        "kaggle",
        "wfdb",
        "pandas",
        "neurokit2",
        "scipy",
        "scikit-learn",
    ],
)
