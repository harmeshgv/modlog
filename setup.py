from setuptools import setup, find_packages

setup(
    name="ml_tracker",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pandas",  
    ],
    python_requires=">=3.8",
)