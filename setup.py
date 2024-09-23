# setup.py

from setuptools import setup, find_packages

setup(
    name='my_outfit_recommender',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'flask',
        'pandas',
        'numpy',
        'scikit-learn',
        'joblib',
        'matplotlib',
        'jupyter'
    ],
)
