from setuptools import setup, find_packages

setup(
    name='mlcom',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'xgboost',
        'lightgbm',
        'shap',
        'lime'
    ],
)
