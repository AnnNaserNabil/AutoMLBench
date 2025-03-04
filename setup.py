from setuptools import setup, find_packages

setup(
    name='AutoMLBench',
    version='0.1.0',
    author='Ann Naser Nabil',
    author_email='ann.n.nabil@gmail.com',
    description='A Python package for automated ML model benchmarking and comparison',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/AnnNaserNabil/AutoMLBench',
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
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.7',
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'AutoMLBench=AutoMLBench.cli:main'  # If you have a CLI tool
        ]
    },
)
