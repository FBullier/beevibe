from setuptools import setup, find_packages

setup(
    name="beevibe",
    version="0.1.0",
    description="A Python package for training and inference of language models on thematic datasets.",
    author="BeeVibe Authors",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.10.12",
    install_requires=[
        "pandas>=2.2.2",
        "numpy>=1.26.4",
        "scipy>=1.13.1",
        "scikit-learn>=1.6.0",
        "torch>=2.5.1",
        "transformers>=4.47.0",
        "tokenizers>=0.21.0",
        "datasets>=3.2.0",
        "seaborn>=0.11.0",
        "matplotlib>=3.3.0",
        "pydantic>=1.10.0",
    ],
    extras_require={
        "dev": [
            "pytest",
            "ruff",
            "black",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    include_package_data=True,
)
