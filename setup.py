from pathlib import Path

from setuptools import setup, find_packages

required = Path("requirements.txt").read_text(encoding="utf-8").split("\n")

setup(

    name="beevibe",
    version="0.1.0",
    description="A lightweight framework for training and deploying language models for thematic text classification.",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="François Bullier",
    author_email="fbullier@360client.fr",
    url="https://github.com/FBullier/beevibe",
    packages=find_packages(exclude=["tests", "tests.*"]),  # same as name
    license="MIT",

    install_requires=required,
    extras_require={
        "dev": [
            "pytest",
            "ruff",
            "black",
        ]
    },
    include_package_data=True,    
    python_requires=">=3.10.12",

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)