from setuptools import setup, find_packages

setup(
    name="beevibe",
    version="0.1.0.dev9",
    description="A lightweight framework for training and deploying language models for thematic text classification.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="François Bullier",
    url="https://github.com/FBullier/beevibe",
    packages=find_packages(exclude=["tests", "tests.*"]),
    license="MIT",
    include_package_data=True,
    python_requires=">=3.10.12",

)
