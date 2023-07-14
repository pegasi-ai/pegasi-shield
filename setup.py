import setuptools
from setuptools import find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="guardrail-ml",
    version="0.0.11",
    packages=find_packages(exclude=["tests", "guardrailmlev", "examples", "docs", "env", "dist"]),
    author="Kevin Wu",
    author_email="kevin@guardrailml.com",
    description="Monitor LLMs with custom metrics to scale with confidence",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://www.guardrailml.com",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "PyPDF2",
        "textstat",
        "transformers",
        "sentencepiece",
        "accelerate",
        "bitsandbytes",
        "cleantext",
        "unidecode",
        "pillow",
        "jsonformer",
        "scipy",
    ],
)
