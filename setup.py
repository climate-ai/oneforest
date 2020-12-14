import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="OneForest", # Replace with your own username
    version="0.0.1",
    author="Kenza Amara",
    author_email="kamara@student.ethz.ch",
    description="A package to fuse multi-view data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/foresttrace",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: ETH License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)