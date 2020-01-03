import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tinynn",
    version="0.0.5",
    author="borgwang",
    author_email="borgwang@126.com",
    description="A lightweight deep learning library",
    long_description="tinynn is an open source, lightweight deep learning library "
                     "written in Python.",
    long_description_content_type="text/markdown",
    url="https://github.com/borgwang/tinynn",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    install_requires=[
        "numpy>=1.16.0",
        "scipy>=1.3.1"
    ],
    python_requires='>=3.6'
)
