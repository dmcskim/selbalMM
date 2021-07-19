import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

    setuptools.setup(
        name="selbalMM",
        version="0.0.1",
        author="Daniel Ian McSkimming",
        author_email="dmcskimming@usf.edu",
        description="Select compositional balances using mixed models.",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/pypa/sampleproject",
        project_urls={
                    "Bug Tracker":
            "https://github.com/pypa/sampleproject/issues",
                },
        classifiers=[
                    "Programming Language :: Python :: 3",
                    "License :: OSI Approved :: MIT License",
                    "Operating System :: OS Independent",
                ],
        py_modules = ['selbalMM.selbalMM', 'selbalMM.core'],
        python_requires=">=3.6",
    )
