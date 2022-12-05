from setuptools import setup, find_packages


setup(
    name="pyboostlss",
    version="0.1.0",
    description="Py-BoostLSS: An extension of Py-Boost to probabilistic modelling",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Alexander März",
    author_email="alex.maerz@gmx.net",
    url="https://github.com/StatMixedML/Py-BoostLSS",
    license="Apache License 2.0",
    packages=find_packages(exclude=["tests"]),
    include_package_data=True,
    package_data={'': ['datasets/*.csv']},
    zip_safe=True,
    python_requires=">=3.8, <3.10",
    install_requires=[
        "py-boost~=0.3.0",
        "optuna~=3.0.3",
        "pyro-ppl~=1.8.3", 
        "scikit-learn=~1.1.3",
        "numpy~=1.23.5",
        "pandas~=1.5.2",
        "plotnine~=0.10.1",
        "scipy~=1.8.1",
        "tqdm~=4.64.1",
        "matplotlib=~3.6.2",
        "ipywidgets=~8.0.2",
    ],
    test_suite="tests",
    tests_require=["flake8", "pytest"],
)