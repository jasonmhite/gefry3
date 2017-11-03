from setuptools import setup

setup(
    name="gefry",
    version="3.4.2",
    author="Jason M. Hite",
    license="BSD",
    packages=["gefry3"],
    install_requires=['shapely', 'numpy'],
    license="2-clause BSD (FreeBSD)",
    extras_require={
        "plots": ["matplotlib", "seaborn"],
        "OrientedPrismDetector": ["pyst"],
    },
)                     
