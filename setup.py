from setuptools import setup

setup(
    name="gefry",
    version="3.5.3",
    author="Jason M. Hite",
    packages=["gefry3", "gefry3.classes"],
    install_requires=['pyyaml', 'shapely', 'numpy'],
    license="2-clause BSD (FreeBSD)",
    extras_require={
        "plots": ["matplotlib", "seaborn"],
        "OrientedPrismDetector": ["pyst"],
    },
)                     
