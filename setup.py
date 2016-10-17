from setuptools import setup

setup(
    name="gefry",
    version="3.0.0-pre-alpha",
    author="Jason M. Hite",
    license="BSD",
    packages=["gefry3"],
    install_requires=['shapely', 'numpy'],
    # package_data={'gefry2': ['data/*.h5']}
    )                     
