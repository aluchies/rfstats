#!/usr/bin/env python

from setuptools import setup, find_packages
import rfstats

setup(
    name='rfstats',
    version=rfstats.__version__,
    description='statistics for random fields',
    long_description='',
    keywords='random fields correlation',
    author='Adam Luchies',
    url='https://github.com/aluchies/rfstats',
    license='MIT',
    packages=find_packages(),
    install_requires=[],
)