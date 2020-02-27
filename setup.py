# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 10:02:43 2019

@author: nghiatp
"""

from setuptools import find_packages
from setuptools import setup

setup(
    name='telco-churn',
    version='0.1',
    author='Nghia',
    author_email='nghiatrinh125@gmail.com',
    install_requires=["numpy", "pandas", "google-cloud-storage", "scikit-learn", "joblib"],
    packages=find_packages(exclude=['data']),
    description='Telco churn prediction',
    url=''
)