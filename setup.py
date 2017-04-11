#!/usr/bin/env python
import os
import shutil
import sys
from setuptools import setup, find_packages

VERSION = '0.0.1'

long_description = '''Code for Coset competition'''

setup_info = dict(
    # Metadata
    name='coset',
    version=VERSION,
    author='Adrian Fernandez and recognai',
    author_email='daniel@recogn.ai',
    url='https://github.com/recognai/coset',
    description='',
    long_description=long_description,
    license='Apache2',

    # Package info
    packages=find_packages(exclude=('test',)),

    zip_safe=True,
    install_requires= [
        'nltk',
        'requests',
        'spacy',
        'visdom',
        'torchtext'
    ],
    dependency_links=['https://github.com/dvsrepo/text']
)

setup(**setup_info)
