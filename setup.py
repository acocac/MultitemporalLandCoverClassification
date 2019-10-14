"""Config for installing a python module/package."""

from setuptools import setup, find_packages

NAME = 'MTLCC'
AUTHOR = 'Alejandro Coca-Castro based on Rußwurm & Körner (2018) Multi-Temporal Land Cover Classification with Sequential Recurrent Encoders',
EMAIL = 'acocac@gmail.com',
VERSION = '0.1'
REQUIRED_PACKAGES = ['configparser','cloudml-hypertune']
LICENSE = 'MIT'
DESCRIPTION = 'Run MTLCC in Google AI'

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    license=LICENSE,
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    zip_safe=False)