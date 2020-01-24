"""A setuptools based setup module.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

from io import open
from os import path

from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='yummycurry',
    version='0.1.0',
    description='Automatic currying, uncurrying and application of functions and methods',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    url='https://github.com/Niriel/yummycurry',
    author='Niriel',
    # author_email='nope, use github',

    classifiers=[
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Developers',
        'Topic :: Software Development',
        'Topic :: Utilities',

        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 3.8',
    ],

    keywords='curry currying uncurry uncurrying partial',

    package_dir={'': 'src'},

    packages=find_packages(where='src'),  # Required

    python_requires='>=3.8',

    install_requires=[],

    extras_require={
        'test': ['pytest'],
    },
)
