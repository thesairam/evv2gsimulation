from setuptools import find_packages, setup

setup(
    name='evv2gsim',
    packages=find_packages(include=['evv2gsim']),
    version='0.1.0',
    description='EV V2G SIMULATION LIBRARY',
    author='Sairam.S',
    install_requires=[],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    test_suite='tests',
)