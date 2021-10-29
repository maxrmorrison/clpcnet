from pathlib import Path
from pkg_resources import parse_requirements
from setuptools import setup


with open('README.md') as file:
    long_description = file.read()


with open(Path(__file__).parent / 'requirements.txt') as file:
    requirements = [str(req) for req in parse_requirements(file)]


setup(
    name='clpcnet',
    version='0.0.1',
    description='Neural pitch-shifting and time-stretching with controllable lpcnet',
    author='Max Morrison',
    author_email='maxrmorrison@gmail.com',
    url='https://github.com/maxrmorrison/clpcnet',
    packages=['clpcnet'],
    package_data={'clpcnet': ['assets/*']},
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords='speech vocoder prosody pitch-shifting time-stretching lpcnet',
    install_requires=requirements)
