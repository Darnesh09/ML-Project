from setuptools import find_packages,setup
from typing import List

def get_requirements(file_path:str)->List[str]:
    """
        This file read the requirements from the file_path
    """
    requirements = []
    with open(file_path) as file:
        requirements = [req.strip() for req in file.readlines()]
        if '-e .' in requirements:
            requirements.remove('-e .')
    return requirements


setup(
    name = 'ML Project',
    version = '0.0.1',
    author = 'Darnesh',
    author_email = 'darnesh@mail.com',
    packages = find_packages(),
    install_requires = get_requirements("requirements.txt")
)