from setuptools import find_packages, setup
from typing import List

def get_requirements() -> List[str]:
    """
    This function will return the list of requirements.
    """
    requirement_list: List[str] = []
    try:
        with open('requirements.txt', 'r') as file:
            lines = file.readlines()
            for line in lines:
                requirement = line.strip()
                # Ignore empty lines and -e .
                if requirement and requirement != '-e .':
                    requirement_list.append(requirement)
    except FileNotFoundError:
        print('requirements.txt file not found.')
    
    return requirement_list

setup(
    name = 'network-security',
    version = '0.0.1',
    author = 'Gianluca De Bonis',
    author_email = 'gianlucadebonis@hotmail.it',
    packages = find_packages(),
    install_requires = get_requirements()
)