from typing import List

from setuptools import find_packages, setup


def get_requirements(file_path: str) -> List[str]:
    with open(file_path) as file_obj:
        requirements = [line.strip() for line in file_obj if line.strip()]

    return requirements


setup(
    name='Linking_Writing',
    version='0.0.1',
    author='Shivansh',
    author_email='uishivansh2503@gamil.com',
    install_requires=get_requirements('requirements.txt'),
    packages=find_packages()
)
