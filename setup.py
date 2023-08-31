from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='transights',
    version='0.1.1',
    author='Bernhard Lehner',
    author_email='berni.lehner@gmail.com',
    description='Automatic collation of full batch output from transformation pipelines.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/berni-lehner/transights',
    project_urls = {
        "Bug Tracker": "https://github.com/berni-lehner/transights/issues"
    },
    license='Apache 2.0',
    install_requires=[],
    packages=find_packages(),
    #package_dir={'': 'transights'},
)
