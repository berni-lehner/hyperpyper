from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='torch-transights',
    version='0.1.0',
    author='Bernhard Lehner',
    author_email='berni.lehner@gmail.com',
    description='Automatic collation of full batch output from transformation pipelines.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/berni-lehner/zippee-ki-yay',
    project_urls = {
        "Bug Tracker": "https://github.com/berni-lehner/torch-transights/issues"
    },
    license='Apache 2.0',
    install_requires=[],
    packages=find_packages(where='src'),  # List of packages to include
    package_dir={'': 'src'},
)
