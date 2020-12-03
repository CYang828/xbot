import sys
from setuptools import setup, find_packages


kwargs = {}
install_requires = []
version = '1.0.0.0'

if sys.version_info < (3, 0):
    with open('README.md') as f:
        kwargs['long_description'] = f.read()

    with open('requirements.txt') as f:
        for require in f:
            install_requires.append(require[:-1])
elif sys.version_info > (3, 0):
    with open('README.md', encoding='utf-8') as f:
        kwargs['long_description'] = f.read()

    with open('requirements.txt', encoding='utf-8') as f:
        for require in f:
            install_requires.append(require[:-1])

kwargs['install_requires'] = install_requires

setup(
    name='xbot',
    version=version,
    include_package_data=True,
    packages=find_packages(),
    package_data={'xbot': ['config/crosswoz_all_context_nlu_intent.json',
                           'config/crosswoz_all_context_nlu_slot.json',
                           'config/crosswoz_all_context_joint_nlu.json']},
    entry_points={
        'console_scripts': [
        ],
    },
    author='bslience',
    author_email='zhangchunyang_pri@126.com',
    description="This is a chatbot for X.",
    url="https://github.com/BSlience/xbot",
    long_description_content_type="text/markdown",
    **kwargs
)
