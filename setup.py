import sys
from setuptools import setup, find_packages


kwargs = {}
install_requires = []
version = '0.0.1'

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
