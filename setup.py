from setuptools import setup

setup(name='indonesian_coreference_resolution',
      version='0.1',
      description='Client for Indonesian Coreference Resolution',
      url='https://github.com/tugas-akhir-nlp/indonesian-coreference-resolution-cnn',
      author='Turfa Auliarachman',
      author_email='turfa_auliarachman@rocketmail.com',
      license='MIT',
      packages=['coref_client'],
      zip_safe=False,
      install_requires=['pandas', 'tensorflow', 'numpy', 'scikit-learn', 'nltk'])
