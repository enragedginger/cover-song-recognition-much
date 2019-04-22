from setuptools import setup

setup(name='audio_much',
      version='0.0.1',
      description='Audio Much library for doing ML on audio things',
      author='Stephen Hopper',
      url='http://github.com/enragedginger/audio_much',
      author_email='stephenmhopper@gmail.com',
      license='MIT',
      packages=[],
      install_requires=[
        'keras >= 2.0.0',
        'numpy >= 1.8.0',
        'librosa >= 0.5',
        'future',
        'kapre',
        'pandas',
        'matplotlib',
        'kapre',
        'sklearn',
        'tables'
      ],
      extras_require={
          'tests': ['tensorflow'],
       },
      keywords='audio music deep learning keras',
      zip_safe=False)