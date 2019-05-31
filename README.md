# Cover Song Recognition Much?

A pile of code for creating a neural network which is capable of identifying cover songs.

I created this project for my own exploratory / learning purposes, but there's been enough
requests from others to "see the code" that I've decided to clean things up a bit for
the masses.

## Instructions for running this whole thing
Note: Mileage may vary. I've run this project on both Mac OS and a couple flavors of
Ubuntu. Winoughs might not work. If you run into issues, just open an issue and we'll
see if we can figure it out.
Second note: I use `conda` for most of these things as it seems to treat me well.
* Install Python 3.6 using.
* Install the packages from `setup.py`. Making a new `env` just for this project isn't
a terrible idea.
* Place your training MP3 / WMA / M4A / WAV files in a directory
* Place your validation MP3 / WMA / M4A / WAV files in another directory.
The filename must match with the matching song from the training set. The full path isn't
used.
* Update `root_*_dir` properties in `core.py` to point to your directories.
* Update the `tempo_map` property in `core.py` with the relevant BPMs for your songs.
Alternatively, if you like to live dangerously, just update the code to allow Librosa
to auto-detect the BPM without any hints if no entry is found in `tempo_map` for
a given song.
* Run the 5 functions specified at the bottom of `core.py`
* Update filenames in `train_raw_lstm.py` to match any filename changes from `core.py`.
* Run `train_raw_lstm.py`. I typically do this by pasting everything in to the Python shell.
Because I like interactivity when things go awry.
* Update paths as necessary in `testing.py`. Pass in full paths to songs you want
validated against your new model like so `python -m audio_much.testing ~/Desktop/hit_me_with_your_best_shot.mp3`


By Stephen Hopper