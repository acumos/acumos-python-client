#!/bin/bash

source testing/env.sh
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

if [ -d "$WORKSPACE/.tox" ]; then rm -rf "$WORKSPACE/.tox"; fi

pyenv update

# install desired interpreters
pyenv install -s 3.4.6
pyenv install -s 3.5.3
pyenv install -s 3.6.2

# set local python versions
pyenv local 3.4.6 3.5.3 3.6.2

# install any python libraries needed for jenkins testing
pip install -r testing/test-requirements.txt

# flaky tests sometimes fail. try running tox a few times.
for n in {1..5}
do
  echo "tox iteration $n"
  tox -vv 
  retval=$?
  if (( $retval == 0 )); then break; fi
done

if (( $retval != 0 )); then { echo "All tox tests failed" ; exit 1; } fi
