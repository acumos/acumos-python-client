#!/bin/bash

# WORKSPACE variable is needed for tests and deriving PYENV_ROOT
if [ -z "$WORKSPACE" ]; then
    export WORKSPACE=$(pwd)
fi

# install pyenv if it isn't already
if ! command -v pyenv &>/dev/null; then
    export PYENV_ROOT="$WORKSPACE/.pyenv"
    curl -L https://raw.githubusercontent.com/pyenv/pyenv-installer/master/bin/pyenv-installer | bash
    
    export PATH="$PYENV_ROOT/bin:$PATH"
else
    echo 'pyenv already installed'
fi
