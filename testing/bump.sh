#!/bin/bash

set -e

source testing/env.sh
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

pyenv virtualenv -f bump
pyenv activate bump
pip install bumpversion

function get_next_version() {
    current_version=$(python setup.py --version)
    next_version=$(python testing/next_version.py $GIT_LOCAL_BRANCH $current_version)
    echo $next_version
}

# convert origin/foobar to foobar
GIT_LOCAL_BRANCH=${GIT_BRANCH#*/}

# a commit to master always bumps the patch
if [[ $GIT_LOCAL_BRANCH =~ ^master$ ]]; then
    bumpversion patch

# release and develop branches always bump the release version. the next_version.py helper script will also determine if
# a new dev or release version needs to be explicitly created
elif [[ $GIT_LOCAL_BRANCH =~ ^develop$ ]] || [[ $GIT_LOCAL_BRANCH =~ ^release-.*$ ]]; then

    next_version=$(get_next_version)

    if [ ! -z "$next_version" ]; then
        bumpversion --new-version $next_version release_version
    else
        bumpversion release_version
    fi

fi

# other branches are left as-is
