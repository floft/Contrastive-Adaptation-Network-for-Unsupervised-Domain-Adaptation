#!/bin/bash
#
# Upload files to high performance cluster
#
. kamiak_config.sh

# Note both have trailing slashes
from="$localdir"
to="$remotessh:$remotedir"

# Copy only select files
rsync -Pahuv --exclude="experiments/dataset/VisDA-2017" "$from" "$to"
