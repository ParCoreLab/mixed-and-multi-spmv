#!/bin/bash

# Command line args
if [[ $# -ne 2 ]] ; then
  echo "Please provide the following arguments only:"
  echo "$0 <path to html> <output name>"
  exit 1
fi

# Check if given path exists
if [ ! -f "$1" ]; then
  echo "The given file '$1' does not exist."
  exit 1
fi

# Parse html
cat $1 | \
egrep -o "MM/[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+.tar.gz" | \
sed -e "s/\.tar\.gz//g" -e "s/MM\///g" | \
sort > $2