#!/bin/sh 
cd "${0%/*}"
matrices=$(cat $1 | egrep "\- \w+$" | sed 's/- //g')

for m in $matrices; do
  echo "Downloading $m"
	./download_by_name.sh $m
done