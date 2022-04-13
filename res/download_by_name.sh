#!/bin/bash 

if [[ $# -ne 1 ]] ; then
  echo "Please provide a matrix name:"
  echo "$0 <path to html>"
  exit 1
fi

mtx=$(grep "/$1$" index/mm.all.index)

wget "https://suitesparse-collection-website.herokuapp.com/MM/$mtx.tar.gz" -O - \
	| tar -xvz -C ./ --strip-components=1 \
	&& exit

# You can use MatrixMarket links at https://sparse.tamu.edu/