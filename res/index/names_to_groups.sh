#!/bin/sh 

# converts "name" to "group/name"

cd "${0%/*}"

INPUT=targets.txt
OUTPUT=mm.jacobifinal.index
MATS=$(cat $INPUT)

# clean output file
> $OUTPUT

# convert name to group/name for each matrix
for m in $MATS; do
  echo $(grep "/$m$" mm.all.index) >> $OUTPUT
done

# sort the output file in-place
sort -o $OUTPUT $OUTPUT