#!/bin/bash
FILENAME="sourceid_list.txt"
LINES=$(cat $FILENAME)
for LINE in $LINES
do
    pushd parallax
    python fit_parallax_model.py --source_id "$LINE"
    popd
    
    pushd simple
    python fit_simple_model.py --source_id "$LINE"
    popd
done
