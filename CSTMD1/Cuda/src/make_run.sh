#!/bin/bash

f=0
t=1

cl=$f
run=$f
ls=$f
mk=$f

while [ $# -gt 0 ]
do
    case "$1" in
  -m) mk=$t
      echo "set to make";;
	-c) cl=$t
	    echo "set to clean";;
	-r) run=$t
	    echo "set to run";;
	-l) ls=$t
	    echo "set to redirect output";;
	-*)
	    echo >&2 \
	    "usage: $0 [-c clean] [-r run] [-m make] [-l redirect output]"
	    exit 1;;
	*)  break;;	# terminate while loop
    esac
    shift
done

if [ $cl -eq $t ]; then
    make clean
fi

if [ $mk -eq $t ]; then
    source /vol/cuda/7.5.18/setup.sh

    if [ $ls -eq $t ]; then
      make > output.txt 2>&1
    else
      make
    fi
fi

if [ $run -eq $t ]; then

    if [ $ls -eq $t ]; then
      ./multi >> output.txt 2>&1
      less output.txt
    else
        ./multi
    fi
fi
