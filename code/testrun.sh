#!/bin/sh

MAX_DENSITY=0.65
SEQUENCES=1000
INSTANCES_PER_SEQUENCE=500
STRATEGY_ARGS="--strategy random"

for elems in 05 10 15 20; do
    echo '**********************'
    echo "Running for $elems elements"
    echo '**********************'
    mkdir -p results/$elems
    seq -w $SEQUENCES | \
        parallel --progress \
            ./find_unsolvable.py --density $MAX_DENSITY \
                --elements $elems --instances $INSTANCES_PER_SEQUENCE \
                $STRATEGY_ARGS \
                ">results/$elems/{}"
done
