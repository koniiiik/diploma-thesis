#!/bin/sh

DENSITIES=".645 .6 .5 .4 .3"
SEQUENCE_SIZES="10 20 30 40 50 60 70"
SEQUENCES=100
INSTANCES_PER_SEQUENCE=100
# List of strategy combos to run:
# strategy,output dir suffix
# This needs to end with a semicolon, otherwise sh won't pick up the last
# element.
STRATEGY_COMBOS="--strategy random,random;\
                 --strategy linear-dep --linear-deps 1,linear-dep/1;\
                 --strategy linear-dep --linear-deps 5,linear-dep/5;"

OUTDIR_TEMPLATE="results/density%.03f/n%02d/%s"

for MAX_DENSITY in $DENSITIES; do
    for ELEMS in $SEQUENCE_SIZES; do
        echo "$STRATEGY_COMBOS" | while IFS=',' read -d ';' STRATEGY_ARGS OUTDIR_SUFFIX; do
            echo '***********************'
            printf "Running for density %.03f, %02d elements, strategy %s\n" \
                    "$MAX_DENSITY" "$ELEMS" "$STRATEGY_ARGS"
            echo '***********************'
            OUTDIR="$(printf "$OUTDIR_TEMPLATE" "$MAX_DENSITY" "$ELEMS" "$OUTDIR_SUFFIX")"
            mkdir -p "$OUTDIR"
            seq -w $SEQUENCES | \
                parallel --progress \
                    ./find_unsolvable.py --density $MAX_DENSITY \
                        --elements $ELEMS --instances $INSTANCES_PER_SEQUENCE \
                        $STRATEGY_ARGS \
                        ">$OUTDIR/{}"
        done
    done
done
