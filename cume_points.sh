#!/usr/bin/env bash

# http://basketballvalue.com/downloads.php



find . -name 'playbyplay2*txt' | xargs -I {} dos2unix {}


cat playbyplay20120510040.txt | head -100 | \
gawk -F$'\t' -v OFS=$'\t' '
$1 !~ /GameID/ {
    teama = substr($1,9,3)
    teamb = substr($1,12,3)
}
$1 !~ /GameID/ && $4 ~ /End of 4th Quarter/ {
    delete cume_pts_a
    delete cume_pts_b
}
$1 !~ /GameID/ && $4 ~ /\([[:digit:]]+ PTS\)/{
    # extract scorer, new score of scorer, new score of opponent
    where = match($4,/\[([[:alpha:]]+) ([[:digit:]]+)-([[:digit:]]+)\]/,arr)
    scorer = arr[1] #substr($4,2,3)
    k = $1 OFS scorer
    if (scorer == teama) {
        cume_pts_a[arr[2]] = 1
    } else {
        cume_pts_b[arr[2]] = 1
    }
}'
