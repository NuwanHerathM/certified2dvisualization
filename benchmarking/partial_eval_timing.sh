#!/bin/bash

# Time comparison of the partial evaluation with error with Horner vs with IDCT

# Variable setting

listFile="random_20_neg.poly random_30_neg.poly random_40_neg.poly random_50_neg.poly random_100_neg.poly"
listResol=(128 1024 2048 4096)
listMethod="horner dct"

rm -rf tmp
mkdir tmp

# Main script

for method in $listMethod
do
  for file in $listFile
  do
    outfile0="tmp/tmp_${method}_${file%.*}"
    for value in ${listResol[@]}
    do
      src="../polys/$file"
      python3 partial_eval_${method}.py $value $src >> $outfile0
    done
  done
done

# Output writing

head=""
for file in $listFile
do
  head+="\t${file%.*}"
done

for method in $listMethod
do
  outfile1=tmp/partial_eval_${method}.tsv
  rm -f $outfile1
  paste tmp/tmp_${method}_random_20_neg tmp/tmp_${method}_random_30_neg tmp/tmp_${method}_random_40_neg tmp/tmp_${method}_random_50_neg tmp/tmp_${method}_random_100_neg > $outfile1
  sed -i '1 i\'${head} $outfile1
done


echo "" > column0
for value in ${listResol[@]}
do
  echo $value >> column0
done

for method in $listMethod
do
  outfile1=tmp/partial_eval_${method}.tsv
  outfile2=output/partial_eval_${method}.tsv
  paste column0 $outfile1 > $outfile2
done

# Cleaning

rm -r tmp
rm column0