#!/bin/bash

# Variables setting

listFile="random.poly nix_100 nix_102 random_100.poly"

if [ -d tmp ]
then
	rm -r tmp
fi
mkdir tmp

listResol=(32 64 128) #256 512 1024 2048)

# Computation

for file in $listFile
do
	for value in ${listResol[@]}
	do
		src="../polys/$file"
		(/usr/bin/time -f "%e" python3 subdivision.py $value -poly "$src" -hide) 2>> tmp/temp_${file}_classic
		(/usr/bin/time -f "%e" python3 subdivision.py $value -poly "$src" -clen -hide) 2>> tmp/temp_${file}_clen
		(/usr/bin/time -f "%e" python3 subdivision.py $value -poly "$src" -idct -hide) 2>> tmp/temp_${file}_idct
	done
done

# Output writing

head="n"

for file in $listFile
do
	head+="\t${file%.*}_classic\t${file%.*}_clen\t${file%.*}_idct"
	paste tmp/temp_${file}_classic tmp/temp_${file}_clen tmp/temp_${file}_idct > tmp/temp_${file}
done

if [ -f output ]
then
	rm output
fi
for value in ${listResol[@]}
do
	echo $value >> output
done
for file in $listFile
do
	paste output tmp/temp_$file > tmp/temp
	mv tmp/temp output
done

sed -i '1 i\'${head} output

# Cleaning

rm -r tmp
