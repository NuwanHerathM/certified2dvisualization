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

echo "Number of files computed"
n=$(wc -w <<< "$listFile")
var=0
echo -ne "$var/$n\r"

for file in $listFile
do
	src="../polys/$file"
	for value in ${listResol[@]}
	do
		(/usr/bin/time -f "%e" python3 subdivision.py $value -poly "$src" -hide) 2>> tmp/temp_${file}_classic
		(/usr/bin/time -f "%e" python3 subdivision.py $value -poly "$src" -clen -hide) 2>> tmp/temp_${file}_clen
		(/usr/bin/time -f "%e" python3 subdivision.py $value -poly "$src" -idct -hide) 2>> tmp/temp_${file}_idct
	done
    echo -ne "$var/$n\r"
done
echo -ne '\n'

# Output writing

head="n"

for file in $listFile
do
	head+="\t${file%.*}_classic\t${file%.*}_clen\t${file%.*}_idct"
	paste tmp/temp_${file}_classic tmp/temp_${file}_clen tmp/temp_${file}_idct > tmp/temp_${file}
done

if [ -f timing.out ]
then
	rm timing.out
fi
for value in ${listResol[@]}
do
	echo $value >> timing.out
done
for file in $listFile
do
	paste timing.out tmp/temp_$file > tmp/temp
	mv tmp/temp timing.out
done

sed -i '1 i\'${head} timing.out

# Cleaning

rm -r tmp
