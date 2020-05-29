#!/bin/bash

# Variable setting

listFile="random.poly nix_100 nix_102 random_100.poly"

if [ -d tmp ]
then
	rm -r tmp
fi
mkdir tmp

listResol=(32 64 128 256 512 1024 2048)

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
        echo -en "0\t0\t" >> tmp/temp_${file}_classic
        python3 subdivision.py $value -poly "$src" -hide; while read step duration remainder; do echo -en "${duration}\t" >> tmp/temp_${file}_classic; done < "subdivision_time.log"
        sed -i '$ s/.$/\n/' tmp/temp_${file}_classic
        echo -en "0\t" >> tmp/temp_${file}_clen
        python3 subdivision.py $value -poly "$src" -clen -hide; while read step duration remainder; do echo -en "${duration}\t" >> tmp/temp_${file}_clen; done < "subdivision_time.log"
        sed -i '$ s/.$/\n/' tmp/temp_${file}_clen
        python3 subdivision.py $value -poly "$src" -idct -hide; while read step duration remainder; do echo -en "${duration}\t" >> tmp/temp_${file}_idct; done < "subdivision_time.log"
        sed -i '$ s/.$/\n/' tmp/temp_${file}_idct
    done
    ((var++))
    echo -ne "$var/$n\r"
done
echo -ne '\n'

# Output writing

head="n"
methods="classic clen idct"

for file in $listFile
do
    for method in $methods
    do
    	head+="\t${file%.*}_${method}_ch\t${file%.*}_${method}_co\t${file%.*}_${method}_ev\t${file%.*}_${method}_su"
	done
    paste tmp/temp_${file}_classic tmp/temp_${file}_clen tmp/temp_${file}_idct > tmp/temp_${file}
done

if [ -f separate.out ]
then
	rm separate.out
fi
for value in ${listResol[@]}
do
	echo $value >> separate.out
done
for file in $listFile
do
	paste separate.out tmp/temp_$file > tmp/temp
	mv tmp/temp separate.out
done

sed -i '1 i\'${head} separate.out

# Cleaning

rm -r tmp