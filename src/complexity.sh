#!/bin/bash

# Variable setting

listFile="random.poly nix_100 nix_102 random_100.poly"

rm -rf tmp
mkdir tmp

listResol=(32 64 128 256 512 1024 2048)

outfile="../output/complexity.out"

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
        python3 main.py $value -poly "$src" -hide; while read step complexity remainder; do echo -en "${complexity}\t" >> tmp/temp_${file}_classic; done < "complexity.log"
        sed -i '$ s/.$/\n/' tmp/temp_${file}_classic
        python3 main.py $value -poly "$src" -clen -hide; while read step complexity remainder; do echo -en "${complexity}\t" >> tmp/temp_${file}_clen; done < "complexity.log"
        sed -i '$ s/.$/\n/' tmp/temp_${file}_clen
        python3 main.py $value -poly "$src" -idct -hide; while read step complexity remainder; do echo -en "${complexity}\t" >> tmp/temp_${file}_idct; done < "complexity.log"
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
    	head+="\t${file%.*}_${method}_ev\t${file%.*}_${method}_su"
	done
    paste tmp/temp_${file}_classic tmp/temp_${file}_clen tmp/temp_${file}_idct > tmp/temp_${file}
done

if [ -f $outfile ]
then
	rm $outfile
fi
for value in ${listResol[@]}
do
	echo $value >> $outfile
done
for file in $listFile
do
	paste $outfile tmp/temp_$file > tmp/temp
	mv tmp/temp $outfile
done

sed -i '1 i\'${head} $outfile

# Cleaning

rm -r tmp