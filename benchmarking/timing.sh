#!/bin/bash

# Variable setting

listFile="random.poly random_50.poly random_100.poly"
listResol=(128 256 512 1024 2048)
listPower=(7 8 9 10 11)

if [[ $USER == "nherathm " ]]
then
    matlab_option="-c 27000@localhost"
else
    matlab_option=""
fi

# Benchmarking

echo "Number of files computed"
n=$(wc -w <<< "$listFile")
var=0
echo -ne "$var/$n\r"

head="n"
for value in ${listResol[@]}
do
    head+="\t$value"
done
head+="\n"

for file in $listFile
do
    src="../polys/$file"
    limit=1800
    previous_exceeds_time_limit=false
    outfile=output/${file%.*}.tsv
    rm -f $outfile
    echo -ne $head > $outfile

    # Julia
    echo -ne "Julia" >> $outfile
    for power in ${listPower[@]}
    do
        echo -ne "\t" >> $outfile
        if ! $previous_exceeds_time_limit
        then
            (time timeout ${limit}s julia --sysimage sys_plots.so compare_julia.jl $power "$src" 2> /dev/null) 2> tortue
            cputime=$(python3 extract_time.py)
            if (( $(echo "$cputime < $limit" | bc -l) ))
            then
                echo -n $cputime >> $outfile
            else
                previous_exceeds_time_limit=true
                echo -n ">$limit" >> $outfile
            fi
        else
            echo -n ">$limit" >> $outfile
        fi
    done
    echo -ne "\n" >> $outfile

    # scikit
    previous_exceeds_time_limit=false
    echo -ne "scikit" >> $outfile
    for value in ${listResol[@]}
    do
        echo -ne "\t" >> $outfile
        if ! $previous_exceeds_time_limit
        then
            (time timeout ${limit}s python3 compare_scikit.py $value "$src" 2> /dev/null) 2> tortue
            cputime=$(python3 extract_time.py)
            if (( $(echo "$cputime < $limit" | bc -l) ))
            then
                echo -n $cputime >> $outfile
            else
                previous_exceeds_time_limit=true
                echo -n ">$limit" >> $outfile
            fi
        else
            echo -n ">$limit" >> $outfile
        fi
    done
    echo -ne "\n" >> $outfile

    # Matlab
    previous_exceeds_time_limit=false
    echo -ne "Matlab" >> $outfile
    cp ${file%.*}.m polynome.m
    for value in ${listResol[@]}
    do
        echo "n=$value;" > resolution.m
        echo -ne "\t" >> $outfile
        if ! $previous_exceeds_time_limit
        then
            (time timeout $((limit+20))s matlab $matlab_option -batch test 2>&1 /dev/null) 2> tortue
            cputime=$(python3 extract_time.py)
            if (( $(echo "$cputime < $limit" | bc -l) ))
            then
                echo -n $cputime >> $outfile
            else
                previous_exceeds_time_limit=true
                echo -n ">$limit" >> $outfile
            fi
        else
            echo -n ">$limit" >> $outfile
        fi
    done
    echo -ne "\n" >> $outfile

    # Our algorithm: subdivision
    previous_exceeds_time_limit=false
    echo -ne "subdivision" >> $outfile
    for value in ${listResol[@]}
    do
        echo -ne "\t" >> $outfile
        if ! $previous_exceeds_time_limit
        then
            (time timeout ${limit}s python3 ../src/main.py $value -poly "$src" -hide -save 2> /dev/null) 2> tortue
            cputime=$(python3 extract_time.py)
            if (( $(echo "$cputime < $limit" | bc -l) ))
            then
                echo -n $cputime >> $outfile
            else
                previous_exceeds_time_limit=true
                echo -n ">$limit" >> $outfile
            fi
        else
            echo -n ">$limit" >> $outfile
        fi
    done
    echo -ne "\n" >> $outfile

    # Our algorithm: taylor
    previous_exceeds_time_limit=false
    echo -ne "taylor" >> $outfile
    for value in ${listResol[@]}
    do
        echo -ne "\t" >> $outfile
        if ! $previous_exceeds_time_limit
        then
            (time timeout ${limit}s python3 ../src/main.py $value -poly "$src" -taylor -hide -save 2> /dev/null) 2> tortue
            cputime=$(python3 extract_time.py)
            if (( $(echo "$cputime < $limit" | bc -l) ))
            then
                echo -n $cputime >> $outfile
            else
                previous_exceeds_time_limit=true
                echo -n ">$limit" >> $outfile
            fi
        else
            echo -n ">$limit" >> $outfile
        fi
    done
    echo -ne "\n" >> $outfile

    ((var++))
    echo -ne "$var/$n\r"
done
echo -ne '\n'

# Cleaning

rm -f image.png tortue polynome.m resolution.m