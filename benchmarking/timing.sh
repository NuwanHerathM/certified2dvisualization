#!/bin/bash

# For each file in listFile create a table with computation times
# for each method and each resolution in listResol in output/.
# Save the graphs in images/ except for sub and taylor,
# for which they are in ../output/.

# For ImplicitEquations from Julia, the resolution is given by a 
# power of two in listPower.

# A timeout is defined in the variable limit.

# ===============================================================

# Variable setting

# listFile="random_30_neg.poly random_40_neg.poly random_50_neg.poly random_100_neg.poly dfold_66.poly"
# listFile="random_20_neg.poly random_40_neg.poly random_20_neg_ell.poly random_40_neg_ell.poly"
listFile="random_100_neg.poly"
listResol=(128) # 256 512 1024 2048 4096)
listPower=(7) # 8 9 10 11 12)

if [[ $USER == "nherathm " ]]
then
    matlab_option="-c 27000@localhost"
else
    matlab_option=""
fi

# Benchmarking

limit=900 # timeout
mkdir -p output
mkdir -p images

echo "Number of files computed"
n=$(wc -w <<< "$listFile")
var=0
echo -ne "$var/$n\r"

head="n" # header with the resolutions
for value in ${listResol[@]}
do
    head+="\t$value"
done
head+="\n"

export LC_ALL=C # for time output independent of the language

for file in $listFile
do
    filename=${file%.*}
    src="../polys/$file"
    outfile=output/${filename}.tsv
    rm -f $outfile
    echo -ne $head > $outfile

    # Julia
    previous_exceeds_time_limit=false
    echo -ne "Julia" >> $outfile
    for power in ${listPower[@]}
    do
        echo -ne "\t" >> $outfile
        if ! $previous_exceeds_time_limit # if the previous execution did not exceed the timeout, run the program
        then
            # sum of user and system time
            tmp_cputime=$(/usr/bin/time -f %U+%S timeout ${limit}s julia --sysimage sys_plots.so compare_julia.jl $power "$src" 2>&1)
            status=$?
            # evaluation of the sum
            cputime=$(echo $tmp_cputime | bc -l)
            if [ $status -eq 0 ] && (( $(echo "$cputime < $limit" | bc -l) ))
            then
                printf "%.1f" $cputime >> $outfile
                cp image.png images/${filename}_${power}_julia.png
            else
                previous_exceeds_time_limit=true
                echo -n ">$limit" >> $outfile
            fi
        else # if the previous execution exceeded the timeout, do not run the program
            echo -n ">$limit" >> $outfile
        fi
        sleep 2
    done
    echo -ne "\n" >> $outfile

    # scikit
    previous_exceeds_time_limit=false
    echo -ne "scikit" >> $outfile
    for value in ${listResol[@]}
    do
        echo -ne "\t" >> $outfile
        if ! $previous_exceeds_time_limit # if the previous execution did not exceed the timeout, run the program
        then
            # sum of user and system time
            tmp_cputime=$(/usr/bin/time -f %U+%S timeout ${limit}s python3 compare_scikit.py $value "$src" 2>&1) 
            status=$?
            # evaluation of the sum
            cputime=$(echo $tmp_cputime | bc -l)
            if [ $status -eq 0 ] && (( $(echo "$cputime < $limit" | bc -l) ))
            then
                printf "%.1f" $cputime >> $outfile
                cp image.png images/${filename}_${value}_scikit.png
            else
                previous_exceeds_time_limit=true
                echo -n ">$limit" >> $outfile
            fi
        else # if the previous execution exceeded the timeout, do not run the program
            echo -n ">$limit" >> $outfile
        fi
        sleep 2
    done
    echo -ne "\n" >> $outfile

    # Matlab
    previous_exceeds_time_limit=false
    echo -ne "Matlab" >> $outfile
    cp ${filename}.m polynome.m
    for value in ${listResol[@]}
    do
        echo "n=$value;" > resolution.m
        echo -ne "\t" >> $outfile
        if ! $previous_exceeds_time_limit # if the previous execution did not exceed the timeout, run the program
        then
            # sum of user and system time
            tmp_cputime=$((/usr/bin/time -f %U+%S timeout $((limit+20))s matlab $matlab_option -batch compare_matlab) 2>&1 > /dev/null)
            status=$?
            # evaluation of the sum
            cputime=$(echo $tmp_cputime | bc -l)
            if [ $status -eq 0 ] && (( $(echo "$cputime < $limit" | bc -l) ))
            then
                printf "%.1f" $cputime >> $outfile
                cp image.png images/${file}_${value}_matlab.png
            else
                previous_exceeds_time_limit=true
                echo -n ">$limit" >> $outfile
            fi
        else # if the previous execution exceeded the timeout, do not run the program
            echo -n ">$limit" >> $outfile
        fi
        sleep 2
    done
    echo -ne "\n" >> $outfile

    # Our algorithm: subdivision
    previous_exceeds_time_limit=false
    echo -ne "subdivision" >> $outfile
    for value in ${listResol[@]}
    do
        echo -ne "\t" >> $outfile
        if ! $previous_exceeds_time_limit # if the previous execution did not exceed the timeout, run the program
        then
            # sum of user and system time
            tmp_cputime=$(/usr/bin/time -f %U+%S timeout ${limit}s python3 ../src/main.py $value -poly "$src" -hide -save 2>&1)
            status=$?
            # evaluation of the sum
            cputime=$(echo $tmp_cputime | bc -l)
            if [ $status -eq 0 ] && (( $(echo "$cputime < $limit" | bc -l) ))
            then
                printf "%.1f" $cputime >> $outfile
            else
                previous_exceeds_time_limit=true
                echo -n ">$limit" >> $outfile
            fi
        else # if the previous execution exceeded the timeout, do not run the program
            echo -n ">$limit" >> $outfile
        fi
        sleep 2
    done
    echo -ne "\n" >> $outfile

    # Our algorithm: taylor
    previous_exceeds_time_limit=false
    echo -ne "taylor" >> $outfile
    for value in ${listResol[@]}
    do
        echo -ne "\t" >> $outfile
        if ! $previous_exceeds_time_limit # if the previous execution did not exceed the timeout, run the program
        then
            # sum of user and system time
            tmp_cputime=$(/usr/bin/time -f %U+%S timeout ${limit}s python3 ../src/main.py $value -poly "$src" -taylor -hide -save 2>&1)
            status=$?
            # evaluation of the sum
            cputime=$(echo $tmp_cputime | bc -l)
            if [ $status -eq 0 ] && (( $(echo "$cputime < $limit" | bc -l) ))
            then
                printf "%.1f" $cputime >> $outfile
            else
                previous_exceeds_time_limit=true
                echo -n ">$limit" >> $outfile
            fi
        else # if the previous execution exceeded the timeout, do not run the program
            echo -n ">$limit" >> $outfile
        fi
        sleep 2
    done
    echo -ne "\n" >> $outfile

    ((var++))
    echo -ne "$var/$n\r"
done
echo -ne '\n'

unset LC_ALL

# Cleaning

rm -f image.png polynome.m resolution.m