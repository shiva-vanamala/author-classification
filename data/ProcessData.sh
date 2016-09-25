head -n -366 raw/*.txt | tail -n +37 >> processed/out.txt
for d in [0-9][0-9]
do
    echo $d
done

for fn in `cat filenames.txt`; do
    echo "the next file is $fn"
    cat $fn
done
