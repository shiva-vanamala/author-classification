head -n -366 raw/*.txt | tail -n +37 >> processed/out.txt
for ((i=1;i<=100;i++));
do
    echo $i
done
#for fn in `cat filenames.txt`; do
#    echo "the next file is $fn"
#    cat $fn
#done
