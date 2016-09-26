for ((i=1;i<=28;i++));
do
    head -n -366 raw/Author$i.txt | tail -n +37 >> processed/Author$i.txt
done
