# for i in `seq 1 9`;do ssh aturquetil@a406-0$i ~/challenge/together/launch_job $i 500 36000;done

cd /cal/homes/aturquetil/challenge/together
/cal/homes/aturquetil/anaconda3/bin/python job.py $1 $2 $3
