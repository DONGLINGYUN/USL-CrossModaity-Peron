for trial in 1 2 3 4 5 6 7 8 9 10
do
CUDA_VISIBLE_DEVICES=0 python label_regdb4.py -b 256 -a agw -d  regdb_rgb --iters 100 --momentum 0.1 --eps 0.6 --num-instances 16 --trial $trial
done
echo 'Do
python train_regdb.py/train_regdb_baseline.py -b 32 -a agw -d  regdb_rgb --iters 100 --momentum 0.1 --eps 0.3 --num-instances 16
