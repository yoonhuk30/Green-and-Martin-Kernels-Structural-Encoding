python main.py --cfg configs/GKSE/ckt-bench301-GKSE.yaml  accelerator "cuda:0" seed 0  &
python main.py --cfg configs/GKSE/ckt-bench301-GKSE.yaml  accelerator "cuda:1" seed 1  &
python main.py --cfg configs/GKSE/ckt-bench301-GKSE.yaml  accelerator "cuda:2" seed 2  &
python main.py --cfg configs/GKSE/ckt-bench301-GKSE.yaml  accelerator "cuda:3" seed 3  &
wait

python main.py --cfg configs/MKSE/ckt-bench301-MKSE.yaml  accelerator "cuda:0" seed 0  &
python main.py --cfg configs/MKSE/ckt-bench301-MKSE.yaml  accelerator "cuda:1" seed 1  &
python main.py --cfg configs/MKSE/ckt-bench301-MKSE.yaml  accelerator "cuda:2" seed 2  &
python main.py --cfg configs/MKSE/ckt-bench301-MKSE.yaml  accelerator "cuda:3" seed 3  &
wait
