echo "exps running..."
CUDA_VISIBLE_DEVICES=2 python train_eval.py --data GLOBO  --model tmgm --time_thresh 600  --version 600
CUDA_VISIBLE_DEVICES=2 python train_eval.py --data GLOBO  --model tmgm --time_thresh 1200 --version 1200 
CUDA_VISIBLE_DEVICES=0 python train_eval.py --data GLOBO  --model tmgm --time_thresh 1800 --version 1800 
CUDA_VISIBLE_DEVICES=0 python train_eval.py --data GLOBO  --model tmgm --time_thresh 2400 --version 2400 
CUDA_VISIBLE_DEVICES=0 python train_eval.py --data GLOBO  --model tmgm --time_thresh 600 --time_p 0.8 --version th600p0.8 
CUDA_VISIBLE_DEVICES=0 python train_eval.py --data GLOBO  --model tmgm --time_thresh 600 --time_p 0.7 --version th600p0.7 
wait
CUDA_VISIBLE_DEVICES=0 python train_eval.py --data GLOBO  --model tmgm --time_thresh 600 --time_p 0.6 --version th600p0.6 
CUDA_VISIBLE_DEVICES=2 python train_eval.py --data GLOBO  --model tmgm --time_thresh 600 --time_p 0.5 --version th600p0.5 
CUDA_VISIBLE_DEVICES=2 python train_eval.py --data GLOBO  --model tmgm --time_thresh 600 --time_p 0.4 --version th600p0.4 
CUDA_VISIBLE_DEVICES=2 python train_eval.py --data GLOBO  --model tmgm --time_thresh 600 --time_p 0.95 --version th600p0.95 