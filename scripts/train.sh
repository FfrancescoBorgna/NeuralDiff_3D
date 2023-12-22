#bash scripts/train.sh P01_01
ora=$(date)
echo "Training of $1 beginning at $ora" >> /scratch/fborgna/NeuralDiff/scripts/train_log
VID=$1; CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python train.py \
  --root data/Epic_converted \
  --vid $VID \
  --exp_name rel/$VID \
  --train_ratio 1 --num_epochs 12 \
  #--ckpt_path  /scratch/fborgna/NeuralDiff/ckpts/rel/$VID/epoch=5.ckpt
#--num_gpus 2
 
#--root data/Epic_converted \
#--root /scratch/fborgna/NeuralDiff/data/EPIC-Diff \

#VID=$1; CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0,1 python train.py \
current_data=$(date)
echo "Finished Training of $1 at $current_data" >> /scratch/fborgna/NeuralDiff/scripts/train_log