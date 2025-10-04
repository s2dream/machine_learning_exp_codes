

# 멀티 GPU / DDP 학습
torchrun --standalone --nnodes=1 --nproc_per_node=4 tasks/trainer.py --config ./configs/example.yaml


# 싱글 GPU or CPU 학습
 python tasks/trainer.py --config ./configs/example.yaml


# 추론
python tasks/inference.py --config ./configs/example.yaml \
  --checkpoint ./checkpoints/best.pth \
  --output ./preds.csv