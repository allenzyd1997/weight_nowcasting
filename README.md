
### An Important thing in the nohup and DDP launch command

```
CUDA_VISIBLE_DEVICES=2,5,6,7 python -m torch.distributed.launch --nproc_per_node 4 --master_port=26500  main.py > May19.log 2>&1 &
```
如果master_port不被指明， nohup会关闭端口 DDP多进程之间无法通信。
CUDA必须放在nohup 之前，因为他是环境参数 并不是一个nohup去执行的文件.