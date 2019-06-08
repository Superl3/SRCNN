# Super Resolution Conv Neural Network

SRCNN : PSNR 32\
normalization images
warning msg disabled
lower learning rate as 1e-4
clip values as 0~1


SRCNN_RNN : second run, PSNR 32

added bias on h->h 
normalization images
tested with 291 datasets
warning msg disabled
lower learning rate as 1e-4
clip values as 0~1  

lower learning rate as 5e-5 -> 1e-4 is still to big, 1e-5 is so small, should find best value between these.
removed bias on all layer -> make loss more stablized too
change stddev as 1e-1

# 
## TODO
- and more...