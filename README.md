# nano-research-gpt
an experimental testbed for novel approaches to machine learning in transformer architectures.
Please propose changes by either email or by opening a new issue.

the current best case work is the tapetransformer. it appears to smoothly descend and not suffer from mode collapse as badly.
a model with only 1406152 trainable params(same as what is in the ipynb) was trained on shakespeare and smoothly and rapidly descended from 4 through 3 to 2 through 2 to 1 through 1 to 0.2(usually mode collapse by this point is observed)
additionally training progresses much more rapidly, reaching the loss of 0.1 around 500 iterations, while for a normal gpt this can take around 10x as many iterations.
