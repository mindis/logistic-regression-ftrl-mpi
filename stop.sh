ps -ef | grep lr_ftrl_mpi | awk '{ print $2 }' | sudo xargs kill -9
