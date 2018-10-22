n_samples=100000
for dim in {1..10}; do
  for seed in {1..10}; do
    echo $dim $seed
    #python calc_pi.py /data/mnist/crop/vector_images.h5 v -n $n_samples -d $dim --seed $seed --save_folder /data/mnist/crop/info_results -g 1

    #python calc_pi.py /data/mnist/crop/images.h5 i -n $n_samples -d $dim --seed $seed --save_folder /data/mnist/crop/info_results -g 1
    python calc_pi.py /data/imnist/crop/images.h5 i -n $n_samples -d $dim --seed $seed --save_folder /data/imnist/crop/info_results -g 1

    #for f in R32_B6.h5 R32_B7.h5 R32_B8.h5; do
    #  python calc_pi.py /data/NWB/R32/simple/$f t -n $n_samples -d $dim --seed $seed --save_folder /data/NWB/R32/info_results
    #done
  done
done
