docker build --tag clpcnet --build-arg HTK=htk/ . && \
docker run -it --rm --name "clpcnet" --shm-size 32g --gpus all \
  -v /home/mrm5248/clpcnet/runs:/clpcnet/runs \
  -v /home/mrm5248/clpcnet/data:/clpcnet/data \
  clpcnet:latest \
  /opt/conda/envs/clpcnet/bin/python -m clpcnet.pitch --dataset vctk-promo --gpu 2
  # /opt/conda/envs/clpcnet/bin/python -m clpcnet.preprocess.augment --dataset vctk-promo --gpu 2
