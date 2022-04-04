docker build --tag clpcnet --build-arg HTK=htk/ . && \
docker run -it --rm --name "clpcnet" --shm-size 32g --gpus all \
  -v /home/mrm5248/clpcnet/runs:/clpcnet/runs \
  -v /home/mrm5248/clpcnet/data:/clpcnet/data \
  clpcnet:latest \
  /opt/conda/envs/clpcnet/bin/python -m clpcnet.train --name 22050 --dataset vctk-promo --gpu 1
