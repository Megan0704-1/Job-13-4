# Env create using conda
- current development version: 3.12
```bash
# To invoke env run
conda env create -f $MLIRGYM_ROOT/envs/env_3.12.yaml

# To export your stable version
conda env export > $MLIRGYM_ROOT/envs/env_3.xx.yaml
```

# Env create using Dockerfile
> Stable builds, but generate approx 150 GB image (another TODO)
```bash
# to build, we need MPS lab github token since its a private repo
docker build --build-arg BUILDKIT_INLINE_CACHE=1 --build-arg GITHUB_TOKEN=<your-token> --cache-from=<image-name>:<tag-name> -t mlir-gym .

# test
docker run --rm -it --gpus all <image_id> /bin/bash
```
