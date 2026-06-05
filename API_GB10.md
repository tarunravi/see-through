# See-through GB10 API Wrapper

This fork adds a small FastAPI wrapper for running See-through as a job service on
an NVIDIA DGX Spark/GB10 host. The wrapper does not change the upstream
See-through inference code; it shells out to `inference/scripts/inference_psd.py`
and exposes stable upload, status, and artifact endpoints.

## Build

```bash
docker build -f Dockerfile.gb10 -t avatar-see-through:gb10 .
```

## Run

```bash
docker run -d --name avatar-see-through-api --gpus all \
  -p 18080:18080 \
  -e SEE_THROUGH_JOBS_ROOT=/workspace/jobs \
  -v "$HOME/avatar/see-through-spike/jobs:/workspace/jobs" \
  -v "$HOME/avatar/see-through-spike/hf-cache:/workspace/hf-cache" \
  --entrypoint python3 \
  avatar-see-through:gb10 \
  /workspace/see-through/api/see_through_api.py --host 0.0.0.0 --port 18080
```

## Endpoints

- `GET /health`
- `POST /v1/decompositions`
- `GET /v1/decompositions/{job_id}`
- `GET /v1/decompositions/{job_id}/artifacts/{artifact_path}`

Example smoke job:

```bash
curl -X POST http://10.0.0.194:18080/v1/decompositions \
  -F image=@image.png \
  -F resolution=512 \
  -F resolution_depth=256 \
  -F inference_steps=1 \
  -F group_offload=true \
  -F save_to_psd=false \
  -F tblr_split=true
```

Validated on `spark-20d3` with CUDA available through PyTorch
`2.11.0+cu130` and device `NVIDIA GB10`.
