# Psi-X Policy Server


```bash
export run_dir=.runs/psix_finetune/mem.rss26r.flow1000.cosine.lr5.0e-05.b128.gpus8.2606091525
```

```bash
source .venv-psi/bin/activate
```

## Download checkpoint
```
download_ckpt nebula99 /data/hongyi/psi $run_dir
```

## Bringup the Psi-X Policy Server

Start Psi-X RTC Server
```bash
serve_psix --run-dir=$run_dir --port=8014 --dashboard --rtc
```

Or run the python directly:
```bash
python src/psi/deploy/serve_psix.py \
    --run-dir=$run_dir \
    --rtc \
    --host=localhost \
    --port=8014 \
    --dashboard
```

## Debugging Using Mock Client
```bash
export HF_HUB_OFFLINE=1
```

RTC version
```bash
python src/psi/deploy/mock_psix_client_rtc.py \
    --run-dir=$run_dir \
    --host=localhost \
    --port=8014 \
    --eps-idx=0 \
    --target-hz=30
```

HTTP version
```bash
python src/psi/deploy/mock_psix_client_http.py \
    --run-dir=$run_dir \
    --host=localhost \
    --port=8014 \
    --eps-idx=0 \
    --target-hz=30 \
    --stride=30
```

## Real G1 Deployment -- SONIC version

### Start Image Server on G1
```bash
ssh unitree@192.168.123.164
# 123

conda activate teleop
cd SONIC
python realsense_server.py
# sudo killall -9 videohub_pc4 && python realsense_server.py
```

Optionally, start image client to visualize the camera

```bash
cd real/teleop/image_server
conda activate psi_deploy
python image_client.py
# change port if needed
# vim real/teleop/image_server/image_client.py 
# ---> socket.connect(f"tcp://{server_ip}:5558")
```

### Start SONIC Real Controller

Change directory to SONIC:
```bash
cd /home/songlin/Projects/hongyi-wbc
```

```bash
cd gear_sonic_deploy
source scripts/setup_env.sh
./deploy.sh --input-type zmq real
```

Next: 

+ Press `]` to engage policy whenever ready
+ Press `Enter` to start `ZMQ` manager to accept client ctrl commands.

### Start RTC Client
```bash
python psix_rtc_sonic_client.py \
    --episode-dir /home/songlin/hfm/data/real_sonic_g1/pick_place_1/episode_0 \
    --prompts-json /home/songlin/hfm/data/real_sonic_g1/prompts.json
```

### Start non-RTC (HTTP) Client

```bash
python psix_rtc_sonic_client.py \
    --episode-dir /home/songlin/hfm/data/real_sonic_g1/pick_place_1/episode_0 \
    --prompts-json /home/songlin/hfm/data/real_sonic_g1/prompts.json
```

## Deploy Psi-0 on g1 -- SONIC+Neck version
``` bash
cd ~/Projects/hongyi-wbc

# 1.a. start camera server on g1 board
./auto_deploy.sh

# 1.b. start realsense camera on g1
conda activate ruohai (2号机)
cd GROOT...
python realsense_native_server.py
python realsense_viewer.py --server 192.168.123.164 --sub 

# 2. start sonic g1 controller
cd gear_sonic_deploy
./deploy.sh --input-type zmq real

# press ] to engage sonic (starding mode)
# press enter to listen client control command

# 3. Reset the initial pose
cd ~/Projects/hongyi-wbc
conda activate psi_deploy
python apply_initial_pose.py --distance near

# 4. RTC Version
cd ~/Projects/psi
source .venv-psi/bin/activate
# 4.1 bring up RTC server
serve_psi0_sonic \
	--policy psi0 \
	--port 8014 \
	--ckpt-step=40000 \
	--rtc \
	--run-dir=.runs/finetune/g1neck.sonic.flow1000.cosine.lr1.0e-04.b128.gpus8.2606211349
# 4.2 start RTC client
start psi0 server
cd ~/Projects/hongyi-wbc
conda activate psi_deploy
python psi_rtc_sonic_client.py \
	--include-neck \
	--instruction=pick_place.hippo.orange_basket

# 5. Non-RTC Version
# 5.1 bring up http server
serve_psi0_sonic_http \
    --policy psi0 \
    --port 8014 \
    --ckpt-step=40000 \
    --run-dir=.runs/finetune/g1neck30fps622.sonic.flow1000.cosine.lr1.0e-04.b128.gpus8.2606270114
	
# 5.2 start psi0 client (which talks to the sonic g1 controller)
python psi_sonic_client.py \
    --port 8014 \
    --include-neck \
    --instruction=pick_place.hippo.orange_basket
```

## Important Design choices

1. The Policy Server automatically clear image/state history, dashboard etc and will re-warmup upon client disconnection.
2. The inference code now use optimized video proprocessing instead of the default qwen3vl preprocessor.

## Troubleshooting

1. Missing field of `simple`

```
pydantic_core._pydantic_core.ValidationError: 2 validation errors for DynamicLaunchConfig
data.transform.repack.simple_g1
  Field required [type=missing, input_value={'dataset_name': 'simple+...eta.number_of_tasks']}}}, input_type=dict]
```

Fix: replace "simple" with "simple_g1" in the `run_config.json`