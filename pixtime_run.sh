python run.py --model PiXTime --data ETTh1 --features M --target OT --seq_len 96 --pred_len 96 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 1 --de_layers 1
python run.py --model PiXTime --data ETTh1 --features M --target OT --seq_len 96 --pred_len 192 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 1 --de_layers 1
python run.py --model PiXTime --data ETTh1 --features M --target OT --seq_len 96 --pred_len 336 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 1 --de_layers 1
python run.py --model PiXTime --data ETTh1 --features M --target OT --seq_len 96 --pred_len 720 --d_model 128 --en_d_ff 512 --de_d_ff 512 --en_layers 1 --de_layers 1

python run.py --model PiXTime --data ETTh2 --features M --target OT --seq_len 96 --pred_len 96 --d_model 256 --en_d_ff 1024 --de_d_ff 1024 --en_layers 1 --de_layers 1
python run.py --model PiXTime --data ETTh2 --features M --target OT --seq_len 96 --pred_len 192 --d_model 128 --en_d_ff 512 --de_d_ff 512 --en_layers 1 --de_layers 1
python run.py --model PiXTime --data ETTh2 --features M --target OT --seq_len 96 --pred_len 336 --d_model 128 --en_d_ff 512 --de_d_ff 512 --en_layers 1 --de_layers 1
python run.py --model PiXTime --data ETTh2 --features M --target OT --seq_len 96 --pred_len 720 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 1 --de_layers 1

python run.py --model PiXTime --data ETTm1 --features M --target OT --seq_len 96 --pred_len 96 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 1 --de_layers 1
python run.py --model PiXTime --data ETTm1 --features M --target OT --seq_len 96 --pred_len 192 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 1 --de_layers 1
python run.py --model PiXTime --data ETTm1 --features M --target OT --seq_len 96 --pred_len 336 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 1 --de_layers 1
python run.py --model PiXTime --data ETTm1 --features M --target OT --seq_len 96 --pred_len 720 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 1 --de_layers 1

python run.py --model PiXTime --data ETTm2 --features M --target OT --seq_len 96 --pred_len 96 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 1 --de_layers 1
python run.py --model PiXTime --data ETTm2 --features M --target OT --seq_len 96 --pred_len 192 --d_model 256 --en_d_ff 1024 --de_d_ff 1024 --en_layers 1 --de_layers 1
python run.py --model PiXTime --data ETTm2 --features M --target OT --seq_len 96 --pred_len 336 --d_model 256 --en_d_ff 1024 --de_d_ff 1024 --en_layers 1 --de_layers 1
python run.py --model PiXTime --data ETTm2 --features M --target OT --seq_len 96 --pred_len 720 --d_model 256 --en_d_ff 1024 --de_d_ff 1024 --en_layers 1 --de_layers 1

python run.py --model PiXTime --data electricity --root_path ./dataset/electricity/ --features M --target OT --seq_len 96 --pred_len 96 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 2 --de_layers 2
python run.py --model PiXTime --data electricity --root_path ./dataset/electricity/ --features M --target OT --seq_len 96 --pred_len 192 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 2 --de_layers 2
python run.py --model PiXTime --data electricity --root_path ./dataset/electricity/ --features M --target OT --seq_len 96 --pred_len 336 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 2 --de_layers 2
python run.py --model PiXTime --data electricity --root_path ./dataset/electricity/ --features M --target OT --seq_len 96 --pred_len 720 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 2 --de_layers 2

python run.py --model PiXTime --data exchange_rate --root_path ./dataset/exchange_rate/ --features M --target OT --seq_len 96 --pred_len 96 --d_model 64 --en_d_ff 256 --de_d_ff 256 --en_layers 1 --de_layers 1
python run.py --model PiXTime --data exchange_rate --root_path ./dataset/exchange_rate/ --features M --target OT --seq_len 96 --pred_len 192 --d_model 64 --en_d_ff 256 --de_d_ff 256 --en_layers 1 --de_layers 1
python run.py --model PiXTime --data exchange_rate --root_path ./dataset/exchange_rate/ --features M --target OT --seq_len 96 --pred_len 336 --d_model 64 --en_d_ff 256 --de_d_ff 256 --en_layers 1 --de_layers 1
python run.py --model PiXTime --data exchange_rate --root_path ./dataset/exchange_rate/ --features M --target OT --seq_len 96 --pred_len 720 --d_model 128 --en_d_ff 512 --de_d_ff 512 --en_layers 1 --de_layers 1

python run.py --model PiXTime --data traffic --root_path ./dataset/traffic/ --features M --target OT --seq_len 96 --pred_len 96 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 1 --de_layers 1
python run.py --model PiXTime --data traffic --root_path ./dataset/traffic/ --features M --target OT --seq_len 96 --pred_len 192 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 1 --de_layers 1
python run.py --model PiXTime --data traffic --root_path ./dataset/traffic/ --features M --target OT --seq_len 96 --pred_len 336 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 1 --de_layers 1
python run.py --model PiXTime --data traffic --root_path ./dataset/traffic/ --features M --target OT --seq_len 96 --pred_len 720 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 1 --de_layers 1

python run.py --model PiXTime --data weather --root_path ./dataset/weather/ --features M --target OT --seq_len 96 --pred_len 96 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 1 --de_layers 1
python run.py --model PiXTime --data weather --root_path ./dataset/weather/ --features M --target OT --seq_len 96 --pred_len 192 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 1 --de_layers 1
python run.py --model PiXTime --data weather --root_path ./dataset/weather/ --features M --target OT --seq_len 96 --pred_len 336 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 1 --de_layers 1
python run.py --model PiXTime --data weather --root_path ./dataset/weather/ --features M --target OT --seq_len 96 --pred_len 720 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 1 --de_layers 1