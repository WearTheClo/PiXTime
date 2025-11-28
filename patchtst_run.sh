python run.py --model PatchTST --data ETTh1 --features M --target OT --seq_len 96 --pred_len 96 --en_layers 1 --n_heads 2
python run.py --model PatchTST --data ETTh1 --features M --target OT --seq_len 96 --pred_len 192 --en_layers 1 --n_heads 8
python run.py --model PatchTST --data ETTh1 --features M --target OT --seq_len 96 --pred_len 336 --en_layers 1 --n_heads 8
python run.py --model PatchTST --data ETTh1 --features M --target OT --seq_len 96 --pred_len 720 --en_layers 1 --n_heads 16

python run.py --model PatchTST --data ETTh2 --features M --target OT --seq_len 96 --pred_len 96 --en_layers 3 --n_heads 4
python run.py --model PatchTST --data ETTh2 --features M --target OT --seq_len 96 --pred_len 192 --en_layers 3 --n_heads 4
python run.py --model PatchTST --data ETTh2 --features M --target OT --seq_len 96 --pred_len 336 --en_layers 3 --n_heads 4
python run.py --model PatchTST --data ETTh2 --features M --target OT --seq_len 96 --pred_len 720 --en_layers 3 --n_heads 4

python run.py --model PatchTST --data ETTm1 --features M --target OT --seq_len 96 --pred_len 96 --en_layers 1 --n_heads 2
python run.py --model PatchTST --data ETTm1 --features M --target OT --seq_len 96 --pred_len 192 --en_layers 3 --n_heads 2
python run.py --model PatchTST --data ETTm1 --features M --target OT --seq_len 96 --pred_len 336 --en_layers 1 --n_heads 4
python run.py --model PatchTST --data ETTm1 --features M --target OT --seq_len 96 --pred_len 720 --en_layers 3 --n_heads 4

python run.py --model PatchTST --data ETTm2 --features M --target OT --seq_len 96 --pred_len 96 --en_layers 3 --n_heads 16
python run.py --model PatchTST --data ETTm2 --features M --target OT --seq_len 96 --pred_len 192 --en_layers 3 --n_heads 2
python run.py --model PatchTST --data ETTm2 --features M --target OT --seq_len 96 --pred_len 336 --en_layers 1 --n_heads 4
python run.py --model PatchTST --data ETTm2 --features M --target OT --seq_len 96 --pred_len 720 --en_layers 3 --n_heads 4

python run.py --model PatchTST --data electricity --root_path ./dataset/electricity/ --features M --target OT --seq_len 96 --pred_len 96 --en_layers 2 
python run.py --model PatchTST --data electricity --root_path ./dataset/electricity/ --features M --target OT --seq_len 96 --pred_len 192 --en_layers 2 
python run.py --model PatchTST --data electricity --root_path ./dataset/electricity/ --features M --target OT --seq_len 96 --pred_len 336 --en_layers 2 
python run.py --model PatchTST --data electricity --root_path ./dataset/electricity/ --features M --target OT --seq_len 96 --pred_len 720 --en_layers 2 

python run.py --model PatchTST --data exchange_rate --root_path ./dataset/exchange_rate/ --features M --target OT --seq_len 96 --pred_len 96 --en_layers 2 
python run.py --model PatchTST --data exchange_rate --root_path ./dataset/exchange_rate/ --features M --target OT --seq_len 96 --pred_len 192 --en_layers 2 
python run.py --model PatchTST --data exchange_rate --root_path ./dataset/exchange_rate/ --features M --target OT --seq_len 96 --pred_len 336 --en_layers 2 
python run.py --model PatchTST --data exchange_rate --root_path ./dataset/exchange_rate/ --features M --target OT --seq_len 96 --pred_len 720 --en_layers 2 

python run.py --model PatchTST --data traffic --root_path ./dataset/traffic/ --features M --target OT --seq_len 96 --pred_len 96 --d_model 256 --factor 3 --en_d_ff 1024 --en_layers 1 
python run.py --model PatchTST --data traffic --root_path ./dataset/traffic/ --features M --target OT --seq_len 96 --pred_len 192 --d_model 256 --factor 3 --en_d_ff 1024 --en_layers 1 
python run.py --model PatchTST --data traffic --root_path ./dataset/traffic/ --features M --target OT --seq_len 96 --pred_len 336 --d_model 256 --factor 3 --en_d_ff 1024 --en_layers 1 
python run.py --model PatchTST --data traffic --root_path ./dataset/traffic/ --features M --target OT --seq_len 96 --pred_len 720 --d_model 256 --factor 3 --en_d_ff 1024 --en_layers 1 

python run.py --model PatchTST --data weather --root_path ./dataset/weather/ --features M --target OT --seq_len 96 --pred_len 96 --en_layers 2 --n_heads 4 
python run.py --model PatchTST --data weather --root_path ./dataset/weather/ --features M --target OT --seq_len 96 --pred_len 192 --en_layers 2 --n_heads 16 
python run.py --model PatchTST --data weather --root_path ./dataset/weather/ --features M --target OT --seq_len 96 --pred_len 336 --en_layers 2 --n_heads 4 
python run.py --model PatchTST --data weather --root_path ./dataset/weather/ --features M --target OT --seq_len 96 --pred_len 720 --en_layers 2 --n_heads 4 