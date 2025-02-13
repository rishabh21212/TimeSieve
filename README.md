# USE PYTHON 3.10

A few next steps - 
1. Please fix the validate and test methods the same way train has been fixed
2. Verify full functionality after training

Command - 

python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ --data_path weather.csv --model_id weather_96_96 --model TimeSieve --data custom --features M --seq_len 96 --label_len 48 --pred_len 48 --e_layers 2 --d_layers 1 --factor 3 --enc_in
 21 --dec_in 21 --c_out 21 --d_model 48 --d_ff 96 --top_k 5 --des 'Exp' --itr 1

Please modify the command for better use. This is an example that works.