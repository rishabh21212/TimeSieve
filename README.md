# Use Python 3.10 

## Next Steps:


NOTE : FOR LONG TERM FORECASTING, WE HAVE FIXED THE TUPLE ERROR, BUT FOR OTHER TASKS, THAT NEEDS TO BE FIXED, PLEASE LOOK AT LONG TERM FORECASTING CODE AND DO THE SAME IF WE REQUIRE OTHER TASKS.

### 1. Verify Full Functionality After Training(TBD)

Run the following command to validate the changes:

```sh
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path weather.csv \
  --model_id weather_96_96 \
  --model TimeSieve \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 48 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model 48 \
  --d_ff 96 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1
```

Single line script 

```sh
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ --data_path weather.csv --model_id weather_96_96 --model TimeSieve --data custom --features M --seq_len 96 --label_len 48 --pred_len 48 --e_layers 2 --d_layers 1 --factor 3 --enc_in
 21 --dec_in 21 --c_out 21 --d_model 48 --d_ff 96 --top_k 5 --des 'Exp' --itr 1
 ```

### 2. Modify the Command for Better Usability

Consider making parameters configurable through a script or argument parsing to simplify reusability and adaptability.

### Summary
- Updated `train()`, `vali()`, and `test()` to properly handle three return values.
- Fixed potential `TypeError` issues.
- Provided a test command for verification.
- Suggested making parameters configurable for better usability.

### Next Steps
- Run the modified script to validate the changes.
- Check logs and outputs to confirm expected behavior.
- Implement dynamic argument parsing if needed.