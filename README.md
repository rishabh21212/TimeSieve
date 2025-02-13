# Use Python 3.10 

## Next Steps:

### 1. Update `vali()` and `test()` Methods( THIS IS TBD)

Ensure that `vali()` and `test()` methods handle three return values similarly to avoid the error:

```
TypeError: tuple indices must be integers or slices, not tuple
```

Update the method calls to unpack the three return values correctly.

### 2. Fix that was implemented in Training Code(THIS IS DONE)

Modify the training process in `train()` to properly handle three return values:

```python
# encoder - decoder
if self.args.use_amp:
    with torch.cuda.amp.autocast():
        outputs, loss_IB, final_output = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        
        f_dim = -1 if self.args.features == 'MS' else 0
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
        loss = criterion(outputs, batch_y) + loss_IB
        train_loss.append(loss.item())
else:
    outputs, loss_IB, final_output = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    
    f_dim = -1 if self.args.features == 'MS' else 0
    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
    loss = criterion(outputs, batch_y) + loss_IB
    train_loss.append(loss.item())
```

### 3. Apply Similar Changes to `vali()` and `test()`(THIS IS TBD - pls do)
Ensure that both `vali()` and `test()` handle three return values as follows:

```python
outputs, loss_IB, final_output = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
```

Modify their respective loss calculations in the same manner.

### 4. Verify Full Functionality After Training(TBD)

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

### 5. Modify the Command for Better Usability

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