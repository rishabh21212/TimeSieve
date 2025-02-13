# USE PYTHON 3.10

A few next steps - 
1. You should also update the vali() and test() methods similarly to handle the three return values.

Fix in the train piece - 

# encoder - decoder
if self.args.use_amp:
    with torch.cuda.amp.autocast():
        if self.args.output_attention:
            outputs, loss_IB, final_output = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            outputs, loss_IB, final_output = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        f_dim = -1 if self.args.features == 'MS' else 0
        # outputs is already sliced for pred_len in the model's forward method
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
        loss = criterion(outputs, batch_y) + loss_IB
        train_loss.append(loss.item())
else:
    if self.args.output_attention:
        outputs, loss_IB, final_output = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    else:
        outputs, loss_IB, final_output = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

    f_dim = -1 if self.args.features == 'MS' else 0
    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
    loss = criterion(outputs, batch_y) + loss_IB
    train_loss.append(loss.item())


Similarly, do for vali() and test() - orelse it might throw TypeError: tuple indices must be integers or slices, not tuple



2. Verify full functionality after training

Command - 

python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ --data_path weather.csv --model_id weather_96_96 --model TimeSieve --data custom --features M --seq_len 96 --label_len 48 --pred_len 48 --e_layers 2 --d_layers 1 --factor 3 --enc_in
 21 --dec_in 21 --c_out 21 --d_model 48 --d_ff 96 --top_k 5 --des 'Exp' --itr 1

Please modify the command for better use. This is an example that works.