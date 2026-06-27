# Running two-stage training


## Run

This command trains the Normal VAE, initializes and trains the vulnerability VAE, saves order-perturbed anomalies, and generates filtered anomaly samples.

```powershell
python generate_op_seq\tain\unknown\train_lstmvae.py `
  --two-stage `
  --normal-dataset normal_all `
  --dataset mix_vulnerabilities `
  --normal-epochs 100 `
  --finetune-epochs 80 `
  --batch-size 128 `
  --output-dir result\lstm_ood_vae_two_stage `
  --perturb-mode all `
  --perturb-window-size 16 `
  --save-extreme-anomaly-csv dataset\embedding\generated_contract\extreme_anomaly_perturbed.csv `
  --generate-boundary-csv dataset\embedding\generated_contract\generated_two_stage_ood.csv `
  --num-generated 5000 `
  --min-normal-ood-score 5.0
```

## Output files

```text
result/lstm_ood_vae_two_stage/NormalLSTMVAE_best.pt
result/lstm_ood_vae_two_stage/ConditionalLSTMVAE_mix_vulnerabilities_best.pt
dataset/embedding/generated_contract/extreme_anomaly_perturbed.csv
dataset/embedding/generated_contract/extreme_anomaly_perturbed_metadata.csv
dataset/embedding/generated_contract/generated_two_stage_ood.csv
```
