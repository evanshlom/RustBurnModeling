# RustBurnModeling
Rust ML Train and Inference with Burn neural network for easy exercise



Gas Price Prediction Model: predicting Ethereum gas price for the next block on Eth blockchain.

## Project Structure
```
.
├── Cargo.toml
├── README.md
└── src/
    ├── main.rs
    └── inference.rs
```

## Running Training

The first time you run training it can take 30-120 seconds, and inference probably 10 seconds

```bash
cargo run --bin train --release
```

This will:
1. Generate synthetic training data
2. Train a simple neural network
3. Save the model to `model.bin`
4. Print training loss

## Running Inference

```bash
cargo run --bin infer --release
```

This will load the trained model and predict gas price for example inputs.

## Reset Cargo (if needed)

```bash
cargo clean
```


## Model Features
- PrevBlockAvg: Previous block average gas price (f32)
- Hour: Hour of the day (u8, 0-23)
- PoolHighBids: Percentage of high bids in pool (f32, 0.0-1.0)

## Dependencies
- burn = "0.15"
- burn-ndarray = "0.15"