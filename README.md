# RustBurnModeling
Rust ML Train and Inference with Burn neural network for easy exercise

Gas Price Prediction Model: predicting Ethereum gas price for the next block on Eth blockchain.

## Project Structure
```
.
├── Cargo.toml
├── README.md
└── src/
    ├── lib.rs
    ├── main.rs
    └── inference.rs
```

## Running Training

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

**Note**: If you encounter build errors, run `cargo clean` first to reset the build cache.

## Model Features
- PrevBlockAvg: Previous block average gas price (f32)
- Hour: Hour of the day (u8, 0-23)
- PoolHighBids: Percentage of high bids in pool (f32, 0.0-1.0)

## Dependencies
- burn = "0.17" (with ndarray and autodiff features)