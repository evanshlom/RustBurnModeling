// 'use' brings items into scope - like Python's 'import'
// The '::' is like a path separator (think folder/subfolder)
// '*' means "import everything from this module"
use burn::prelude::*;
// We need Backend trait to make our model work on different hardware (CPU/GPU)
use burn::tensor::backend::Backend;
// Autodiff wraps NdArray to enable automatic differentiation (backprop)
// NdArray is the CPU backend - "Nd" means N-dimensional, like numpy arrays
use burn::backend::{Autodiff, NdArray};
// Import loss function and optimizer components
use burn::nn::loss::{MseLoss, Reduction};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::record::{BinFileRecorder, FullPrecisionSettings};
// Import our model from lib.rs - gas_predictor is the package name from Cargo.toml
use gas_predictor::GasModel;

// Function to generate synthetic training data
// No 'pub' means it's private to this file
fn generate_data<B: Backend>(device: &B::Device) -> (Tensor<B, 2>, Tensor<B, 2>) {
    // 'mut' means mutable - we can modify these vectors
    // vec![] creates an empty vector (dynamic array)
    let mut features = vec![];
    let mut targets = vec![];
    
    // Loop from 0 to 95 (96 total iterations)
    for i in 0..96 {
        // Integer division: 0/4=0, 1/4=0, ..., 4/4=1
        let hour = (i / 4) as f32;  // 'as f32' converts to 32-bit float
        // Modulo gives remainder: 0%4=0, 1%4=1, 2%4=2, 3%4=3, 4%4=0...
        let variant = (i % 4) as f32;
        
        // Generate features with patterns
        let prev_avg = 40.0 + hour * 1.5 + variant * 3.0;
        
        // Cosine wave for high bids (peaks at midnight & noon)
        // std::f32::consts::PI is π constant
        let high_bids = (0.5 + 0.3 * (hour * std::f32::consts::PI / 12.0).cos()).clamp(0.0, 1.0);
        
        // extend_from_slice adds multiple items to vector at once
        // '&' creates a reference to the array
        features.extend_from_slice(&[prev_avg, hour, high_bids]);
        
        // Calculate target with different relationships
        let hour_effect = 50.0 + 20.0 * (hour * std::f32::consts::PI / 12.0).sin();
        let prev_effect = prev_avg * 0.8;
        let bid_effect = high_bids * high_bids * 40.0;  // Quadratic
        
        targets.push(hour_effect + prev_effect + bid_effect);
    }
    
    // Convert vectors to tensors and reshape
    // from_floats takes a slice (view of array)
    // reshape([96, 3]) means 96 rows, 3 columns
    let x = Tensor::<B, 2>::from_floats(features.as_slice(), device).reshape([96, 3]);
    let y = Tensor::<B, 2>::from_floats(targets.as_slice(), device).reshape([96, 1]);
    // Return tuple of (features, targets)
    (x, y)
}

// Main function - program entry point
fn main() {
    // Type alias for convenience - Autodiff wraps NdArray to enable backprop
    type MyBackend = Autodiff<NdArray>;
    // Default device is CPU
    let device = Default::default();
    
    // Create model - 'mut' because we'll update weights
    let mut model = GasModel::<MyBackend>::new(&device);
    // Generate training data
    let (x_train, y_train) = generate_data::<MyBackend>(&device);
    
    // Create Adam optimizer (adaptive learning rate algorithm)
    let mut optim = AdamConfig::new().init();
    
    // Training loop - 500 iterations
    for epoch in 0..500 {
        // Forward pass - get predictions
        // .clone() makes a copy (required by Burn's design)
        let pred = model.forward(x_train.clone());
        // Calculate Mean Squared Error loss
        let loss = MseLoss::new().forward(pred.clone(), y_train.clone(), Reduction::Mean);
        
        // Backward pass - calculate gradients
        let grads = loss.backward();
        // Extract gradients for model parameters
        let grads = GradientsParams::from_grads(grads, &model);
        // Update model weights (0.01 is learning rate)
        model = optim.step(0.01, model, grads);
        
        // Print progress every 100 epochs
        if epoch % 100 == 0 {
            // into_scalar() converts 1x1 tensor to plain number
            println!("Epoch {}: Loss = {:.4}", epoch, loss.into_scalar());
        }
    }
    
    // Save trained model to disk
    // &Default::default() uses default save options
    // .unwrap() crashes if save fails (ok for simple example)
    model.save_file("model.bin", &Default::default()).unwrap();
    println!("Model saved to model.bin");
}