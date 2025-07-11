use burn::prelude::*;
use burn_ndarray::{NdArray, NdArrayDevice};

// Same model definition - must be duplicated because Rust doesn't share
// code between binaries without making a library crate
#[derive(Module, Debug)]
pub struct GasModel<B: Backend> {
    fc1: nn::Linear<B>,
    fc2: nn::Linear<B>,
    fc3: nn::Linear<B>,
}

impl<B: Backend> GasModel<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.fc1.forward(x).relu();
        let x = self.fc2.forward(x).relu();
        self.fc3.forward(x)
    }
}

fn main() {
    type B = NdArray;
    let device = NdArrayDevice::default();
    
    // Load the saved model from disk
    // :: is used to call associated function (like static method)
    // load_file returns Result type - either Ok(model) or Err(error)
    // .unwrap() extracts the model or panics if error
    let model = GasModel::<B>::load_file("model.bin", &device).unwrap();
    
    // Create input tensor with our 3 features
    // Array literal [55.0, 14.0, 0.7] creates fixed-size array
    let input = Tensor::<B, 2>::from_floats([55.0, 14.0, 0.7], &device).reshape([1, 3]);
    //                                       ^prev  ^hour ^bids
    // reshape([1, 3]) means 1 row (single prediction), 3 columns
    
    // Run inference - no mut needed since model weights don't change
    let prediction = model.forward(input);
    
    // Extract single value and print
    // {:.2} formats float to 2 decimal places
    println!("Predicted gas price: {:.2}", prediction.into_scalar());
}