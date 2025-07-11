use burn::prelude::*;
use burn::backend::NdArray;
// Need Recorder trait to use the .load() method
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
// Import model from our library - no duplication needed!
use gas_predictor::GasModel;

fn main() {
    // For inference, we don't need Autodiff - just NdArray
    type B = NdArray;
    let device = Default::default();
    
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