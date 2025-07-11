use burn::prelude::*;
use burn_ndarray::{NdArray, NdArrayDevice};

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
    
    let model = GasModel::<B>::load_file("model.bin", &device).unwrap();
    
    // Example: prev_avg=55.0, hour=14, high_bids=0.7
    let input = Tensor::<B, 2>::from_floats([55.0, 14.0, 0.7], &device).reshape([1, 3]);
    let prediction = model.forward(input);
    
    println!("Predicted gas price: {:.2}", prediction.into_scalar());
}