use burn::prelude::*;
use burn::backend::NdArray;
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};

#[derive(Module, Debug)]
pub struct GasModel<B: Backend> {
    fc1: nn::Linear<B>,
    fc2: nn::Linear<B>,
    fc3: nn::Linear<B>,
    relu: nn::Relu,
}

impl<B: Backend> GasModel<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            fc1: nn::LinearConfig::new(3, 16).init(device),
            fc2: nn::LinearConfig::new(16, 8).init(device),
            fc3: nn::LinearConfig::new(8, 1).init(device),
            relu: nn::Relu::new(),
        }
    }
    
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.fc1.forward(x);
        let x = self.relu.forward(x);
        let x = self.fc2.forward(x);
        let x = self.relu.forward(x);
        self.fc3.forward(x)
    }
}

fn main() {
    type B = NdArray;
    let device = Default::default();
    
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    let record = recorder.load("model.bin".into(), &device).unwrap();
    let model = GasModel::<B>::new(&device).load_record(record);
    
    let input = Tensor::<B, 1>::from_floats([55.0, 14.0, 0.7], &device).reshape([1, 3]);
    let prediction = model.forward(input);
    
    println!("Predicted gas price: {:.2}", prediction.into_scalar());
}