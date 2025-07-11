use burn::prelude::*;
use burn::tensor::backend::Backend;

#[derive(Module, Debug)]
pub struct GasModel<B: Backend> {
    pub fc1: nn::Linear<B>,
    pub fc2: nn::Linear<B>,
    pub fc3: nn::Linear<B>,
    pub relu: nn::Relu,
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