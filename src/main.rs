use burn::prelude::*;
use burn::tensor::backend::Backend;
use burn_ndarray::{NdArray, NdArrayDevice};

#[derive(Module, Debug)]
pub struct GasModel<B: Backend> {
    fc1: nn::Linear<B>,
    fc2: nn::Linear<B>,
    fc3: nn::Linear<B>,
}

impl<B: Backend> GasModel<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            fc1: nn::LinearConfig::new(3, 16).init(device),
            fc2: nn::LinearConfig::new(16, 8).init(device),
            fc3: nn::LinearConfig::new(8, 1).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.fc1.forward(x).relu();
        let x = self.fc2.forward(x).relu();
        self.fc3.forward(x)
    }
}

fn generate_data<B: Backend>(device: &B::Device) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let mut features = vec![];
    let mut targets = vec![];
    
    for i in 0..96 {
        let hour = (i / 4) as f32;  // 0,0,0,0,1,1,1,1...23,23,23,23
        let variant = (i % 4) as f32;  // 0,1,2,3,0,1,2,3...
        
        // Previous block avg: linear trend with variation
        let prev_avg = 40.0 + hour * 1.5 + variant * 3.0;
        
        // High bids: peaks at midnight and noon
        let high_bids = (0.5 + 0.3 * (hour * std::f32::consts::PI / 12.0).cos()).clamp(0.0, 1.0);
        
        features.extend_from_slice(&[prev_avg, hour, high_bids]);
        
        // Target: sin wave for hour, linear for prev_avg, quadratic for high_bids
        let hour_effect = 50.0 + 20.0 * (hour * std::f32::consts::PI / 12.0).sin();
        let prev_effect = prev_avg * 0.8;
        let bid_effect = high_bids * high_bids * 40.0;
        
        targets.push(hour_effect + prev_effect + bid_effect);
    }
    
    let x = Tensor::<B, 2>::from_floats(features.as_slice(), device).reshape([96, 3]);
    let y = Tensor::<B, 2>::from_floats(targets.as_slice(), device).reshape([96, 1]);
    (x, y)
}

fn main() {
    type B = NdArray;
    let device = NdArrayDevice::default();
    
    let mut model = GasModel::<B>::new(&device);
    let (x_train, y_train) = generate_data::<B>(&device);
    
    let mut optim = nn::AdamConfig::new().init();
    
    for epoch in 0..500 {
        let pred = model.forward(x_train.clone());
        let loss = nn::loss::MseLoss::new().forward(pred.clone(), y_train.clone(), nn::loss::Reduction::Mean);
        
        let grads = loss.backward();
        let grads = GradientParams::from_grads(grads, &model);
        model = optim.step(0.01, model, grads);
        
        if epoch % 100 == 0 {
            println!("Epoch {}: Loss = {:.4}", epoch, loss.into_scalar());
        }
    }
    
    model.save_file("model.bin", &Default::default()).unwrap();
    println!("Model saved to model.bin");
}