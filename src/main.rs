use burn::prelude::*;
use burn::tensor::backend::Backend;
use burn::backend::{Autodiff, NdArray};
use burn::nn::loss::{MseLoss, Reduction};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::record::{BinFileRecorder, FullPrecisionSettings};
use gas_predictor::GasModel;

fn generate_data<B: Backend>(device: &B::Device) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let mut features = vec![];
    let mut targets = vec![];
    
    for i in 0..96 {
        let hour = (i / 4) as f32;
        let variant = (i % 4) as f32;
        
        let prev_avg = 40.0 + hour * 1.5 + variant * 3.0;
        let high_bids = (0.5 + 0.3 * (hour * std::f32::consts::PI / 12.0).cos()).clamp(0.0, 1.0);
        
        features.extend_from_slice(&[prev_avg, hour, high_bids]);
        
        let hour_effect = 50.0 + 20.0 * (hour * std::f32::consts::PI / 12.0).sin();
        let prev_effect = prev_avg * 0.8;
        let bid_effect = high_bids * high_bids * 40.0;
        
        targets.push(hour_effect + prev_effect + bid_effect);
    }
    
    let x = Tensor::<B, 1>::from_floats(features.as_slice(), device).reshape([96, 3]);
    let y = Tensor::<B, 1>::from_floats(targets.as_slice(), device).reshape([96, 1]);
    (x, y)
}

fn main() {
    type MyBackend = Autodiff<NdArray>;
    let device = Default::default();
    
    let mut model = GasModel::<MyBackend>::new(&device);
    let (x_train, y_train) = generate_data::<MyBackend>(&device);
    
    let mut optim = AdamConfig::new().init();
    
    for epoch in 0..500 {
        let pred = model.forward(x_train.clone());
        let loss = MseLoss::new().forward(pred.clone(), y_train.clone(), Reduction::Mean);
        
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(0.01, model, grads);
        
        if epoch % 100 == 0 {
            println!("Epoch {}: Loss = {:.4}", epoch, loss.into_scalar());
        }
    }
    
    model.save_file("model.bin", &BinFileRecorder::<FullPrecisionSettings>::new()).unwrap();
    println!("Model saved to model.bin");
}