use burn::prelude::*;
use burn::backend::NdArray;
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
use gas_predictor::GasModel;

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