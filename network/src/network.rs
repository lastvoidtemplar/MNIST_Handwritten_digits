pub mod layer;
use layer::*;
pub mod train_network {

    use std::fs::File;

    use byteorder::{WriteBytesExt, LittleEndian};
    use simple_matrix::Matrix;

    use super::layer::{train_layer::TrainLayer, ActivationFunction};
    use std::io::Error;
    #[derive(Debug)]
    pub struct TrainNetwork {
        layers: Vec<TrainLayer>,
        learning_rate: f64,
        input_size: usize,
        output_size: usize,
        output: Vec<f64>,
    }

    impl TrainNetwork {
        pub fn new(
            layer_sizes: Vec<usize>,
            activation_functions: Vec<ActivationFunction>,
            learning_rate: f64,
        ) -> TrainNetwork {
            if layer_sizes.len() < 2 {
                panic!("Too small vec layer_sizes");
            }
            if layer_sizes.len() != activation_functions.len() + 1 {
                panic!("Mismatch between sizes and activation functions");
            }

            let mut layers = Vec::<TrainLayer>::new();
            for ind in 1..layer_sizes.len() {
                layers.push(TrainLayer::new(
                    layer_sizes[ind - 1],
                    layer_sizes[ind],
                    activation_functions[ind - 1],
                ))
            }
            TrainNetwork {
                layers,
                learning_rate,
                input_size: layer_sizes[0],
                output_size: layer_sizes.last().unwrap() + 0,
                output: vec![0.0; layer_sizes.last().unwrap() + 0],
            }
        }
        pub fn forward(&mut self, inp: &Vec<f64>) -> Result<Vec<f64>, &str> {
            if inp.len() != self.input_size {
                return Err("Invalid size");
            }
            let mut input = inp.clone();
            input.push(1.0);
            for layer in self.layers.iter_mut() {
                input = match layer.forward(input) {
                    Ok(res) => res,
                    Err(msg) => return Err(msg),
                }
            }
            input.pop();
            self.output = input.clone();
            Ok(input)
        }
        pub fn backward(&mut self, mut target: Vec<f64>) -> Result<(), &str> {
            if target.len() != self.output_size {
                return Err("Invalid size");
            }
            for ind in 0..self.output_size {
                target[ind] = -target[ind] + self.output[ind];
            }
            for (ind, layer) in self.layers.iter_mut().enumerate().rev() {
                target = match layer.backwards(&target, ind != 0) {
                    Ok(res) => res,
                    Err(msg) => return Err(msg),
                }
            }
            Ok(())
        }
        pub fn apply_gradiant(&mut self) {
            for layer in self.layers.iter_mut(){
                layer.apply_gradiant(self.learning_rate);
            }
        }
        pub fn save(&self,mut file:File) -> Result<(), Error> {
            match file.write_u64::<LittleEndian>(self.layers.len() as u64){
                Err(err) => return Err(err),
                _ => (),
            }
            for layer in self.layers.iter() {
                match layer.save_layer(&mut file) {
                    Err(err) => return Err(err),
                    _ => (),
                };
            }

            Ok(())
        }
        pub fn get_input_size(&self) ->usize{
            self.input_size
        }
        pub fn get_output_size(&self) ->usize{
            self.output_size
        }
    }
}
pub mod network {
    use byteorder::{LittleEndian, ReadBytesExt};

    use super::layer::{layer::Layer, ActivationFunction};
    use std::{io::Error, fs::File};
    #[derive(Debug)]
    pub struct Network {
        layers: Vec<Layer>,
        input_size: usize,
        output_size: usize,
    }

    impl Network {
        pub fn new(
            layer_sizes: Vec<usize>,
            activation_functions: Vec<ActivationFunction>,
        ) -> Network {
            if layer_sizes.len() < 2 {
                panic!("Too small vec layer_sizes");
            }
            if layer_sizes.len() != activation_functions.len() + 1 {
                panic!("Mismatch between sizes and activation functions");
            }

            let mut layers = Vec::<Layer>::new();
            for ind in 1..layer_sizes.len() {
                layers.push(Layer::new(
                    layer_sizes[ind - 1],
                    layer_sizes[ind],
                    activation_functions[ind - 1],
                ))
            }
            Network {
                layers,
                input_size: layer_sizes[0],
                output_size: layer_sizes.last().unwrap() + 0,
            }
        }
        pub fn forward(&self,inp: &Vec<f64>) -> Result<Vec<f64>, &str> {
            if inp.len() != self.input_size {
                return Err("Invalid size");
            }
            let mut input = inp.clone();
            input.push(1.0);
            for layer in self.layers.iter() {
                input = match layer.forward(&input) {
                    Ok(res) => res,
                    Err(msg) => return Err(msg),
                }
            }
            input.pop();
            Ok(input)
        }
        pub fn from(filename: &str)->Result<Network,Error>{
            let mut file = match File::open(filename){
                Ok(res)=> res,
                Err(err)=> return Err(err)
            };
            let layers_count = match file.read_u64::<LittleEndian>() {
                Ok(res)=> res as usize,
                Err(err)=> return Err(err)
            };
            let mut layers = Vec::<Layer>::new();
            for ind in 0..layers_count {
                let layer=match Layer::from(&mut file){
                    Ok(res)=> res,
                    Err(err)=> return Err(err)
                };
                
                layers.push(layer);
            }
            Ok(Network { 
                input_size: layers[0].get_input_size(), 
                output_size:layers[layers_count-1] .get_output_size(), 
                layers
            })
        }
        pub fn get_input_size(&self) ->usize{
            self.input_size
        }
        pub fn get_output_size(&self) ->usize{
            self.output_size
        }
    }
}
