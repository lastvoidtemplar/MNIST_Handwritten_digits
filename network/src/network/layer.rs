use std::{f64::consts::E};
#[derive(Debug, Clone, Copy)]
pub enum ActivationFunction {
    Sigmoid,
    RELU,
    InverseTan,
}
fn apply_activation_function(x: f64, f: ActivationFunction) -> f64 {
    match f {
        ActivationFunction::Sigmoid => 1.0 / (1.0 + E.powf(-x)),
        ActivationFunction::RELU => x.max(0.0),
        ActivationFunction::InverseTan => x.tanh(),
    }
}
fn apply_derivitive_activation_function(x: f64, f: ActivationFunction) -> f64 {
    match f {
        ActivationFunction::Sigmoid => {
            let sigmoid = apply_activation_function(x, ActivationFunction::Sigmoid);
            sigmoid * (1.0 - sigmoid)
        }
        ActivationFunction::RELU => {
            if x > 0.0 {
                1.0
            } else {
                0.0
            }
        }
        ActivationFunction::InverseTan => 1.0 / (1.0 + x * x),
    }
}
pub mod train_layer {
    use crate::network::ActivationFunction;
    use byteorder::{ LittleEndian, WriteBytesExt};
    use rand::prelude::*;
    use simple_matrix::Matrix;
    use std::{ fs::File, usize};

    use super::{apply_activation_function, apply_derivitive_activation_function};

    pub struct TrainLayer {
        input_size: usize,
        output_size: usize,
        weights: Matrix<f64>,
        input: Vec<f64>,
        activation_function: ActivationFunction,
        sum_output: Vec<f64>,
        changes: Matrix<f64>,
        count_iter: u32,
    }
    impl TrainLayer {
        pub fn new(
            input_size: usize,
            output_size: usize,
            activation_function: ActivationFunction,
        ) -> TrainLayer {
            let mut rng = rand::thread_rng();
            let mut input = vec![1.0; input_size + 1];
            let mut weights = Matrix::new(output_size, input_size + 1);
            for ind in 0..input_size {
                input[ind] = rng.gen();
            }
            for row in 0..output_size {
                for col in 0..=input_size {
                    weights.set(row, col, rng.gen());
                }
            }
            TrainLayer {
                input_size,
                output_size,
                weights,
                input,
                activation_function,
                sum_output: vec![0.0; output_size],
                changes: Matrix::new(output_size, input_size + 1),
                count_iter: 0,
            }
        }
        pub fn forward(&mut self, input: Vec<f64>) -> Result<Vec<f64>, &str> {
            if input.len() != self.input_size + 1 {
                return Err("Input size doesn`t match initual input size!");
            }
            let mut output = vec![0.0; self.output_size + 1];
            self.input = input.clone();
            for row in 0..self.output_size {
                for col in 0..=self.input_size {
                    output[row] += self.weights.get(row, col).unwrap() * input[col];
                }
                self.sum_output[row] = output[row];
                output[row] = apply_activation_function(output[row], self.activation_function);
            }
            output[self.output_size] = 1.0; //bias
            Ok(output)
        }
        pub fn backwards(
            &mut self,
            output_partial_derivitive: Vec<f64>,
            generate_derivitives: bool,
        ) -> Result<Vec<f64>, &str> {
            if output_partial_derivitive.len() != self.output_size {
                return Err("Output size doesn`t match initual output size!");
            }
            let mut perv_layer_derivitive = vec![0.0; self.input_size];
            for row in 0..self.output_size {
                let cost_change = output_partial_derivitive[row]
                    * apply_derivitive_activation_function(
                        self.sum_output[row],
                        self.activation_function,
                    );
                //change on weights
                for col in 0..=self.input_size {
                    let weight_change =
                        self.changes.get(row, col).unwrap() + cost_change * self.input[col];
                    self.changes.set(row, col, weight_change);
                }
                if generate_derivitives {
                    //changes for perv layer
                    for col in 0..self.input_size {
                        perv_layer_derivitive[col] +=
                            cost_change * self.changes.get(row, col).unwrap();
                    }
                }
            }
            self.count_iter += 1; //bias
            Ok(perv_layer_derivitive)
        }
        pub fn apply_gradiant(&mut self, learning_rate: f64) {
            for row in 0..self.output_size {
                for col in 0..=self.input_size {
                    let new_weight = self.weights.get(row, col).unwrap()
                        - learning_rate / (self.count_iter as f64)
                            * self.changes.get(row, col).unwrap();
                    self.weights.set(row, col, new_weight);
                }
            }
            self.changes = Matrix::new(self.input_size, self.output_size + 1);
            self.count_iter = 0;
        }
        pub fn save_layer(&self, file: &mut File) -> std::io::Result<()> {
            file.write_u64::<LittleEndian>(self.input_size as u64)
                .unwrap();
            file.write_u64::<LittleEndian>(self.output_size as u64)
                .unwrap();
            file.write_u8(
            match self.activation_function {
                ActivationFunction::Sigmoid=>0,
                ActivationFunction::RELU=>1,
                ActivationFunction::InverseTan=>2
            });

            for row in 0..self.output_size {
                for col in 0..=self.input_size {
                    file.write_f64::<LittleEndian>(self.weights.get(row, col).unwrap() + 0.0)
                        .unwrap();
                }
            }

            Ok(())
        }
    }
}
pub mod layer {
    use std::fs::File;

    use byteorder::{ReadBytesExt, LittleEndian};
    use rand::Rng;
    use simple_matrix::Matrix;

    use super::{ActivationFunction, apply_activation_function};

    pub struct Layer {
        input_size: usize,
        output_size: usize,
        weights: Matrix<f64>,
        activation_function: ActivationFunction,
    }
    impl Layer {
        pub fn new(
            input_size: usize,
            output_size: usize,
            activation_function: ActivationFunction,
        ) -> Layer {
            let mut rng = rand::thread_rng();
            let mut weights = Matrix::new(output_size, input_size + 1);
            for row in 0..output_size {
                for col in 0..=input_size {
                    weights.set(row, col, rng.gen());
                }
            }
            Layer {
                input_size,
                output_size,
                weights,
                activation_function,
            }
        }
        pub fn forward(&mut self, input: Vec<f64>) -> Result<Vec<f64>, &str> {
            if input.len() != self.input_size + 1 {
                return Err("Input size doesn`t match initual input size!");
            }
            let mut output = vec![0.0; self.output_size + 1];
            for row in 0..self.output_size {
                for col in 0..=self.input_size {
                    output[row] += self.weights.get(row, col).unwrap() * input[col];
                }
                output[row] = apply_activation_function(output[row], self.activation_function);
            }
            output[self.output_size] = 1.0; //bias
            Ok(output)
        }
        pub fn from(file:&mut File)->Layer{
            const ERR_MSG:&str = "Problem with reading the file!";
            let input_size = file.read_u64::<LittleEndian>().expect(ERR_MSG) as usize;
            let output_size = file.read_u64::<LittleEndian>().expect(ERR_MSG) as usize;
            let f_byte = file.read_u8().expect(ERR_MSG);
            let activation_function = match f_byte {
                0=>ActivationFunction::Sigmoid,
                1=>ActivationFunction::RELU,
                2=>ActivationFunction::InverseTan,
                _=>panic!("Unknown activation function")
            };
            let mut weights = Matrix::new(output_size,input_size+1);
            for row in 0..output_size {
                for col in 0..=input_size {
                    let value = file.read_f64::<LittleEndian>().expect(ERR_MSG);
                    weights.set(row, col, value);
                }
            }
            Layer { input_size, output_size, weights, activation_function }
        }
    }
}
