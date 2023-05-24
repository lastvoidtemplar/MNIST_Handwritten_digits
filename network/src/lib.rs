pub fn add(left: usize, right: usize) -> usize {
    left + right
}
mod network;



#[cfg(test)]
mod tests {
    use std::fs::File;

    use crate::network::layer::{train_layer::*, ActivationFunction, layer::Layer};
    
    #[test]
    fn save() {
        let mut layer:TrainLayer = TrainLayer::new(2, 3, ActivationFunction::Sigmoid);
        let res = layer.forward(vec![1.0,2.0,1.0]).unwrap();
        let mut file = File::create("test").unwrap();
        layer.save_layer(&mut file).unwrap();
        panic!("{:?}",res);
        assert_eq!(4, 4);
    }
    #[test]
    fn read() {
        let mut file = File::open("test").unwrap();
        let mut layer:Layer = Layer::from(&mut file);
        let res = layer.forward(vec![1.0,2.0,1.0]).unwrap();
        panic!("{:?}",res);
        assert_eq!(4, 4);
    }
}
