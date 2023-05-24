pub fn add(left: usize, right: usize) -> usize {
    left + right
}
mod network;

#[cfg(test)]
mod tests {
    use std::fs::File;

    use crate::network::{
        layer::{layer::Layer, train_layer::*, ActivationFunction},
        train_network::TrainNetwork, network::Network,
    };

    #[test]
    fn read() {
        let mut file = File::open("test").unwrap();
        let mut layer: Layer = Layer::from(&mut file).unwrap();
        let res = layer.forward(&vec![1.0, 2.0, 1.0]).unwrap();
        panic!("{:?}", res);
        let mut t = vec![0, 1, 2];
        assert_eq!(4, 4);
    } 
    #[test]
    fn save() {
        let mut layer: TrainLayer = TrainLayer::new(2, 3, ActivationFunction::Sigmoid);
        let res = layer.forward(&vec![1.0, 2.0, 1.0]).unwrap();
        let mut file = File::create("test").unwrap();
        layer.save_layer(&mut file).unwrap();
        panic!("{:?}", res);
        assert_eq!(4, 4);
    }
    #[test]
    fn read_network() {
        let mut network = Network::from("test").unwrap();
        let res = network.forward(vec![1.0, 0.0]).unwrap();
        panic!("{:?}", res);
        assert_eq!(4, 4);
    } 
    #[test]
    fn save_network() {
        let mut newtork = TrainNetwork::new(
            vec![2, 3, 2],
            vec![ActivationFunction::Sigmoid, ActivationFunction::Sigmoid],
            1.0,
        );
        let res = newtork.forward(vec![1.0, 0.0]).unwrap();
        newtork.save("test").unwrap();
        panic!("{:?}", res);
        assert_eq!(4, 4);
    }
}
