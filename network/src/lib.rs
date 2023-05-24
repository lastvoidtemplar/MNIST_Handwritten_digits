pub fn add(left: usize, right: usize) -> usize {
    left + right
}
mod layer;


#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer::*;
    #[test]
    fn it_works() {
        
        let mut layer:Layer = Layer::new(2, 3, ActivationFunction::Sigmoid);
        let output = layer.forward(vec![1.0,2.0,1.0]);
        layer.backwards(vec![1.0,2.0,1.0]);
        layer.backwards(vec![0.5,0.0,0.25]);
        layer.apply_gradiant(1.0);
        assert_eq!(4, 4);
    }
}
