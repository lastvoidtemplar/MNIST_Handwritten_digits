pub fn add(left: usize, right: usize) -> usize {
    left + right
}
pub mod mnist_reader;
use crate::mnist_reader::*;
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result =
            MnistReader::load_mnist("/home/dyan/Programming/Rust/hand_written/mnist").unwrap();
        panic!(
            "{} {}",
            display_digit(result.get_test_image(2)),
            result.get_test_label(2)
        );
        assert_eq!(4, 4);
    }
}
