use std::{env, fs::File};

use mnist_reader::mnist_reader::{display_digit, MnistReader};
use network::network::{layer::ActivationFunction, network::Network, train_network::TrainNetwork};
fn train(mnist: &MnistReader, network: &mut TrainNetwork, epoches: usize, batch_size: usize) {
    let mut counter = 0;
    let train_size = mnist.get_train_size();
    let test_size = mnist.get_test_size();
    for epoch in 1..=epoches {
        for ind in 0..train_size {
            network.forward(mnist.get_train_image(ind)).unwrap();
            let mut target = vec![0.0; network.get_output_size()];
            target[mnist.get_train_label(ind)] = 1.0;
            network.backward(target).unwrap();
            counter += 1;
            if counter == batch_size {
                network.apply_gradiant();
                counter = 0;
            }
        }

        let mut succ_rate = 0.0;
        for ind in 0..test_size {
            let res = network.forward(mnist.get_test_image(ind)).unwrap();
            let mut max = -100.0;
            let mut perdiction = 0;
            for dig in 0..10 {
                if max < res[dig] {
                    max = res[dig];
                    perdiction = dig;
                }
            }
            if mnist.get_test_label(ind) == perdiction {
                succ_rate += 1.0;
            }
        }
        println!("Epoch {}: {}%", epoch, 100.0 * succ_rate / test_size as f64);
    }
}
fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 || args[1]!="-f"{
        panic!("Must enter only one argument: -f [filename where weight will saved]");
    }

    println!("Loading datasets...");
    let mnist = MnistReader::load_mnist(
        "/home/dyan/Programming/Rust/MNIST_Handwritten_digits/mnist").unwrap();
    
    println!("Creating the file...");
    let file = File::create(&args[2]).unwrap();

    println!("Training...");
    let mut network = TrainNetwork::new(
        vec![784, 100, 10],
        vec![
            ActivationFunction::Sigmoid,
            ActivationFunction::Sigmoid
        ],
        3.0,
    );

    train(&mnist, &mut network, 5, 10);
    network.save(file).unwrap();
}
// fn main() {
//     println!("Loading datasets...");
//     let mnist = MnistReader::load_mnist("/home/dyan/Programming/Rust/hand_written/mnist").unwrap();
//     let network = Network::from("/home/dyan/Programming/Rust/hand_written/weights8.txt").unwrap();
//     let test_size = mnist.get_test_size();
//     let mut buf = String::new();
//     for ind in 0..test_size {
//         let image = mnist.get_test_image(ind);
//         let res = network.forward(image).unwrap();
//         let mut max = -100.0;
//         let mut perdiction = 0;
//         for dig in 0..10 {
//             if max < res[dig] {
//                 max = res[dig];
//                 perdiction = dig;
//             }
//         }
//         println!("{} {}", perdiction,display_digit(image));
        
//         std::io::stdin().read_line(&mut buf).unwrap();
//     }
// }
