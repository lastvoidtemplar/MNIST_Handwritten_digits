use network::network::network::Network;
use network::network::{layer::ActivationFunction, train_network::TrainNetwork};
use std::io::Error;
use std::{fs::File, io::Read};
// use byteorder::ReadBytesExt;

fn display_digit(digit: Vec<u8>) {
    for row in 0..28 {
        for col in 0..28 {
            let ch = if digit[row * 28 + col] < 50 { '.' } else { '#' };
            print!("{} ", ch);
        }
        println!();
    }
}
fn read_label_and_digit(file: &mut File) -> Result<(u8, Vec<f64>), Error> {
    let mut digit = Vec::<f64>::new();
    let mut buf: [u8; 1] = [0; 1];
    match file.read(&mut buf) {
        Err(err) => return Err(err),
        _ => (),
    };
    let label = buf[0] - b'0';
    match file.read(&mut buf) {
        Err(err) => return Err(err),
        _ => (),
    };

    let mut str = String::new();
    loop {
        match file.read(&mut buf) {
            Err(err) => return Err(err),
            _ => (),
        };
        let c = buf[0] as char;
        if buf[0] == 0 {
            str.pop();
            let num = match str.parse::<f64>() {
                Ok(res) => res,
                Err(err) => return Err(Error::new(std::io::ErrorKind::Other, err)),
            };
            digit.push(num / 255.0);
            println!("end");
            break;
        }
        if c == '\n' {
            str.pop();
            let num = match str.parse::<f64>() {
                Ok(res) => res,
                Err(err) => return Err(Error::new(std::io::ErrorKind::Other, err)),
            };
            digit.push(num / 255.0);
            break;
        }
        if c == ',' {
            digit.push(str.parse::<f64>().unwrap() / 255.0);
            str.clear();
        } else {
            str.push(c);
        }
    }
    Ok((label, digit))
}
fn train(network:&mut TrainNetwork) -> u32{
    let mut file =
        File::open("/home/dyan/Programming/Rust/hand_written/digit-recognizer/train.csv").unwrap();
    let mut counter = 0;
    let mut batch = 1;
    let mut succ = 0;
    const BATCH_SIZE: u8 = 32;
    loop {
        let (label, digit) = match read_label_and_digit(&mut file) {
            Ok(res) => res,
            Err(err) => break,
        };
        let res = network.forward(digit).unwrap();
        
        let mut max = 0.0;
        let mut max_ind = 0;
        for ind in 0..10 {
            if max < res[ind]{
                max_ind = ind;
                max = res[ind];
            }
        }
        if max_ind as u8==label{
            succ+=1;
        }
        let mut target = vec![0.0; 10];
        target[label as usize] = 1.0;
        network.backward(target).unwrap();
        counter += 1;
        if counter == BATCH_SIZE {
            batch += 1;
            network.apply_gradiant();
            counter = 0;
        }
    }
    succ
}

fn read_test(file: &mut File)->Result<Vec<u8>,Error>{
    let mut digit = Vec::<u8>::new();
    let mut buf: [u8; 1] = [0; 1];
    let mut str = String::new();
    loop {
        match file.read(&mut buf) {
            Err(err) => return Err(err),
            _ => (),
        };
        let c = buf[0] as char;
        if buf[0] == 0 {
            str.pop();
            let num = match str.parse::<u8>() {
                Ok(res) => res,
                Err(err) => return Err(Error::new(std::io::ErrorKind::Other, err)),
            };
            digit.push(num);
            println!("end");
            break;
        }
        if c == '\n' {
            str.pop();
            let num = match str.parse::<u8>() {
                Ok(res) => res,
                Err(err) => return Err(Error::new(std::io::ErrorKind::Other, err)),
            };
            digit.push(num);
            break;
        }
        if c == ',' {
            digit.push(str.parse::<u8>().unwrap());
            str.clear();
        } else {
            str.push(c);
        }
    }
    Ok( digit)
}
fn main() {
    let mut network = TrainNetwork::new(
        vec![784, 100, 10],
        vec![
            ActivationFunction::Sigmoid,
            ActivationFunction::Sigmoid
        ],
        0.05,
    );
    for epoch in 1..=10{
        print!("Epoch {}:", epoch);
        let res = train(&mut network);
        println!("{}",res*100/42000);
    }
    network.save("/home/dyan/Programming/Rust/hand_written/weights6.txt").unwrap();
}
// fn main(){
//     let mut file = File::open("/home/dyan/Programming/Rust/hand_written/digit-recognizer/train.csv").unwrap();
//     let network = Network::from("/home/dyan/Programming/Rust/hand_written/weights2.txt").unwrap();
//     let mut buf = String::new();
//     let mut buff: [u8; 1] = [0; 1];
//     let mut target = vec![0.0; 10];
//     target[2] = 1.0;
//     println!("{:?}",target);
//     loop {
//          file.read(&mut buff).unwrap();
//         file.read(&mut buff).unwrap();
//         let digit = match read_test(&mut file) {
//             Ok(res) => res,
//             Err(err) => break,
//         };
//         let input = digit.iter().map(|&x|(x as f64)/255.0).collect();
        
//         let output = network.forward(input).unwrap();
//         let mut max = 0.0;
//         let mut max_ind = 0;
//         for ind in 0..10 {
//             if max < output[ind]{
//                 max_ind = ind;
//                 max = output[ind];
//             }
//         }
//         println!("{}",max_ind);
//         println!("{:?}",output);
//         display_digit(digit);
//         std::io::stdin().read_line(&mut buf).unwrap();
//     }
// }

