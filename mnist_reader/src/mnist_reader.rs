use std::{
    fs::File,
    io::{Error, Read},
    ops::RemAssign,
};

use byteorder::{BigEndian, ReadBytesExt};
const TRAIN_IMAGES_FILE_NAME: &str = "train-images.idx3-ubyte";
const TRAIN_LABELS_FILE_NAME: &str = "train-labels.idx1-ubyte";
const TEST_IMAGES_FILE_NAME: &str = "t10k-images.idx3-ubyte";
const TEST_LABELS_FILE_NAME: &str = "t10k-labels.idx1-ubyte";
pub struct MnistReader {
    train_images: Vec<Vec<f64>>, //flatten images
    train_labels: Vec<usize>,
    test_images: Vec<Vec<f64>>,
    test_labels: Vec<usize>,
    train_size: usize,
    test_size: usize,
}

impl MnistReader {
    pub fn load_mnist(directory_name: &str) -> Result<MnistReader, Error> {
        let (train_images, train_image_size) =
            match load_images(directory_name, TRAIN_IMAGES_FILE_NAME) {
                Ok(res) => res,
                Err(err) => return Err(err),
            };

        let (train_labels, train_label_size) =
            match load_labels(directory_name, TRAIN_LABELS_FILE_NAME) {
                Ok(res) => res,
                Err(err) => return Err(err),
            };
        if train_image_size != train_label_size {
            return Err(Error::new(
                std::io::ErrorKind::InvalidInput,
                "Mismatch between train image size and train label size!",
            ));
        }
        let (test_images, test_image_size) =
            match load_images(directory_name, TEST_IMAGES_FILE_NAME) {
                Ok(res) => res,
                Err(err) => return Err(err),
            };

        let (test_labels, test_label_size) =
            match load_labels(directory_name, TEST_LABELS_FILE_NAME) {
                Ok(res) => res,
                Err(err) => return Err(err),
            };
        if test_image_size != test_label_size {
            return Err(Error::new(
                std::io::ErrorKind::InvalidInput,
                "Mismatch between test image size and train label size!",
            ));
        }
        Ok(MnistReader {
            train_images,
            train_labels,
            test_images,
            test_labels,
            train_size:train_image_size,
            test_size:test_image_size
        })
    }
    pub fn get_train_image(&self, ind: usize) -> &Vec<f64> {
        &self.train_images[ind]
    }
    pub fn get_train_label(&self, ind: usize) -> usize {
        self.train_labels[ind]
    }
    pub  fn get_train_size(&self) -> usize{
        self.train_size
    }
    pub fn get_test_image(&self, ind: usize) -> &Vec<f64> {
        &self.test_images[ind]
    }
    pub fn get_test_label(&self, ind: usize) -> usize {
        self.test_labels[ind]
    }
    pub  fn get_test_size(&self) -> usize{
        self.test_size
    }
}
//util function
pub fn display_digit(digit: &Vec<f64>) -> String {
    let mut str = String::from("\n");
    for row in 0..28 {
        for col in 0..28 {
            let ch = if digit[row * 28 + col] < 50.0 / 255.0 {
                '.'
            } else {
                '#'
            };
            str.push(ch);
            str.push(' ');
        }
        str.push('\n');
    }
    str
}
fn load_images(directory_name: &str, filename: &str) -> Result<(Vec<Vec<f64>>, usize), Error> {
    let mut file = match File::open(directory_name.to_string() + "/" + filename) {
        Ok(res) => res,
        Err(err) => return Err(err),
    };

    //reading magic number
    match file.read_u32::<BigEndian>() {
        Err(err) => return Err(err),
        _ => (),
    }

    let images_count = match file.read_u32::<BigEndian>() {
        Ok(res) => res as usize,
        Err(err) => return Err(err),
    };

    let image_rows = match file.read_u32::<BigEndian>() {
        Ok(res) => res as usize,
        Err(err) => return Err(err),
    };

    let image_cols = match file.read_u32::<BigEndian>() {
        Ok(res) => res as usize,
        Err(err) => return Err(err),
    };

    let mut images = Vec::<Vec<f64>>::new();

    for ind in 0..images_count {
        let image = match read_image(&mut file, image_rows, image_cols) {
            Ok(res) => res,
            Err(err) => return Err(err),
        };
        images.push(image);
    }

    Ok((images, images_count))
}

fn read_image(file: &mut File, image_rows: usize, image_cols: usize) -> Result<Vec<f64>, Error> {
    let mut flatten_image = Vec::<f64>::new();
    for row in 0..image_rows {
        for col in 0..image_cols {
            let pixel = match file.read_u8() {
                Ok(res) => res as f64,
                Err(err) => return Err(err),
            };
            flatten_image.push(pixel / 255.0);
        }
    }
    Ok(flatten_image)
}
fn load_labels(directory_name: &str, filename: &str) -> Result<(Vec<usize>, usize), Error> {
    let mut file = match File::open(directory_name.to_string() + "/" + filename) {
        Ok(res) => res,
        Err(err) => return Err(err),
    };

    //reading magic number
    match file.read_u32::<BigEndian>() {
        Err(err) => return Err(err),
        _ => (),
    }

    let labels_count = match file.read_u32::<BigEndian>() {
        Ok(res) => res as usize,
        Err(err) => return Err(err),
    };
    let mut labels = Vec::<usize>::new();
    for ind in 0..labels_count {
        let label = match file.read_u8() {
            Ok(res) => res as usize,
            Err(err) => return Err(err),
        };
        labels.push(label);
    }
    Ok((labels, labels_count))
}
