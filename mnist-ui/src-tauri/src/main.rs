// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::Mutex;

use mnist_reader::mnist_reader::display_digit;
use network::network::{layer::ActivationFunction, network::Network};
use tauri::State;

pub struct NetworkState(Mutex<Network>);

#[tauri::command]
fn pridict(state:State<NetworkState>,image: Vec<f64>) -> usize {
    let network = state.0.lock().unwrap();
    println!("{}", display_digit(&image));
    let res = network.forward(&image).unwrap();
    let mut max = -100.0;
    let mut perdiction = 0;
    for dig in 0..10 {
        if max < res[dig] {
            max = res[dig];
            perdiction = dig;
        }
    }
    return perdiction;
}
#[tauri::command]
fn read_network(filename:&str,state:State<NetworkState>) {
    let mut network = state.0.lock().unwrap();
    *network = Network::from(filename).unwrap();
}

fn main() {
    tauri::Builder::default()
        .manage(NetworkState(Mutex::from( Network::new(
            vec![784, 10], 
            vec![ActivationFunction::Sigmoid]))))
        .invoke_handler(tauri::generate_handler![pridict,read_network])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
