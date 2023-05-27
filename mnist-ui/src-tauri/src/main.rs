// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use network::network::network::Network;
// Learn more about Tauri commands at https://tauri.app/v1/guides/features/command
#[tauri::command]
fn preidict(image:Vec<f64>) -> usize {
    let network = Network::from("D:\\Programming\\Rust\\MNIST_Handwritten_digits\\weights1.txt").unwrap();
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

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![preidict])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
