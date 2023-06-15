# Summary

The project consists of two parts trainer and mnist-ui. Trainer is a console application written in Rust that trains a neural network with the MNIST dataset and tests the neural network's accuracy.
At the end of the execution, it saves the network parameters to a binary file.
Mnist-ui is a desktop application written in React.js and Tauri that allows the user to load the neural network generated by Trainer and test it with hand drawn numbers using a canvas.

## Compile and run

To compile the projects you will need yarn and cargo.

Trainer:

```bash
cd trainer
cargo build --release
#for linux
./target/release/trainer -f [weights file location] -d [mnist folder location]
#Windows
.\target\release\trainer.exe -f [weights file location] -d [mnist folder location]
```
weights file location - where it saves the network parameters in the end of the program.\
mnist folder location - where the mnist dataset folder is located 

Mnist UI

```bash
cd mnist-ui
yarn
yarn tauri build
#for linux
./src-tauri/target/release/mnist-ui
#Windows
.\src-tauri\target\release\mnist-ui.exe
```
In weights.txt is example file for mnist-ui. It has 97.00% accuracy.

In mnist-ui/src-tauri/target/release/bundle you can find installer for the destop app for your operating system after succsessful compilation.