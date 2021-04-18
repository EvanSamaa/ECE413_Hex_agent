#!/bin/bash

echo '** (1/4) Installing rust dependencies...'

apt update
apt -yq install build-essential

# Install rustup
if ! command -v rustup &> /dev/null
then
        curl https://sh.rustup.rs -sSf | sh -s -- -y
        source $HOME/.cargo/env
fi

# Set nightly
rustup set profile minimal
rustup default nightly

echo '** (2/4) Building rust project...'

cd mcts
cargo build --release
cd ..

echo '** (3/4) Install python dependencies...'

apt -yq install python3 python3-pip

cd py

pip3 install -r requirements.txt

# The default symlink in the repo is for mac, this replaces with the linux version
rm -f mcts_py.so
ln -s ../mcts/target/release/libmcts_py.so mcts_py.so

cd ..

