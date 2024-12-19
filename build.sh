#! /bin/bash
set -e

rm -rf build
mkdir build

pip3 install torch==1.13.1

pushd nemo;
rm -rf build dist
pip3 install  pytest-runner build
python3 -m build --no-isolation --wheel 
cp ./dist/* ../build/;
popd


pushd apex;
rm -rf build dist
python3 -m build --no-isolation --wheel
cp ./dist/* ../build/;
popd


echo ""
echo "Build done"
