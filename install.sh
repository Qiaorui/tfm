#!/bin/bash

if command -v python3 &>/dev/null; then
    echo "Python 3 is installed"
else
    echo "Python 3 is not installed"
    exit
fi

pip3 install -r requirements.txt

echo "Installation completed"