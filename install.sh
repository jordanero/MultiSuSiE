if command -v python &>/dev/null; then
    pip install -r MultiSuSiE/requirements.txt
    pip install MultiSuSiE/ -U
    python MultiSuSiE//examples/example_for_install_script.py
elif command -v python3 &>/dev/null; then
    pip3 install -r MultiSuSiE/requirements.txt
    pip3 install MultiSuSiE/ -U
    python3 MultiSuSiE//examples/example_for_install_script.py
else
    echo "Error: Python is not installed." >&2
    exit 1
fi
