multisusie_directory="$(dirname "$(readlink -f "$0")")"
python -m venv ${multisusie_directory}/multisusie_env
${multisusie_directory}/multisusie_env/bin/pip install -r ${multisusie_directory}/requirements.txt
${multisusie_directory}/multisusie_env/bin/pip install ${multisusie_directory} -U
${multisusie_directory}/multisusie_env/bin/python ${multisusie_directory}/examples/example_for_install_script.py ${multisusie_directory}
