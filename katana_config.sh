#!/usr/bin/env bash
# Katana Cluster Connection Configuration
# Source this file to set up connection variables

# SSH Connection Info
export KATANA_HOST="katana1.restech.unsw.edu.au"
export KATANA_USER="z5536858"
export KATANA_PASS="Qzt20010803!!!"
export KATANA_WORK_DIR="/srv/scratch/z5536858/PAE_katana"

# SSH connect function
katana_ssh() {
    expect << EOF
set timeout 300
spawn ssh -o StrictHostKeyChecking=no ${KATANA_USER}@${KATANA_HOST}
expect "password:"
send "${KATANA_PASS}\r"
expect "$ "
send "$1\r"
expect "$ "
send "exit\r"
expect eof
EOF
}

# Interactive SSH
katana_login() {
    expect << EOF
set timeout -1
spawn ssh -o StrictHostKeyChecking=no ${KATANA_USER}@${KATANA_HOST}
expect "password:"
send "${KATANA_PASS}\r"
interact
EOF
}

# SCP transfer function
katana_scp() {
    local src="$1"
    local dst="$2"
    expect << EOF
set timeout 3600
spawn scp -o StrictHostKeyChecking=no "$src" ${KATANA_USER}@${KATANA_HOST}:${KATANA_WORK_DIR}/$dst
expect "password:"
send "${KATANA_PASS}\r"
expect eof
EOF
}

# PBS Job Commands
alias katana_qstat="katana_ssh 'qstat -u ${KATANA_USER}'"
alias katana_qsub_gpu="katana_ssh 'qsub -I -l select=1:ncpus=4:mem=32gb:ngpus=1:gpu_model=A100 -l walltime=20:00:00'"

echo "Katana config loaded. Available commands:"
echo "  katana_login    - Interactive SSH login"
echo "  katana_ssh 'cmd' - Run command on Katana"
echo "  katana_scp src dst - Copy file to Katana"
echo "  katana_qstat    - Check job status"
