"""Deploy depth-7 GPU pilot to Atlas via paramiko."""
import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('100.68.134.21', username='claude', password='roZes9090!~')
sftp = c.open_sftp()

# 1. Dataset loader (standalone, no CLI arg parsing)
with sftp.file('/home/claude/tensor-3body/dataset_loader.py', 'w') as f:
    f.write(open(r'C:\source\tensor-3body\dataset_loader.py').read())
print("Uploaded dataset_loader.py")

# 2. Pilot script
with sftp.file('/home/claude/tensor-3body/pilot_depth7.py', 'w') as f:
    f.write(open(r'C:\source\tensor-3body\pilot_depth7.py').read())
print("Uploaded pilot_depth7.py")

sftp.close()

# Kill old, launch
c.exec_command('tmux kill-session -t pilot 2>/dev/null')
import time; time.sleep(1)
c.exec_command('tmux new-session -d -s pilot "cd /home/claude/tensor-3body && /home/claude/env/bin/python pilot_depth7.py 2>&1 | tee /tmp/pilot_d7.log"')
time.sleep(5)

_, o, _ = c.exec_command('tail -15 /tmp/pilot_d7.log 2>/dev/null')
print("Log tail:")
print(o.read().decode().strip())

_, o, _ = c.exec_command('nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader')
print("GPU:", o.read().decode().strip())

_, o, _ = c.exec_command('tmux list-sessions')
print("Sessions:", o.read().decode().strip())

c.close()
