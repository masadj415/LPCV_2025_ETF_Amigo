#!/bin/bash
## Skripta koja pokrece treniranje modela i prati njegov napredak
# namenjeno za koriscenje na ubuntu operativnom sistemu

# 1. Open a terminal window and run 'python lightning_train.py'
gnome-terminal -- bash -c "source .venv/bin/activate && python lightning_train.py; exec bash"

# 2. Run 'tensorboard --log-dir logs/' in the background and open the URL in the browser, without showing the terminal window
gnome-terminal -- bash -c "source .venv/bin/activate && tensorboard --logdir logs/; exec bash"

# 3. Run 'tensorboard --log-dir tb_logs/' in the background and open the URL in the browser, without showing the terminal window
gnome-terminal -- bash -c "source .venv/bin/activate && tensorboard --logdir tb_logs/; exec bash"

# 4. Open another terminal running 'nvtop'
gnome-terminal -- bash -c "nvtop; exec bash"

# 5. Open the System Monitor
gnome-system-monitor & disown
