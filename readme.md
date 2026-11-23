# Suction pick: Fairino FR10 + Orbbec Gemini 2

Before you start: robot on at `192.168.58.2` (laptop Ethernet at `192.168.58.24/24`), Gemini 2 on a USB 3 port, Ubuntu 22.04.

## 1. Install (once)

```bash
sudo apt install python3.10-venv
cd FoundationPose
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-pick.txt
pip install --no-deps pyorbbecsdk2==2.1.2

# camera USB permission (then unplug and replug the camera)
sudo bash "$(python -c "import importlib.util, os; print(os.path.dirname(importlib.util.find_spec('pyorbbecsdk').origin))")/shared/install_udev_rules.sh"

# only for the RViz simulation
sudo apt install ros-humble-desktop
```

## 2. Run (every time)

```bash
cd FoundationPose
source .venv/bin/activate
cd suction_pick

python run.py bin              # detect the table / bin (robot does not move)
python run.py pick             # dry run: plans the pick, nothing moves
python run.py sim              # watch the planned pick in RViz
python run.py pick --execute   # pick one parcel (type y to confirm)
```
