#!/bin/bash
cd /home/cade/COMPASS
source venv/bin/activate
nohup python compass.py >> /home/cade/COMPASS/compass.log 2>&1 &
echo $! > /home/cade/COMPASS/compass.pid
