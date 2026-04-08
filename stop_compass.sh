#!/bin/bash
if [ -f /home/cade/COMPASS/compass.pid ]; then
    kill $(cat /home/cade/COMPASS/compass.pid)
    rm /home/cade/COMPASS/compass.pid
fi
