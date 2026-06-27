#!/bin/bash

read -p "Enter video source " video_source

python3 main_frame.py --pathIn "$video_source" --pathOut frames