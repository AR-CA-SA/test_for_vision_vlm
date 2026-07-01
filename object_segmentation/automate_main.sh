#!/bin/bash

read -p "Enter video source " video_source

python3 frame_extraction.py --pathIn "$video_source" --pathOut frames