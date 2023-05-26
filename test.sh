#!/bin/bash

start_time=`date +%s`

python whisper/transcribe.py ocean.mp3 --model small --beam_size 5

end_time=`date +%s`

run_time=$((end_time - start_time))

echo $run_time