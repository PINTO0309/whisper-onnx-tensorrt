#!/bin/bash

start_time=`date +%s`

python test_faster_whisper.py

end_time=`date +%s`

run_time=$((end_time - start_time))

echo $run_time