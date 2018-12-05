#!/usr/bin/env bash

LOG_TEMP_EVERY_SECOND=30
LOG_TEMP_FILENAME="temp.log"

while [[ True ]]
do
    echo "========================================" >> ${LOG_TEMP_FILENAME}
    echo $(date) >> ${LOG_TEMP_FILENAME}
    echo "========================================" >> ${LOG_TEMP_FILENAME}
    echo "$(nvidia-smi -q -a | grep -E 'Power Draw|Memory Current|GPU Current Temp|Gpu|Used GPU Memory')" >> ${LOG_TEMP_FILENAME}
#    printf "\n" >> ${LOG_TEMP_FILENAME}
#    echo "$(sensors)" >> ${LOG_TEMP_FILENAME}
    printf "\n\n" >> ${LOG_TEMP_FILENAME}
    sleep ${LOG_TEMP_EVERY_SECOND}
done