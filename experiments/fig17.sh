#/bin/bash

set -e

DURATION=600

python experiments/run_all_chatbot_astra.py --duration $DURATION
python experiments/run_all_chatbot_orca.py --duration $DURATION --len-estimator oracle
python experiments/run_all_chatbot_orca.py --duration $DURATION --len-estimator power2
python experiments/run_all_chatbot_orca.py --duration $DURATION --len-estimator constant
