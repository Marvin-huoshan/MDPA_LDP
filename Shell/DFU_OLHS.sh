#!/bin/bash
screen -dmS diff_OLHS_vsepsilon_zipf_AO_0_SP_10 bash -c 'python -m Fake_User_Detection --splits=10 --h_ao=0 --vswhat=epsilon --dataset=zipf --protocol=OLH_Server'
screen -dmS diff_OLHS_vsepsilon_zipf_AO_0_SP_8 bash -c 'python -m Fake_User_Detection --splits=8  --h_ao=0 --vswhat=epsilon_1 --dataset=zipf --protocol=OLH_Server'
screen -dmS diff_OLHS_vsepsilon_zipf_AO_0_SP_6 bash -c 'python -m Fake_User_Detection --splits=6 --h_ao=0 --vswhat=epsilon_1 --dataset=zipf --protocol=OLH_Server'
screen -dmS diff_OLHS_vsepsilon_zipf_AO_0_SP_4 bash -c 'python -m Fake_User_Detection --splits=4 --h_ao=0 --vswhat=epsilon --dataset=zipf --protocol=OLH_Server'
screen -dmS diff_OLHS_vsepsilon_zipf_AO_0_SP_2 bash -c 'python -m Fake_User_Detection --splits=2 --h_ao=0 --vswhat=epsilon_1 --dataset=zipf --protocol=OLH_Server'
