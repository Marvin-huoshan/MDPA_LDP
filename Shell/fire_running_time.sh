#!/bin/bash
screen -dmS diff_OUE_vsepsilon_fire_AO_0_SP_10 bash -c 'python -m running_time --splits=10 --h_ao=0 --vswhat=epsilon --dataset=fire --protocol=OUE'
screen -dmS diff_OLHU_vsepsilon_fire_AO_0_SP_10 bash -c 'python -m running_time --splits=10 --h_ao=0 --vswhat=epsilon --dataset=fire --protocol=OLH_User'
screen -dmS diff_OLHS_vsepsilon_fire_AO_0_SP_10 bash -c 'python -m running_time --splits=10 --h_ao=0 --vswhat=epsilon --dataset=fire --protocol=OLH_Server'
screen -dmS diff_HSTU_vsepsilon_fire_AO_0_SP_10 bash -c 'python -m running_time --splits=10 --h_ao=0 --vswhat=epsilon --dataset=fire --protocol=HST_User'
screen -dmS diff_HSTS_vsepsilon_fire_AO_0_SP_10 bash -c 'python -m running_time --splits=10 --h_ao=0 --vswhat=epsilon --dataset=fire --protocol=HST_Server'

screen -dmS diff_OUE_beta_fire_AO_0_SP_10 bash -c 'python -m running_time --splits=10 --h_ao=0 --vswhat=beta --dataset=fire --protocol=OUE'
screen -dmS diff_OLHU_beta_fire_AO_0_SP_10 bash -c 'python -m running_time --splits=10 --h_ao=0 --vswhat=beta --dataset=fire --protocol=OLH_User'
screen -dmS diff_OLHS_beta_fire_AO_0_SP_10 bash -c 'python -m running_time --splits=10 --h_ao=0 --vswhat=beta --dataset=fire --protocol=OLH_Server'
screen -dmS diff_HSTU_beta_fire_AO_0_SP_10 bash -c 'python -m running_time --splits=10 --h_ao=0 --vswhat=beta --dataset=fire --protocol=HST_User'
screen -dmS diff_HSTS_beta_fire_AO_0_SP_10 bash -c 'python -m running_time --splits=10 --h_ao=0 --vswhat=beta --dataset=fire --protocol=HST_Server'


screen -dmS diff_OUE_r_fire_AO_0_SP_10 bash -c 'python -m running_time --splits=10 --h_ao=0 --vswhat=r --dataset=fire --protocol=OUE'
screen -dmS diff_OLHU_r_fire_AO_0_SP_10 bash -c 'python -m running_time --splits=10 --h_ao=0 --vswhat=r --dataset=fire --protocol=OLH_User'
screen -dmS diff_OLHS_r_fire_AO_0_SP_10 bash -c 'python -m running_time --splits=10 --h_ao=0 --vswhat=r --dataset=fire --protocol=OLH_Server'
screen -dmS diff_HSTU_r_fire_AO_0_SP_10 bash -c 'python -m running_time --splits=10 --h_ao=0 --vswhat=r --dataset=fire --protocol=HST_User'
screen -dmS diff_HSTS_r_fire_AO_0_SP_10 bash -c 'python -m running_time --splits=10 --h_ao=0 --vswhat=r --dataset=fire --protocol=HST_Server'
