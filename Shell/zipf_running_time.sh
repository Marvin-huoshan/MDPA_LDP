#!/bin/bash
screen -dmS diff_OUE_vsepsilon_zipf_AO_0_SP_10 bash -c 'python -m running_time --splits=10 --h_ao=0 --vswhat=epsilon --dataset=zipf --protocol=OUE'
screen -dmS diff_OLHU_vsepsilon_zipf_AO_0_SP_10 bash -c 'python -m running_time --splits=10 --h_ao=0 --vswhat=epsilon --dataset=zipf --protocol=OLH_User'
screen -dmS diff_OLHS_vsepsilon_zipf_AO_0_SP_10 bash -c 'python -m running_time --splits=10 --h_ao=0 --vswhat=epsilon --dataset=zipf --protocol=OLH_Server'
screen -dmS diff_HSTU_vsepsilon_zipf_AO_0_SP_10 bash -c 'python -m running_time --splits=10 --h_ao=0 --vswhat=epsilon --dataset=zipf --protocol=HST_User'
screen -dmS diff_HSTS_vsepsilon_zipf_AO_0_SP_10 bash -c 'python -m running_time --splits=10 --h_ao=0 --vswhat=epsilon --dataset=zipf --protocol=HST_Server'

screen -dmS diff_OUE_beta_zipf_AO_0_SP_10 bash -c 'python -m running_time --splits=10 --h_ao=0 --vswhat=beta --dataset=zipf --protocol=OUE'
screen -dmS diff_OLHU_beta_zipf_AO_0_SP_10 bash -c 'python -m running_time --splits=10 --h_ao=0 --vswhat=beta --dataset=zipf --protocol=OLH_User'
screen -dmS diff_OLHS_beta_zipf_AO_0_SP_10 bash -c 'python -m running_time --splits=10 --h_ao=0 --vswhat=beta --dataset=zipf --protocol=OLH_Server'
screen -dmS diff_HSTU_beta_zipf_AO_0_SP_10 bash -c 'python -m running_time --splits=10 --h_ao=0 --vswhat=beta --dataset=zipf --protocol=HST_User'
screen -dmS diff_HSTS_beta_zipf_AO_0_SP_10 bash -c 'python -m running_time --splits=10 --h_ao=0 --vswhat=beta --dataset=zipf --protocol=HST_Server'


screen -dmS diff_OUE_r_zipf_AO_0_SP_10 bash -c 'python -m running_time --splits=10 --h_ao=0 --vswhat=r --dataset=zipf --protocol=OUE'
screen -dmS diff_OLHU_r_zipf_AO_0_SP_10 bash -c 'python -m running_time --splits=10 --h_ao=0 --vswhat=r --dataset=zipf --protocol=OLH_User'
screen -dmS diff_OLHS_r_zipf_AO_0_SP_10 bash -c 'python -m running_time --splits=10 --h_ao=0 --vswhat=r --dataset=zipf --protocol=OLH_Server'
screen -dmS diff_HSTU_r_zipf_AO_0_SP_10 bash -c 'python -m running_time --splits=10 --h_ao=0 --vswhat=r --dataset=zipf --protocol=HST_User'
screen -dmS diff_HSTS_r_zipf_AO_0_SP_10 bash -c 'python -m running_time --splits=10 --h_ao=0 --vswhat=r --dataset=zipf --protocol=HST_Server'
