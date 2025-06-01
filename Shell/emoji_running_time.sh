#!/bin/bash
screen -dmS diff_OUE_vsepsilon_emoji_AO_0_SP_10 bash -c 'python -m running_time --splits=10 --h_ao=0 --vswhat=epsilon --dataset=emoji --protocol=OUE'
screen -dmS diff_OLHU_vsepsilon_emoji_AO_0_SP_10 bash -c 'python -m running_time --splits=10 --h_ao=0 --vswhat=epsilon --dataset=emoji --protocol=OLH_User'
screen -dmS diff_OLHS_vsepsilon_emoji_AO_0_SP_10 bash -c 'python -m running_time --splits=10 --h_ao=0 --vswhat=epsilon --dataset=emoji --protocol=OLH_Server'
screen -dmS diff_HSTU_vsepsilon_emoji_AO_0_SP_10 bash -c 'python -m running_time --splits=10 --h_ao=0 --vswhat=epsilon --dataset=emoji --protocol=HST_User'
screen -dmS diff_HSTS_vsepsilon_emoji_AO_0_SP_10 bash -c 'python -m running_time --splits=10 --h_ao=0 --vswhat=epsilon --dataset=emoji --protocol=HST_Server'

screen -dmS diff_OUE_beta_emoji_AO_0_SP_10 bash -c 'python -m running_time --splits=10 --h_ao=0 --vswhat=beta --dataset=emoji --protocol=OUE'
screen -dmS diff_OLHU_beta_emoji_AO_0_SP_10 bash -c 'python -m running_time --splits=10 --h_ao=0 --vswhat=beta --dataset=emoji --protocol=OLH_User'
screen -dmS diff_OLHS_beta_emoji_AO_0_SP_10 bash -c 'python -m running_time --splits=10 --h_ao=0 --vswhat=beta --dataset=emoji --protocol=OLH_Server'
screen -dmS diff_HSTU_beta_emoji_AO_0_SP_10 bash -c 'python -m running_time --splits=10 --h_ao=0 --vswhat=beta --dataset=emoji --protocol=HST_User'
screen -dmS diff_HSTS_beta_emoji_AO_0_SP_10 bash -c 'python -m running_time --splits=10 --h_ao=0 --vswhat=beta --dataset=emoji --protocol=HST_Server'


screen -dmS diff_OUE_r_emoji_AO_0_SP_10 bash -c 'python -m running_time --splits=10 --h_ao=0 --vswhat=r --dataset=emoji --protocol=OUE'
screen -dmS diff_OLHU_r_emoji_AO_0_SP_10 bash -c 'python -m running_time --splits=10 --h_ao=0 --vswhat=r --dataset=emoji --protocol=OLH_User'
screen -dmS diff_OLHS_r_emoji_AO_0_SP_10 bash -c 'python -m running_time --splits=10 --h_ao=0 --vswhat=r --dataset=emoji --protocol=OLH_Server'
screen -dmS diff_HSTU_r_emoji_AO_0_SP_10 bash -c 'python -m running_time --splits=10 --h_ao=0 --vswhat=r --dataset=emoji --protocol=HST_User'
screen -dmS diff_HSTS_r_emoji_AO_0_SP_10 bash -c 'python -m running_time --splits=10 --h_ao=0 --vswhat=r --dataset=emoji --protocol=HST_Server'
