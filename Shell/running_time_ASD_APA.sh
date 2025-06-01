#!/bin/bash
screen -dmS ASD_OUE_vsepsilon_zipf_AO_20_SP_4 bash -c 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate OLH_Attack_post && python -m running_time_Overall --ratio=0.1 --splits=4 --h_ao=20 --vswhat=epsilon --dataset=zipf --protocol=OUE'
screen -dmS ASD_OLHU_vsepsilon_zipf_AO_20_SP_4 bash -c 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate OLH_Attack_post && python -m running_time_Overall --ratio=0.1 --splits=4 --h_ao=20 --vswhat=epsilon --dataset=zipf --protocol=OLH_User'
screen -dmS ASD_HSTU_vsepsilon_zipf_AO_20_SP_4 bash -c 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate OLH_Attack_post && python -m running_time_Overall --ratio=0.1 --splits=4 --h_ao=20 --vswhat=epsilon --dataset=zipf --protocol=HST_User'
screen -dmS ASD_OUE_vsepsilon_emoji_AO_20_SP_4 bash -c 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate OLH_Attack_post && python -m running_time_Overall --ratio=0.1 --splits=4 --h_ao=20 --vswhat=epsilon --dataset=emoji --protocol=OUE'
screen -dmS ASD_OLHU_vsepsilon_emoji_AO_20_SP_4 bash -c 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate OLH_Attack_post && python -m running_time_Overall --ratio=0.1 --splits=4 --h_ao=20 --vswhat=epsilon --dataset=emoji --protocol=OLH_User'
screen -dmS ASD_HSTU_vsepsilon_emoji_AO_20_SP_4 bash -c 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate OLH_Attack_post && python -m running_time_Overall --ratio=0.1 --splits=4 --h_ao=20 --vswhat=epsilon --dataset=emoji --protocol=HST_User'
screen -dmS ASD_OUE_vsepsilon_fire_AO_20_SP_4 bash -c 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate OLH_Attack_post && python -m running_time_Overall --ratio=0.1 --splits=4 --h_ao=20 --vswhat=epsilon --dataset=fire --protocol=OUE'
screen -dmS ASD_OLHU_vsepsilon_fire_AO_20_SP_4 bash -c 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate OLH_Attack_post && python -m running_time_Overall --ratio=0.1 --splits=4 --h_ao=20 --vswhat=epsilon --dataset=fire --protocol=OLH_User'
screen -dmS ASD_HSTU_vsepsilon_fire_AO_20_SP_4 bash -c 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate OLH_Attack_post && python -m running_time_Overall --ratio=0.1 --splits=4 --h_ao=20 --vswhat=epsilon --dataset=fire --protocol=HST_User'


screen -dmS ASD_OUE_vsbeta_zipf_AO_20_SP_4 bash -c 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate OLH_Attack_post && python -m running_time_Overall --splits=4 --h_ao=20 --vswhat=beta --dataset=zipf --protocol=OUE'
screen -dmS ASD_OLHU_vsbeta_zipf_AO_20_SP_4 bash -c 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate OLH_Attack_post && python -m running_time_Overall --splits=4 --h_ao=20 --vswhat=beta --dataset=zipf --protocol=OLH_User'
screen -dmS ASD_HSTU_vsbeta_zipf_AO_20_SP_4 bash -c 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate OLH_Attack_post && python -m running_time_Overall --splits=4 --h_ao=20 --vswhat=beta --dataset=zipf --protocol=HST_User'
screen -dmS ASD_OUE_vsbeta_emoji_AO_20_SP_4 bash -c 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate OLH_Attack_post && python -m running_time_Overall --splits=4 --h_ao=20 --vswhat=beta --dataset=emoji --protocol=OUE'
screen -dmS ASD_OLHU_vsbeta_emoji_AO_20_SP_4 bash -c 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate OLH_Attack_post && python -m running_time_Overall --splits=4 --h_ao=20 --vswhat=beta --dataset=emoji --protocol=OLH_User'
screen -dmS ASD_HSTU_vsbeta_emoji_AO_20_SP_4 bash -c 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate OLH_Attack_post && python -m running_time_Overall --splits=4 --h_ao=20 --vswhat=beta --dataset=emoji --protocol=HST_User'
screen -dmS ASD_OUE_vsbeta_fire_AO_20_SP_4 bash -c 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate OLH_Attack_post && python -m running_time_Overall --splits=4 --h_ao=20 --vswhat=beta --dataset=fire --protocol=OUE'
screen -dmS ASD_OLHU_vsbeta_fire_AO_20_SP_4 bash -c 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate OLH_Attack_post && python -m running_time_Overall --splits=4 --h_ao=20 --vswhat=beta --dataset=fire --protocol=OLH_User'
screen -dmS ASD_HSTU_vsbeta_fire_AO_20_SP_4 bash -c 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate OLH_Attack_post && python -m running_time_Overall --splits=4 --h_ao=20 --vswhat=beta --dataset=fire --protocol=HST_User'

