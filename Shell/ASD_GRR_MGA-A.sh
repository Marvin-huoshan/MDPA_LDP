#!/bin/bash
screen -dmS ASD_GRR_vsepsilon_zipf_AO_0_SP_4 bash -c 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate OLH_Attack_post && python -m Overall_Detection --ratio=0.1 --splits=4 --h_ao=0 --vswhat=epsilon --dataset=zipf --protocol=GRR'
screen -dmS ASD_GRR_vsepsilon_emoji_AO_0_SP_4 bash -c 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate OLH_Attack_post && python -m Overall_Detection --ratio=0.1 --splits=4 --h_ao=0 --vswhat=epsilon --dataset=emoji --protocol=GRR'
screen -dmS ASD_GRR_vsepsilon_fire_AO_0_SP_4 bash -c 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate OLH_Attack_post && python -m Overall_Detection --ratio=0.1 --splits=4 --h_ao=0 --vswhat=epsilon --dataset=fire --protocol=GRR'
screen -dmS ASD_GRR_vsepsilon_zipf_UA bash -c 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate OLH_Attack_post && python -m Overall_Detection --ratio=0 --splits=4 --h_ao=0 --vswhat=epsilon --dataset=zipf --protocol=GRR'
screen -dmS ASD_GRR_vsepsilon_emoji_UA bash -c 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate OLH_Attack_post && python -m Overall_Detection --ratio=0 --splits=4 --h_ao=0 --vswhat=epsilon --dataset=emoji --protocol=GRR'
screen -dmS ASD_GRR_vsepsilon_fire_UA bash -c 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate OLH_Attack_post && python -m Overall_Detection --ratio=0 --splits=4 --h_ao=0 --vswhat=epsilon --dataset=fire --protocol=GRR'
screen -dmS ASD_GRR_vsbeta_zipf_AO_0_SP_4 bash -c 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate OLH_Attack_post && python -m Overall_Detection --splits=4 --h_ao=0 --vswhat=beta --dataset=zipf --protocol=GRR'
screen -dmS ASD_GRR_vsbeta_emoji_AO_0_SP_4 bash -c 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate OLH_Attack_post && python -m Overall_Detection --splits=4 --h_ao=0 --vswhat=beta --dataset=emoji --protocol=GRR'
screen -dmS ASD_GRR_vsbeta_fire_AO_0_SP_4 bash -c 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate OLH_Attack_post && python -m Overall_Detection --splits=4 --h_ao=0 --vswhat=beta --dataset=fire --protocol=GRR'
screen -dmS ASD_GRR_vsr_prime_zipf_AO_0_SP_4 bash -c 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate OLH_Attack_post && python -m Overall_Detection --ratio=0.1 --splits=4 --h_ao=0 --vswhat=r_prime --dataset=zipf --protocol=GRR'
screen -dmS ASD_GRR_vsr_prime_emoji_AO_0_SP_4 bash -c 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate OLH_Attack_post && python -m Overall_Detection --ratio=0.1 --splits=4 --h_ao=0 --vswhat=r_prime --dataset=emoji --protocol=GRR'
screen -dmS ASD_GRR_vsr_prime_fire_AO_0_SP_4 bash -c 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate OLH_Attack_post && python -m Overall_Detection --ratio=0.1 --splits=4 --h_ao=0 --vswhat=r_prime --dataset=fire --protocol=GRR'


