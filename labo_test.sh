# $1: cfg path (e.g., cfg/asso_opt/flower/flower_allshot_fac.py)
# $2: ckpt path
python main.py --cfg $1 --func asso_opt_main --test --cfg-options bs=512 ckpt_path=$2 ${@:3}