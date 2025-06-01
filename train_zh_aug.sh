# bash train_zh_aug.sh path=exp/bart.0/model seed=0

set -o nounset
set -o errexit
set -o pipefail

{
    . scripts/set_environment.sh
    args=$@
    for arg in $args; do
        eval "$arg"
    done

    echo "devices:   ${devices:=0,1}"
    echo "update:    ${update:=6}"
    echo "seed:      ${seed:=0}"
    echo "path:      ${path:=exp/transformer/model}"
    echo "encoder:   ${encoder:=bart}"
    echo "config:    ${config:=configs/bart.zh.ini}"

    printf "Current commits:\n$(git log -1 --oneline)\n3rd parties:\n"
    cd 3rdparty/parser/ && printf "parser\n$(git log -1 --oneline)\n" && cd ../..

    exp_dir="$(dirname "$path")"
    mkdir -p $exp_dir
    mkdir -p $exp_dir/data

    python -u seq2seq.py train -b -d $devices --seed=$seed --update-steps=$update -c $config -p $path.aug.1 --cache --amp --encoder $encoder --bart "fnlp/bart-large-chinese" --train data/cgec/lang8_5xhsk.yuezhang.train --dev data/cgec/mucgec.dev

    cp $path.aug.1 $path.aug.2

    python -u seq2seq.py train -d $devices --seed=$seed --update-steps=$update -c $config -p $path.aug.2 --cache --amp --encoder $encoder --bart "fnlp/bart-large-chinese" --train data/cgec/w.hcjiang.v2.train --dev data/cgec/fcgec.dev --warmup-steps=200

    cp $path.aug.2 $path.aug.full_shot.3
    
    python -u seq2seq.py train -d $devices --seed=$seed --update-steps=$update -c $config -p $path.aug.full_shot.3 --cache --amp --encoder $encoder --bart "fnlp/bart-large-chinese" --train data/cgec/fcgec.train --dev data/cgec/fcgec.dev --warmup-steps=200
}