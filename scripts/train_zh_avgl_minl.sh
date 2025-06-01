# bash train_zh_avgl_minl.sh path=exp/bart.0/model seed=0 first_stage_steps=510

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

    python -u seq2seq.py train -b -d $devices --seed=$seed --update-steps=$update -c $config -p $path.avgl_minl.1 --cache --amp --encoder $encoder --bart "fnlp/bart-large-chinese" --train data/cgec/lang8_5xhsk.yuezhang.train --dev data/cgec/mucgec.dev

    cp $path.avgl_minl.1 $path.avgl_minl.2

    python -u seq2seq.py train -d $devices --seed=$seed --update-steps=$update -c $config -p $path.avgl_minl.2 --cache --amp --encoder $encoder --bart "fnlp/bart-large-chinese" --train data/cgec/fcgec.train.mr --dev data/cgec/fcgec.dev.mr --warmup-steps=200 --ref mr --aggs='avg-min' --first-stage-steps=$first_stage_steps
}