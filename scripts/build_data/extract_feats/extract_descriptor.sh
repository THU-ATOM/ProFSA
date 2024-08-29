# python scripts/build_data/extract_feats/csv2smi.py \
#     --input_dir /data/prot_frag/moleculenet_test_smi

for file in /data/prot_frag/moleculenet_all/*.smi; do
    python -m mordred $file -o ${file%.smi}_desc.csv -p 1
done
