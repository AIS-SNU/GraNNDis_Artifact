rm -rf partitions/

python main.py \
--dataset ogbn-arxiv \
--dropout 0.5 \
--lr 0.01 \
--n-partitions 4 \
--n-epochs 100 \
--time-calc \
--model graphsage \
--n-layers 4 \
--n-linear 1 \
--n-hidden 64 \
--log-every 10 \
--backend nccl \
--dataset-path ~/datasets/granndis_ae/ \
--create-json 1 \
--json-path ./baseline_test_logs \
--project granndis_test \
--no-eval \
--inductive \
--total-nodes 1 \

python main.py \
--dataset reddit \
--dropout 0.5 \
--lr 0.01 \
--n-partitions 4 \
--n-epochs 100 \
--time-calc \
--model graphsage \
--n-layers 4 \
--n-linear 1 \
--n-hidden 64 \
--log-every 10 \
--backend nccl \
--dataset-path ~/datasets/granndis_ae/ \
--create-json 1 \
--json-path ./baseline_test_logs \
--project granndis_test \
--no-eval \
--inductive \
--total-nodes 1 \

python main.py \
--dataset ogbn-products \
--dropout 0.3 \
--lr 0.003 \
--n-partitions 4 \
--n-epochs 100 \
--time-calc \
--model graphsage \
--n-layers 4 \
--n-linear 1 \
--n-hidden 64 \
--log-every 10 \
--backend nccl \
--dataset-path ~/datasets/granndis_ae/ \
--create-json 1 \
--json-path ./baseline_test_logs \
--project granndis_test \
--no-eval \
--inductive \
--total-nodes 1 \