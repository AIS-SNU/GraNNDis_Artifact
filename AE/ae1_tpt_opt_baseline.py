import os
import sys
import time

import torch.multiprocessing as mp
import select
import paramiko

import configs

class Commands:
    def __init__(self, retry_time=0):
        self.retry_time = retry_time
        pass

    def run_cmd(self, host_ip, command):
        i = 0
        while True:
            try:
                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(host_ip)
                break
            except paramiko.AuthenticationException:
                print("Authentication failed when connecting to %s" % host_ip)
                sys.exit(1)
            except:
                print("Could not SSH to %s, waiting for it to start" % host_ip)
                i += 1
                time.sleep(2)

            # If we could not connect within time limit
            if i >= self.retry_time:
                print("Could not connect to %s. Giving up" % host_ip)
                sys.exit(1)
            # After connection is successful
            # Send the command

        # print command
        print('> ' + command)
        # execute commands
        stdin, stdout, stderr = ssh.exec_command(command)
        
        print(stderr.read().decode("euc-kr"))
        stdin.close()

        # TODO() : if an error is thrown, stop further rules and revert back changes
        # Wait for the command to terminate
        while not stdout.channel.exit_status_ready():
            # Only print data if there is data to read in the channel
            if stdout.channel.recv_ready():
                rl, wl, xl = select.select([ stdout.channel ], [ ], [ ], 0.0)
                if len(rl) > 0:
                    tmp = stdout.channel.recv(1024)
                    output = tmp.decode()
                    print(output)

        # Close SSH connection
        ssh.close()
        return

if __name__ == '__main__':

    """
    Set Server Environment
    """
    env_loc = configs.global_configs['env_loc']
    runner_loc = configs.global_configs['runner_loc']
    workspace_loc = configs.global_configs['workspace_loc']
    data_loc = configs.global_configs['data_loc']
    num_runners = configs.global_configs['num_runners']
    gpus_per_server = configs.global_configs['gpus_per_server']
    hosts = configs.global_configs['hosts']
    assert len(hosts) == num_runners, 'our script requires a host per a runner'

    """
    SSH Connection Class
    """
    runners = list()
    for i in range(num_runners):
        runners.append(Commands())
    

    """
    Set Common Dataset Information
    """
    dataset_list = list()
    arxiv_dict = {
        'dataset': 'ogbn-arxiv',
        'dropout': 0.5,
        'lr': 0.01
    }
    reddit_dict = {
        'dataset': 'reddit',
        'dropout': 0.5,
        'lr': 0.01
    }
    product_dict = {
        'dataset': 'ogbn-products',
        'dropout': 0.3,
        'lr': 0.003
    }


    dataset_list = [arxiv_dict, reddit_dict, product_dict]
    sampler_list = ['sage']

    """
    Iteration
    """

    for dataset_dict in dataset_list:
        for n_layer in [3]:
            for sampler in sampler_list:
                remove_tmp = True
                for num_server in [2]:
                    for check_intra_only in [False]:
                        for hidden_size in [64]:
                            """
                            Make an Experiment
                            """
                            if n_layer > 5:
                                model_type = 'deepgcn'
                            else:
                                model_type = 'graphsage'
                            shared_cmd = """{env_loc} {runner_loc} \
                            --dataset {dataset} \
                            --dropout {dropout} \
                            --sampler {sampler} \
                            --lr {lr} \
                            --parts-per-node {gpus_per_server} \
                            --n-partitions {num_parts} \
                            --n-epochs 100 \
                            --model {model_type} \
                            --n-layers {n_layers} \
                            --n-linear 1 \
                            --n-hidden {hidden_size} \
                            --log-every 10 \
                            --master-addr {master_addr} \
                            --port 7524 \
                            --debug \
                            --time-calc \
                            --no-eval \
                            --fix-seed \
                            --seed 7524 \
                            --backend nccl \
                            --dataset-path {data_loc} \
                            --exp-id 1 \
                            --create-json 1 \
                            --json-path {workspace_loc}/Logs/granndis_opt_baseline \
                            --project granndis_ae_opt_baseline \
                            """.format(
                                env_loc      = env_loc,
                                runner_loc   = runner_loc,
                                data_loc     = data_loc,
                                workspace_loc = workspace_loc,
                                gpus_per_server = gpus_per_server,
                                num_parts = gpus_per_server * num_server,
                                dataset      = dataset_dict['dataset'],
                                sampler = sampler,
                                model_type   = model_type,
                                dropout      = dataset_dict['dropout'],
                                lr           = dataset_dict['lr'],
                                n_layers     = (n_layer + 1), # for deepgcn... we need to plus 1
                                hidden_size  = hidden_size,
                                master_addr  = hosts[0],
                            )

                            if remove_tmp:
                                shared_cmd = shared_cmd + """--remove-tmp """
                                remove_tmp = False

                            # if dataset_dict['dataset'] == 'ogbn-papers100m':
                            #     shared_cmd = shared_cmd + """--partition-method random """

                            if check_intra_only:
                                shared_cmd = shared_cmd + """--check-intra-only """

                            def __is_list_in_target(list, target):
                                is_in = False
                                for element in list:
                                    if element in target:
                                        is_in = True
                                        break
                                return is_in

                            if not __is_list_in_target(list=['papers'], target=dataset_dict['dataset']):
                                shared_cmd = shared_cmd + """--inductive """

                            """
                            Pre-Cleaning
                            """
                            kill_cmds = list()
                            for i in range(num_server):
                                kill_cmds.append('pkill -ef spawn && rm -rf ~/partitions')

                            processes = []
                            mp.set_start_method('spawn', force=True)
                            for host, runner, cmd in zip(hosts, runners, kill_cmds):
                                p = mp.Process(target=runner.run_cmd, args=(host, cmd))
                                p.start()
                                processes.append(p)
                            
                            for p in processes:
                                p.join()

                            """
                            Run an Experiment :)
                            """

                            cmds = list()
                            for i in range(num_server):
                                runner_cmd = shared_cmd + """--total-nodes %d """ % num_server
                                runner_cmd += """--node-rank %d """ % i
                                runner_cmd += "\n"
                                cmds.append(runner_cmd)

                            processes = []
                            mp.set_start_method('spawn', force=True)
                            # Note that python zip only iterates for shorter list!!!
                            # so do not worry for running on not dedicated runners
                            for host, runner, cmd in zip(hosts, runners, cmds):
                                p = mp.Process(target=runner.run_cmd, args=(host, cmd))
                                p.start()
                                processes.append(p)
                            
                            for p in processes:
                                p.join()

                            """
                            Post-Cleaning
                            """
                            kill_cmds = list()
                            for i in range(num_server):
                                kill_cmds.append('pkill -ef spawn')

                            processes = []
                            mp.set_start_method('spawn', force=True)
                            for host, runner, cmd in zip(hosts, runners, kill_cmds):
                                p = mp.Process(target=runner.run_cmd, args=(host, cmd))
                                p.start()
                                processes.append(p)
                            
                            for p in processes:
                                p.join()