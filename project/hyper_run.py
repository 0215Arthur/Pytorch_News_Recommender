# 爆调参数

import signal
from subprocess import Popen, DEVNULL
from argparse import ArgumentParser
from sys import stdout
import sys
import threading
from time import sleep
import GPUtil
from typing import Dict, List

MODELS = {
    'SASRec': 'SASRec_Train.py',
    'NARM': 'NARM_train.py',
    'LESSR': 'LESSR_train.py',
    'SRGNN': 'SRGNN_train.py',
    'CDSBR': 'CDSBR_Train.py',
    'DAGCN': 'DAGCN_Train.py',
    'IAGNN': 'IAGNN_Train.py',
    'StarGNN+': 'StarGNN_Train.py',
    'StarGNN-+': 'StarGNN_Train.py',
    'DAGCN_Origin': 'DAGCN_Origin_Train.py'
}

MODEL_OURS = 'IAGNN'
MODELS['ours'] = MODELS[MODEL_OURS]

SEARCH_RANGE = {
    'lr': [0.001, 0.002, 0.003],  # [0.001, 0.002, 0.005],  # CDSBR: 0.002
    'emb_size': [96, 128, 198, 256],
    'batch': [128, 256, 512],
    # [2, 3, 4], # CDSBR: {lr: 0.001, step:3}, {lr: 0.002, step:3}
    'lr_step': [1, 2, 3, 4],
    'lr_gama': [0.01, 0.05, 0.1, 0.5],
    'GL': [1, 2, 3, 4],   # CDSBR: 1
    'fdrop': [0.1, 0.2, 0.3],  # CDSBR: 0.2
    'tao': [0.1, 1.0, 0.3, 3, 6],    # CDSBR: 1.0
    'epochs': [10, 20]
}

# -1 means variational parameter, or it's fixed
FIXED_PARAMETERS = {
    # [0.001, 128, 512, 3, 0.1, 1, 0.2]
    'CDSBR': [0.001, 128, 512, 3, 0.1, 1, 0.2, 1.0, 10],
    'DAGCN': [-1, 128, 512, -1, 0.1, -1, 0.2, 1.0, 10],
    'DAGCN_Origin': [0.002, 128, 512, 2, 0.1, -1, 0.2, 1.0, 10],
    # 'IAGNN': [0.001, 128, 512, 1, 0.1, -1, 0.2, 1.0, 10],  # yc_BT_4
    'IAGNN': [0.003, 128, 512, 3, 0.1, -1, 0.2, 1.0, 10],  # jdata_cd
    'StarGNN+': [0.001, 256, 100, 3, 0.1, -1, 0.2, 1.0, 15],
    'StarGNN-+': [0.001, 256, 100, 3, 0.1, -1, 0.2, 1.0, 15]
}
FIXED_PARAMETERS['ours'] = FIXED_PARAMETERS[MODEL_OURS]


class HyperPSearcher:

    running_processes_count: threading.Semaphore
    running_processes: List[Popen] = []
    args: dict

    def __init__(self, args: dict) -> None:
        self.running_processes_count = threading.Semaphore(
            args['max_process_count'])
        self.args = args

    def process_end_handler(self, p: Popen):
        print('going to wait for process: {}'.format(p.pid))
        p.wait()
        self.running_processes_count.release()
        print('process {} ends'.format(p.pid))
        self.running_processes.remove(p)

    def search(self):
        # combinations = self.get_hyperparameter_combinations(
        #     FIXED_PARAMETERS[self.args['model']], 0, {})

        # print(combinations)
        device_ids = self.args['gpus']
        threads = []
        is_first_run = True
        for comb in [600, 1800, 2400, 3000, 3600, 4200, 4800, 5400, 6000, 6600]:#[0.1, 0.2, 0.30, 0.40]:
            #[60, 600, 1200, 1800, 2400, 3000, 3600, 4200, 4800, 5400, 6000, 6600]: #[0.95, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]:#[4200, 5400, 6000, 6000]
            # trying to get available semaphore of allowed running processes left
            self.running_processes_count.acquire()

            if self.args['sgpu']:
                device_ids = []
                while len(device_ids) == 0:
                    if is_first_run:
                        is_first_run = False
                    else:
                        sleep(self.args['sgpu_timeout'])

                    try:
                        device_ids = GPUtil.getFirstAvailable(
                            order='memory', maxLoad=self.args['max_gpu_load'], maxMemory=self.args['max_gpu_mem'], verbose=True)
                    except:
                        print('no avaiable GPU for now, will try later...')
                        device_ids = []
                device_id = device_ids[0]
            else:
                device_id = device_ids[0]
                device_ids = device_ids[1:] + device_ids[:1]

            print(device_ids)
            print('got available device id: {}'.format(device_id))
            # start
            pargs = 'python train_eval.py {} {} {} {} {} {}'.format(
                # MODELS[self.args['model']],
                '--gpu={}'.format(device_id),
                '--version=th{}'.format(comb),
                '--data={}'.format(self.args['data']),
                '--model={}'.format(self.args['model']),
                '--time_thresh={}'.format(comb),
                '--time_p={}'.format(0.9)
                # ' '.join([
                #     '--{}={}'.format(k, v) for k, v in comb.items()
                # ]),
            ).split(' ')
            print('going to start process with command: {}'.format(pargs))
            if self.args['show_subprocess_output']:
                p = Popen(pargs)
            else:
                p = Popen(pargs, stdout=DEVNULL)
            self.running_processes.append(p)
            print('started process: {}'.format(p.pid))

            # check process ending in thread, and release semaphore then
            t = threading.Thread(target=self.process_end_handler, args=(p,))
            t.start()
            threads.append(t)

        # block until all processes end
        for t in threads:
            t: threading.Thread = t
            t.join()

    def get_hyperparameter_combinations(self,
                                        fixed_params: List,
                                        index: int,
                                        visited: Dict[int, List[Dict[str, List]]]) -> List[Dict[str, List]]:
        '''
        get all combinations of hyperparams

        Args:
            fixed_params (List): fixed params for current Model.
            index (int): which hyperparam is checking.
            visited (Dict[int,List[Dict[str,List]]]): a cached for recursively traverse all possible combinations.
        '''
        if index in visited.keys():
            return visited[index]

        if index >= len(SEARCH_RANGE):
            # nothing here
            visited[index] = []
            return visited[index]

        key: str = list(SEARCH_RANGE.keys())[index]
        ret: List[Dict[List]] = []
        fixed_param: int = fixed_params[index]
        comb_left: List[Dict[str, List]] = self.get_hyperparameter_combinations(
            fixed_params, index+1, visited)
        if fixed_param != -1:
            if len(comb_left) > 0:
                for comb in comb_left:
                    ret.append(dict({key: fixed_param}, **comb))
            else:
                ret.append({key: fixed_param})

        else:
            for v in SEARCH_RANGE[key]:
                if len(comb_left) > 0:
                    for comb in comb_left:
                        ret.append(dict({key: v}, **comb))
                else:
                    ret.append({key: v})

        visited[index] = ret
        return ret

    def kill_all_running_processes(self):
        for p in self.running_processes:
            print('killing process: {}'.format(p.pid))
            p.kill()


def main():
    p = ArgumentParser('HyperParametersSearch',
                       usage='python HyperPSearch.py --model=CDSBR --sgpu --hide_subprocess_output')
    p.add_argument('--model', type=str, default='tmgm')
    p.add_argument('--gpus', nargs='+', default=[0, 1, 2, 3], type=int)
    p.add_argument('--sgpu', action='store_true')
    p.add_argument('--show_subprocess_output', action='store_true')
    p.add_argument('--max_process_count', type=int, default=4)
    p.add_argument('--max_gpu_load', type=float, default=0.8)
    p.add_argument('--max_gpu_mem', type=float, default=0.6)
    p.add_argument('--sgpu_timeout', type=int, default=90,
                   help='wait for sgpu_timeout seconds to start another process')
    p.add_argument('--comment', type=str, default='None')
    p.add_argument('--data', type=str, default='GLOBO')
    args = vars(p.parse_args())

    def signal_handler(sig, frame):
        print('Ctrl+C pressed, killing all running processes')
        hps.kill_all_running_processes()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    hps = HyperPSearcher(args)
    hps.search()


main()