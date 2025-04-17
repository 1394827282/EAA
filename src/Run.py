# region 导包
import logging
import statistics

import os
import time
import pickle
import warnings

import torch
import torch.multiprocessing as mp
from scipy.stats import mannwhitneyu

warnings.simplefilter("ignore")
warnings.filterwarnings('ignore')

from Agent import NetworkAgent
from Env import get_scenario
from Reward import APHF, ATCF
from Visual import visualize


# endregion

class dotdict(dict):
    def __getattr__(self, name):
        if name in self:
            if isinstance(self[name], type):
                return lambda *args, **kwargs: self[name](*args, **kwargs)
            return self[name]
        else:
            raise AttributeError(f"'dotdict' object has no attribute '{name}'")


# 配置日志
def logger_config(log_path):
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    # logger.addHandler(console)
    return logger



def preprocess_continuous(state, scenario_metadata, histlen):
    if scenario_metadata['maxExecTime'] > scenario_metadata['minExecTime']:
        time_since = (scenario_metadata['maxExecTime'] - state['LastRun']).total_seconds() / (
                scenario_metadata['maxExecTime'] - scenario_metadata['minExecTime']).total_seconds()
    else:
        time_since = 0
    history = [1 if res else 0 for res in state['LastResults'][0:histlen]]

    if len(history) < histlen:
        history.extend([1] * (histlen - len(history)))
    row = [
        state['Duration'] / scenario_metadata['totalTime'],
        time_since,
    ]
    row.extend(history)

    return tuple(row)



def process_scenario(agent, sc, preprocess):
    scenario_metadata = sc.get_ta_metadata()

    if agent.single_testcases:
        for row in sc.testcases():
            x = preprocess(row, scenario_metadata, agent.histlen)
            action = agent.get_action(x)
            row['CalcPrio'] = action

    else:
        states = [preprocess(row, scenario_metadata, agent.histlen) for row in sc.testcases()]
        actions = agent.get_all_actions(states)

        for (tc_idx, action) in enumerate(actions):
            sc.set_testcase_prio(action, tc_idx)

    return sc.submit()

def process_scenario_short(agent, sc, preprocess, unexecuted):
    scenario_metadata = sc.get_ta_metadata()
    if agent.single_testcases:
        for row in sc.testcases():
            # if len(row['LastResults']) == 0:
            x = preprocess(row, scenario_metadata, agent.histlen)
            agent.setshort(x, row['CalcPrio'])

    return sc.submit()
class PrioLearning(object):
    def __init__(self, args, scenario_provider, reward_function, file_prefix, output_dir):
        self.args = args
        self.agent = self.args.Agent(histlen=self.args.his_length, state_size=self.args.his_length + 2,
                                     hidden_size=self.args.hidden_size, lr=self.args.learning_rate,
                                     lambda_reg=self.args.lambda_reg_loss)
        self.scenario_provider = scenario_provider
        self.reward_function = reward_function
        self.preprocess_function = self.args.Preprocess
        self.his_length = self.args.his_length
        self.now_length = self.args.now_length

        self.file_prefix = file_prefix
        self.stats_file = os.path.join(output_dir, '%s_stats' % file_prefix)
        self.agent_file = os.path.join(output_dir, '%s_agent' % file_prefix)

        self.ut = []
        self.ut_napfd = []

    def process_scenario(self, sc, iteration):

        result, unexecuted, flag = process_scenario(self.agent, sc, self.preprocess_function)
        reward, threshold_value = self.reward_function(result, sc, self.args.lambda_reg_reward)

        sum_reward = sum(reward)

        short_reward = ATCF(result, sc, self.args.lambda_reg_reward)
        process_scenario_short(self.agent, sc, self.preprocess_function, unexecuted)

        self.ut.append(result[3]) # napfd

        len_ut = len(self.ut)
        len_win = self.now_length + self.his_length

        retrain = 0
        p_value = 1
        if len_ut >= len_win:
            his = self.ut[len_ut - len_win:len_ut - self.now_length]
            now = self.ut[len_ut - self.now_length:]
            all = self.ut[len_ut - self.now_length - 1:]

            if set(his) == set(now):
                p_value = 1
            else:
                statistic, p_value = mannwhitneyu(his, now)

            alpha = -1
            De = -1
            na = len(all)
            seita = abs(statistics.mean(now))

            if p_value < 0.05:
                nc = 0
                for i in range(0, len(all) - 1):
                    if all[i + 1] - all[i] < 0 and abs(all[i + 1] - all[i]) * 1.0 / all[i] > 0.1:
                        nc = nc + 1
                if nc != 0:
                    De = 1 - nc / na
                    alpha = self.args.lambda_alpha ** (1 - De)
                    retrain = 1
            if 0 < alpha < 1 or result[3] == 0:
                if result[3] == 0:
                    alpha = self.args.lambda_alpha ** (1 - (1 - na / na))

                new_agent = self.args.Agent(histlen=self.args.his_length, state_size=self.args.his_length + 2,
                                            hidden_size=self.args.hidden_size, lr=self.args.learning_rate,
                                            lambda_reg=self.args.lambda_reg_loss)

                new_agent.experience = self.agent.experience

                new_agent.learn_from_experience(iteration, drift='true', threshold_value=threshold_value)


                params_model1 = new_agent.model.state_dict()
                params_model2 = self.agent.model.state_dict()


                combined_params = {}
                for name in params_model1:
                    combined_params[name] = (1 - alpha) * params_model1[name] + alpha * params_model2[name]


                self.agent.model.load_state_dict(combined_params)

                self.agent.episode_history = []

                result, unexecuted, flag = process_scenario(self.agent, sc, self.preprocess_function)

                reward, _ = self.reward_function(result, sc, self.args.lambda_reg_reward)

                short_reward = ATCF(result, sc, self.args.lambda_reg_reward)

                sum_reward = sum(reward)

                self.ut[len(self.ut) - 1] = result[3]



        self.agent.reward_short(short_reward)
        loss = self.agent.reward(reward, iteration, threshold_value=threshold_value)

        return result, reward, retrain, p_value, loss, flag

    def comput2(self, lista):
        try:

            tensor = torch.tensor(lista, dtype=torch.float)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tensor = tensor.to(device)

            ave = tensor.mean()

            squared_diff = (tensor - ave) ** 2

            return ave.item(), squared_diff.mean().item()

        except Exception as e:
            print(f"Error occurred: {e}")
            return 0.0, 0.0

    def find_descending_percentage(self, result):
        if len(result) < 2:
            return None, None, 0

        start_index = None

        for i in range(len(result) - 2, -1, -1):
            if result[i] > result[i + 1]:
                if start_index is None:
                    start_index = i
            else:
                if start_index is not None:
                    break

        if start_index is None:
            return None, None, 0

        percentage_decrease = ((result[start_index] - result[-1]) / result[start_index]) * 100

        return percentage_decrease

    def train(self, iteration):
        stats = {
            'agent': self.agent.name,
            'scenarios': [],
            'rewards': [],
            'rewards_variance': [],
            'durations': [],
            'detected': [],
            'missed': [],
            'ttf': [],
            'napfd': [],
            'recall': [],
            'avg_precision': [],
            'result': [],
            'step': [],
            'env': self.scenario_provider.name,
            'history_length': self.agent.histlen,
            'rewardfun': self.reward_function.__name__,
            'retrain_nums': 0.0,
            'retrain_ratio': 0.0,
            'p_value': [],
        }

        sum_scenarios = 0
        retrain_nums = 0

        for (i, sc) in enumerate(self.scenario_provider, start=1):

            start = time.time()
            (result, reward, retrain, p_value, loss, flag) = self.process_scenario(sc, iteration)
            if flag:
                continue
            end = time.time()
            duration = end - start

            retrain_nums += retrain
            rewards_mean, rewards_variance = self.comput2(reward)

            # Statistics
            sum_scenarios += 1
            stats['scenarios'].append(sc.name)
            stats['rewards'].append(rewards_mean)
            stats['rewards_variance'].append(rewards_variance)
            stats['durations'].append(duration)
            stats['detected'].append(result[0])
            stats['missed'].append(result[1])
            stats['ttf'].append(result[2])
            stats['napfd'].append(result[3])
            stats['recall'].append(result[4])
            stats['avg_precision'].append(result[5])
            stats['result'].append(result)
            stats['step'].append(sum_scenarios)
            stats['p_value'].append(p_value)

        stats['retrain_nums'] = retrain_nums
        stats['retrain_ratio'] = retrain_nums / self.scenario_provider.max_cycles

        pickle.dump(stats, open(self.stats_file + '.p', 'wb'))


def single_experiment(worker_id, init_args):
    args, agent_name, project, reward_name, reward_fun, iteration, gpu = init_args[worker_id]

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    logging.info(f'Current execute: dataset = {project}, reward = {reward_name}')
    file_appendix = 'rq_%s_%s_%s_%d' % (agent_name, project, reward_name, iteration)
    scenario = get_scenario(project)

    rl_learning = PrioLearning(
        args,
        scenario_provider=scenario,
        reward_function=reward_fun,
        file_prefix=file_appendix,
        output_dir=Res_Dir)

    rl_learning.train(iteration)


def run_experiment(args, datasets, reward_funs):
    agent_name = args.Agent(histlen=args.his_length, state_size=args.his_length + 2, hidden_size=args.hidden_size,
                            lr=args.learning_rate, lambda_reg=args.lambda_reg_loss).name
    for project in datasets:
        tasks = []
        num_gpus = torch.cuda.device_count()
        gpu = 0
        for iteration in range(args.Iterations):
            for reward_name, reward_fun in reward_funs.items():
                tasks.append((args, agent_name, project, reward_name, reward_fun, iteration, gpu % num_gpus))
                gpu = gpu + 1
        pool_size = args.Pool_Size

        if project == 'apache_tajo':
            pool_size = 4
        logging.info(f'Dataset {project} begin')
        begin_time = time.time()

        for index in range(0, len(tasks), pool_size):
            tmp_task = tasks[index:index + pool_size]
            mp.spawn(single_experiment,
                     args=(tmp_task,),
                     nprocs=len(tmp_task),
                     join=True)
        logging.info(f'Dataset {project} success costs: {(time.time() - begin_time):.3f}s')


# region
Res_Dir = 'results'
Summarize_DIR = os.path.join('results', 'summarize')
#
datasets = ['apache_drill', 'iofrol', 'google_auto', 'paintcontrol', 'apache_parquet',
            'dspace', 'apache_commons', 'rails', 'google_closure', 'google_guava',
            'mybatis', 'apache_tajo']

reward_funs = {
    "APHF": APHF,
}
args = dotdict({
    'now_length': 5,
    'his_length': 7,
    'hidden_size': 16,
    'learning_rate': 0.02,
    'lambda_alpha': 0.45,
    'lambda_reg_loss': 0.05,
    'lambda_reg_reward': 0.2,

    'Iterations': 30,
    'Pool_Size': 5,
    'Agent': NetworkAgent,
    'Preprocess': preprocess_continuous,
})

# endregion

if __name__ == '__main__':
    os.makedirs(Res_Dir, exist_ok=True)
    os.makedirs(Summarize_DIR, exist_ok=True)
    logger = logger_config(log_path='TCP.log')
    b_time = time.time()

    run_experiment(args, datasets, reward_funs)

    logging.info(f"Experiments run time: {(time.time() - b_time):.3f}s")

    visualize(name='add', Res_Dir=Res_Dir, Summarize_DIR=Summarize_DIR)
    logging.info("All finished.")

    logging.info(f"Theoretical number: {args.Iterations * len(datasets) * len(reward_funs)}")
    logging.info(f"Actual number: {len(os.listdir('results')) - 1}")
    logging.info(f"{'-' * 50}\n")
