import numpy as np
import math


def TCF(result, sc, lambda_reg_reward):
    # result 包含以下内容 [detected_failures, undetected_failures, ttf, napfd, recall, avg_precision, detection_ranks]
    # sc 是一个循环周期的数据，['Id', 'Name', 'Duration', 'CalcPrio', 'LastRun', 'LastResults']
    ordered_rewards = []
    if result[0] == 0:
        return [0.0] * sc.no_testcases

    rank_idx = np.array(result[-1]) - 1  # 执行过且检测到错误的测试用例的索引
    # scheduled_testcases 是执行过的测试用例列表，一条测试用例信息是一个元素
    no_scheduled = len(sc.scheduled_testcases)

    rewards = np.zeros(no_scheduled)
    rewards[rank_idx] = 1

    for tc in sc.testcases():
        try:
            idx = sc.scheduled_testcases.index(tc)
            ordered_rewards.append(rewards[idx])
        except ValueError:
            ordered_rewards.append(0.0)  # Unscheduled test case

    ordered_rewards = np.array(ordered_rewards)
    ordered_rewards = (1 - lambda_reg_reward) * ordered_rewards + lambda_reg_reward * np.ones_like(ordered_rewards)
    return ordered_rewards

# 陈凡亮学长的ATCF奖励函数，用于后续消融实验
def ATCF(result, sc, lambda_reg_reward):
    # result 包含以下内容 [detected_failures, undetected_failures, ttf, napfd, recall, avg_precision, detection_ranks]
    # sc 是一个循环周期的数据，['Id', 'Name', 'Duration', 'CalcPrio', 'LastRun', 'LastResults']
    ordered_rewards = []
    if result[0] == 0:
        return [0.0] * sc.no_testcases

    rank_idx = np.array(result[-1]) - 1  # 执行过且检测到错误的测试用例的索引
    # scheduled_testcases 是执行过的测试用例列表，一条测试用例信息是一个元素
    no_scheduled = len(sc.scheduled_testcases)

    rewards = np.zeros(no_scheduled)
    rewards[rank_idx] = 1

    for tc in sc.testcases():
        v = 0.0
        try:
            # 如果测试用例是新的，进行附加奖励
            if len(tc["LastResults"]) == 0:
                v = 1.0
            idx = sc.scheduled_testcases.index(tc)
            ordered_rewards.append(v + rewards[idx] * 2)
        except ValueError:
            ordered_rewards.append(0.0)  # Unscheduled test case

    ordered_rewards = np.array(ordered_rewards)
    ordered_rewards = (1 - lambda_reg_reward) * ordered_rewards + lambda_reg_reward * np.ones_like(ordered_rewards)
    return ordered_rewards

def ATCF_new(result, sc, lambda_reg_reward):
    # result 包含以下内容 [detected_failures, undetected_failures, ttf, napfd, recall, avg_precision, detection_ranks]
    # sc 是一个循环周期的数据，['Id', 'Name', 'Duration', 'CalcPrio', 'LastRun', 'LastResults']
    ordered_rewards = []
    if result[0] == 0:
        return [0.0] * sc.no_testcases

    rank_idx = np.array(result[-1]) - 1
    no_scheduled = len(sc.scheduled_testcases)

    rewards = np.zeros(no_scheduled)
    rewards[rank_idx] = 1

    for tc in sc.testcases():
        v = 0.0
        try:
            if len(tc["LastResults"]) == 0:
                v = 1.0
            idx = sc.scheduled_testcases.index(tc)
            ordered_rewards.append((v + rewards[idx] * 2)/ 3 * 60)
        except ValueError:
            ordered_rewards.append(0.0)

    ordered_rewards = np.array(ordered_rewards)
    return ordered_rewards

def APHF(result, sc, lambda_reg_reward):
    ordered_rewards = []
    if result[0] == 0:
        for tc in sc.testcases():
            hisresults = tc['LastResults'].copy()
            hisresults.insert(0, 0)
            detection_ranks = []
            rank_counter = 1
            no_testcases = len(hisresults)
            total_failure_count = sum(hisresults)
            for i in hisresults:
                if i:
                    detection_ranks.append(rank_counter)
                rank_counter += 1
            if total_failure_count > 0 and no_testcases > 0:
                aphf = float('%.2f' % (
                        1.0 - float(sum(detection_ranks)) / (total_failure_count * no_testcases) + 1.0 / (
                        2 * no_testcases))) * 100
            else:
                aphf = 0.0

            aphf = (1 - lambda_reg_reward) * aphf + lambda_reg_reward * 100
            ordered_rewards.append(aphf)

    # detected error, update hisresults
    else:
        rank_idx = np.array(result[-1]) - 1
        no_scheduled = len(sc.scheduled_testcases)

        rewards = np.zeros(no_scheduled)
        rewards[rank_idx] = 100

        for tc in sc.testcases():
            hisresults = tc['LastResults'].copy()
            try:
                idx = sc.scheduled_testcases.index(tc)

                # hisresults for failed tc
                if rewards[idx] == 100:
                    hisresults.insert(0, 1)

                # hisresults for pass tc
                else:
                    hisresults.insert(0, 0)

            except ValueError:
                pass  # Unscheduled test case

            detection_ranks = []
            rank_counter = 1
            no_testcases = len(hisresults)
            total_failure_count = sum(hisresults)
            for i in hisresults:
                if i:
                    detection_ranks.append(rank_counter)
                rank_counter += 1
            if total_failure_count > 0 and no_testcases > 0:
                aphf = float('%.2f' % (
                        1.0 - float(sum(detection_ranks)) / (total_failure_count * no_testcases) + 1.0 / (
                        2 * no_testcases))) * 100
            else:
                aphf = 0.0
            aphf = (1 - lambda_reg_reward) * aphf + lambda_reg_reward * 100
            ordered_rewards.append(aphf)


    ordered_rewards_sorted = np.sort(ordered_rewards)[::-1]
    threshold_index = len(ordered_rewards_sorted) // 3
    threshold_value = ordered_rewards_sorted[threshold_index]
    return ordered_rewards, threshold_value

