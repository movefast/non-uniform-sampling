metrics = {"msbpe":{},"ve":{}, "all_reward_sums": {}}

results = pd.DataFrame(columns = ['agent', 'score', 'params'],
                                  index = list(range(MAX_EVALS)))
