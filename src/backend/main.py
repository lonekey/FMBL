# -*- coding: utf-8 -*-
import argparse
import json
import os
from config import Config
from utils.log import log

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='buglocate')
    parser.add_argument('--doCollect', action='store_true', default=False)
    parser.add_argument('--doMatch', action='store_true', default=False)
    parser.add_argument('--doMakeDataset', action='store_true', default=False)
    parser.add_argument('--doTrain', action='store_true', default=False)
    parser.add_argument('--doPredict', action='store_true', default=False)

    # parser.add_argument('--config', help='user config for buglocate')

    # parser.add_argument('--useTFIDF', action='store_true', default=False)
    # parser.add_argument('--useLearning', action='store_true', default=False)
    # parser.add_argument('--useLength', action='store_true', default=False)
    #
    # parser.add_argument('--file', action='store_true', default=False)
    # parser.add_argument('--method', action='store_true', default=False)
    #
    # parser.add_argument('--bugRepo', help='a website for your bug repository')
    # parser.add_argument('--gitRepo', help='path to local git repository whitch contains folder .git')
    # parser.add_argument('--product', help='product name in bugRepo')
    # parser.add_argument('--maxDatasetSize', help='max history bug reports number for bug localization')
    parser.add_argument('--query', help='file store some information about the bug')
    args = parser.parse_args()

    config = Config()

    hasBugRepo = config.bugRepo != ""

    if not os.path.exists(f'cache/{config.product}'):
        os.makedirs(f'cache/{config.product}')
    if args.doCollect:
        # collect bug reports
        if hasBugRepo and not os.path.exists(f'cache/{config.product}/bug_report.json'):
            from collect import collect_bug_report

            collect_bug_report(config.product, config.bugRepo)
            log('BugRepo Collected !')
        # collect git logs
        if not os.path.exists(f'cache/{config.product}/git_log.csv'):
            from collect import collect_git_log

            collect_git_log(config.product, config.gitRepo)
            log('GitLog Collected !')
    if args.doMatch and hasBugRepo:
        # match bugId-fixCommitId
        if not os.path.exists(f'cache/{config.product}/bug_repo.json'):
            from collect import matchRC

            matchRC(config.product)
        log('Matched bugId-fixCommitId !')

    if args.doMakeDataset:
        if not os.path.exists(f'cache/{config.product}/{config.product}.pkl'):
            from makeDataset import make_pkl

            make_pkl(config.product, config.gitRepo, config.maxDatasetSize, hasBugRepo)
        log('Built dataset !')
    if args.doTrain:
        import bl_Learning_CNN
        bl_Learning_CNN.doTrain(config)

    if args.doPredict:
        from buglocate import predict_M
        # import time
        if not hasBugRepo:
            config.useLearning = False
        # print("doPredict")
        # start = time.time()
        result = predict_M(config, args.query)
        print(json.dumps(result), end='')
        # end = time.time()
        # print("总时间", end-start)
