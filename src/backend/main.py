import argparse
import json
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='buglocate')
    parser.add_argument('--doCollect', action='store_true', default=False)
    parser.add_argument('--doMatch', action='store_true', default=False)
    parser.add_argument('--doMakeDataset', action='store_true', default=False)
    parser.add_argument('--doTrain', action='store_true', default=False)
    parser.add_argument('--doPredict', action='store_true', default=False)

    parser.add_argument('--useTFIDF', action='store_true', default=False)
    parser.add_argument('--useLearning', action='store_true', default=False)
    parser.add_argument('--useLength', action='store_true', default=False)

    parser.add_argument('--file', action='store_true', default=False)
    parser.add_argument('--method', action='store_true', default=False)

    parser.add_argument('--bugRepo', help='a website for you bug repository')
    parser.add_argument('--gitRepo', help='path to local git repository whitch contains folder .git')
    parser.add_argument('--product', help='product name in bugRepo')
    parser.add_argument('--maxDatasetSize', help='max history bug reports number for bug localization')
    parser.add_argument('--query', help='file store some information about the bug')


    args = parser.parse_args()
    if args.doCollect:
        # collect bug reports
        if not os.path.exists(f'cache/{args.product}/bug_report.json'):
            from collect import collect_bug_report
            collect_bug_report(args.product, args.bugRepo)
        print('BugRepo Collected !')
        # collect git logs
        if not os.path.exists(f'cache/{args.product}/git_log.csv'):
            from collect import collect_git_log
            collect_git_log(args.product, args.gitRepo)
        print('GitLog Collected !', end='')
    if args.doMatch:
        # match bugId-fixCommitId
        if not os.path.exists(f'cache/{args.product}/bug_repo.json'):
            from collect import matchRC
            matchRC(args.product)
        print('Matched bugId-fixCommitId !', end='')

    if args.doMakeDataset:
        if not os.path.exists(f'cache/{args.product}/{args.product}.pkl'):
            from makeDataset import make_pkl
            make_pkl(args.product, args.gitRepo, args.maxDatasetSize)
        print('Built dataset !', end='')
    if args.doPredict:
        from buglocate import predict
        import time
        # print("doPredict")
        start = time.time()
        result = predict(args.product, args.query)
        print(json.dumps(result), end='')
        end = time.time()
        # print(end-start)




