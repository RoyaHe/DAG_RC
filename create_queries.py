import pickle
import os.path as osp
import numpy as np
from collections import defaultdict
import random
from copy import deepcopy
import time
import logging
import os
import argparse

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Generating Explanation Datasets',
        usage='create_queries1.py [<args>] [-h | --help]'
    )

    parser.add_argument('--base_path', default="/Users/royahe/Documents/PythonProjects/DAG_Queries/data")
    parser.add_argument('--dataset', default="FB15k-237_DAG_Easy")
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--gen_train_num', type=int, default=0)
    parser.add_argument('--gen_valid_num', type=int, default=0)
    parser.add_argument('--gen_test_num', type=int, default=0)
    parser.add_argument('--max_ans_num', type=int, default=1e6)
    parser.add_argument('--reindex', type=bool, default=False)
    parser.add_argument('--gen_train', type=bool, default=False)
    parser.add_argument('--gen_valid', type=bool, default=False)
    parser.add_argument('--gen_test', type=bool, default=False)
    parser.add_argument('--gen_id', type=str, default='6.7') #7
    parser.add_argument('--save_name', type=bool, default=False)
    parser.add_argument('--index_only', type=bool, default=False)

    return parser.parse_args(args)

def set_logger(save_path, query_name, print_on_screen=False):
    '''
    Write logs to checkpoint and console
    '''

    log_file = os.path.join(save_path, '%s.log' % (query_name))

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    if print_on_screen:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)


def set_global_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def index_dataset(base_path, dataset_name, force=False):
    base_path = base_path + '/' + dataset_name
    files = ['train.txt', 'valid.txt', 'test.txt']
    indexified_files = ['train_indexified.txt', 'valid_indexified.txt', 'test_indexified.txt']

    return_flag = True
    for i in range(len(indexified_files)):
        if not osp.exists(osp.join(base_path, indexified_files[i])):
            return_flag = False
            break
    if return_flag and not force:
        print("index file exists")
        return

    ent2id, rel2id, id2rel, id2ent = {}, {}, {}, {}

    entid, relid = 0, 0

    with open(osp.join(base_path, files[0])) as f:
        lines = f.readlines()
        file_len = len(lines)

    for p, indexified_p in zip(files, indexified_files):
        fw = open(osp.join(base_path, indexified_p), "w")
        with open(osp.join(base_path, p), 'r') as f:
            for i, line in enumerate(f):
                print('[%d/%d]' % (i, file_len), end='\r')
                e1, rel, e2 = line.split('\t')
                e1 = e1.strip()
                e2 = e2.strip()
                rel = rel.strip()
                rel_reverse = '-' + rel
                rel = '+' + rel
                # rel_reverse = rel+ '_reverse'

                if p == "train.txt":
                    if e1 not in ent2id.keys():
                        ent2id[e1] = entid
                        id2ent[entid] = e1
                        entid += 1

                    if e2 not in ent2id.keys():
                        ent2id[e2] = entid
                        id2ent[entid] = e2
                        entid += 1

                    if not rel in rel2id.keys():
                        rel2id[rel] = relid
                        id2rel[relid] = rel
                        assert relid % 2 == 0
                        relid += 1

                    if not rel_reverse in rel2id.keys():
                        rel2id[rel_reverse] = relid
                        id2rel[relid] = rel_reverse
                        assert relid % 2 == 1
                        relid += 1

                if e1 in ent2id.keys() and e2 in ent2id.keys():
                    fw.write("\t".join([str(ent2id[e1]), str(rel2id[rel]), str(ent2id[e2])]) + "\n")
                    fw.write("\t".join([str(ent2id[e2]), str(rel2id[rel_reverse]), str(ent2id[e1])]) + "\n")
        fw.close()

    with open(osp.join(base_path, "stats.txt"), "w") as fw:
        fw.write("numentity: " + str(len(ent2id)) + "\n")
        fw.write("numrelations: " + str(len(rel2id)))
    with open(osp.join(base_path, 'ent2id.pkl'), 'wb') as handle:
        pickle.dump(ent2id, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(osp.join(base_path, 'rel2id.pkl'), 'wb') as handle:
        pickle.dump(rel2id, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(osp.join(base_path, 'id2ent.pkl'), 'wb') as handle:
        pickle.dump(id2ent, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(osp.join(base_path, 'id2rel.pkl'), 'wb') as handle:
        pickle.dump(id2rel, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('num entity: %d, num relation: %d' % (len(ent2id), len(rel2id)))
    print("indexing finished!!")

def construct_graph(base_path, indexified_files):
    # knowledge graph
    # kb[e][rel] = set([e, e, e])
    ent_in, ent_out = defaultdict(lambda: defaultdict(set)), defaultdict(lambda: defaultdict(set))
    rel_dr = defaultdict(lambda: defaultdict(set))
    ent_in_ent, ent_out_ent = defaultdict(lambda: defaultdict(set)), defaultdict(lambda: defaultdict(set))
    for indexified_p in indexified_files:
        with open(os.path.join(base_path, indexified_p)) as f:
            for i, line in enumerate(f):
                if len(line) == 0:
                    continue
                e1, rel, e2 = line.split('\t')
                e1 = int(e1.strip())
                e2 = int(e2.strip())
                rel = int(rel.strip())
                ent_out[e1][rel].add(e2)
                ent_in[e2][rel].add(e1)
                rel_dr[rel]['dom'].add(e1)
                rel_dr[rel]['range'].add(e2)
                ent_in_ent[e2][e1].add(rel)
                ent_out_ent[e1][e2].add(rel)

    rel_ent_in_ent = defaultdict(lambda: defaultdict(set))
    for i in ent_in_ent.keys():
        for j in ent_in_ent[i].keys():
            num_rel = len(ent_in_ent[i][j])
            rel_ent_in_ent[num_rel][i].add(j)

    rel_ent_out_ent = defaultdict(lambda: defaultdict(set))
    for i in ent_out_ent.keys():
        for j in ent_out_ent[i].keys():
            num_rel = len(ent_out_ent[i][j])
            rel_ent_out_ent[num_rel][i].add(j)
    return ent_in, ent_out, rel_dr, ent_in_ent, ent_out_ent, rel_ent_in_ent, rel_ent_out_ent



def list2tuple(l):
    return tuple(list2tuple(x) if type(x) == list else x for x in l)


def tuple2list(t):
    return list(tuple2list(x) if type(x) == tuple else x for x in t)


def write_links(dataset, ent_out, small_ent_out, max_ans_num, name):
    queries = defaultdict(set)
    tp_answers = defaultdict(set)
    fn_answers = defaultdict(set)
    fp_answers = defaultdict(set)
    num_more_answer = 0
    for ent in ent_out:
        for rel in ent_out[ent]:
            if len(ent_out[ent][rel]) <= max_ans_num:
                queries[('e', ('r',))].add((ent, (rel,)))
                tp_answers[(ent, (rel,))] = small_ent_out[ent][rel]
                fn_answers[(ent, (rel,))] = ent_out[ent][rel]
            else:
                num_more_answer += 1

    with open('./data/%s/%s-queries.pkl' % (dataset, name), 'wb') as f:
        pickle.dump(queries, f)
    with open('./data/%s/%s-tp-answers.pkl' % (dataset, name), 'wb') as f:
        pickle.dump(tp_answers, f)
    with open('./data/%s/%s-fn-answers.pkl' % (dataset, name), 'wb') as f:
        pickle.dump(fn_answers, f)
    with open('./data/%s/%s-fp-answers.pkl' % (dataset, name), 'wb') as f:
        pickle.dump(fp_answers, f)
    print(num_more_answer)


def ground_queries(dataset, query_structure, ent_in, ent_out, small_ent_in, small_ent_out, gen_num, max_ans_num,
                   query_name, mode, ent2id, rel2id, rel_dr, small_rel_dr, ent_in_ent, ent_out_ent, small_ent_in_ent, small_ent_out_ent,
                   rel_ent_in_ent, rel_ent_out_ent, small_rel_ent_in_ent, small_rel_ent_out_ent):
    num_sampled, num_try, num_repeat, num_more_answer, num_broken, num_no_extra_answer, num_no_extra_negative, num_empty = 0, 0, 0, 0, 0, 0, 0, 0
    tp_ans_num, fp_ans_num, fn_ans_num = [], [], []
    queries = defaultdict(set)
    tp_answers = defaultdict(set)
    fp_answers = defaultdict(set)
    fn_answers = defaultdict(set)
    s0 = time.time()
    old_num_sampled = -1
    while num_sampled < gen_num:
        '''
        if num_sampled != 0:
            if num_sampled % (gen_num // 100) == 0 and num_sampled != old_num_sampled:
                logging.info(
                    '%s %s: [%d/%d], avg time: %s, try: %s, repeat: %s: more_answer: %s, broken: %s, no extra: %s, no negative: %s empty: %s' % (
                    mode,
                    query_structure,
                    num_sampled, gen_num, (time.time() - s0) / num_sampled, num_try, num_repeat, num_more_answer,
                    num_broken, num_no_extra_answer, num_no_extra_negative, num_empty))
                old_num_sampled = num_sampled
        '''
        print(
            '%s %s: [%d/%d], avg time: %s, try: %s, repeat: %s: more_answer: %s, broken: %s, no extra: %s, no negative: %s empty: %s' % (
            mode,
            query_structure,
            num_sampled, gen_num, (time.time() - s0) / (num_sampled + 0.001), num_try, num_repeat, num_more_answer,
            num_broken, num_no_extra_answer, num_no_extra_negative, num_empty), end='\r')
        num_try += 1
        empty_query_structure = deepcopy(query_structure)
        '''
        answer_pool = set()
        for i in range(2, max(rel_ent_in_ent.keys()) + 1):
            answer_pool = answer_pool.union(rel_ent_in_ent[i].keys())
        answer = random.sample(answer_pool, 1)[0]
        '''
        answer = random.sample(ent_in.keys(), 1)[0]
        broken_flag = fill_query(empty_query_structure, ent_in, ent_out, answer, ent2id, rel2id, ent_in_ent, ent_out_ent, rel_ent_in_ent, rel_ent_out_ent)
        if broken_flag:
            num_broken += 1
            continue
        query = empty_query_structure
        answer_set = achieve_answer(query, ent_in, ent_out, rel_dr, ent_in_ent, ent_out_ent)
        small_answer_set = achieve_answer(query, small_ent_in, small_ent_out, small_rel_dr, small_ent_in_ent, small_ent_out_ent)
        if len(answer_set) == 0:
            num_empty += 1
            continue
        if mode != 'train':
            if len(answer_set - small_answer_set) == 0:
                num_no_extra_answer += 1
                continue
            if 'n' in query_name:
                if len(small_answer_set - answer_set) == 0:
                    num_no_extra_negative += 1
                    continue
        if max(len(answer_set - small_answer_set), len(small_answer_set - answer_set)) > max_ans_num:
            num_more_answer += 1
            continue
        if list2tuple(query) in queries[list2tuple(query_structure)]:
            num_repeat += 1
            continue
        queries[list2tuple(query_structure)].add(list2tuple(query))
        tp_answers[list2tuple(query)] = small_answer_set
        fp_answers[list2tuple(query)] = small_answer_set - answer_set
        fn_answers[list2tuple(query)] = answer_set - small_answer_set
        num_sampled += 1
        tp_ans_num.append(len(tp_answers[list2tuple(query)]))
        fp_ans_num.append(len(fp_answers[list2tuple(query)]))
        fn_ans_num.append(len(fn_answers[list2tuple(query)]))

    print()
    logging.info("{} tp max: {}, min: {}, mean: {}, std: {}".format(mode, np.max(tp_ans_num), np.min(tp_ans_num),
                                                                    np.mean(tp_ans_num), np.std(tp_ans_num)))
    logging.info("{} fp max: {}, min: {}, mean: {}, std: {}".format(mode, np.max(fp_ans_num), np.min(fp_ans_num),
                                                                    np.mean(fp_ans_num), np.std(fp_ans_num)))
    logging.info("{} fn max: {}, min: {}, mean: {}, std: {}".format(mode, np.max(fn_ans_num), np.min(fn_ans_num),
                                                                    np.mean(fn_ans_num), np.std(fn_ans_num)))

    name_to_save = '%s-%s' % (mode, query_name)
    with open('./data/%s/%s-queries.pkl' % (dataset, name_to_save), 'wb') as f:
        pickle.dump(queries, f)
    with open('./data/%s/%s-fp-answers.pkl' % (dataset, name_to_save), 'wb') as f:
        pickle.dump(fp_answers, f)
    with open('./data/%s/%s-fn-answers.pkl' % (dataset, name_to_save), 'wb') as f:
        pickle.dump(fn_answers, f)
    with open('./data/%s/%s-tp-answers.pkl' % (dataset, name_to_save), 'wb') as f:
        pickle.dump(tp_answers, f)
    return queries, tp_answers, fp_answers, fn_answers


def generate_queries(base_path, dataset, query_structures, gen_num, max_ans_num, gen_train, gen_valid, gen_test, query_names, save_name, idx):
    indexified_files = ['train.txt', 'valid.txt', 'test.txt']
    base_path = base_path + '/' + dataset
    if gen_train or gen_valid:
        train_ent_in, train_ent_out, train_rel_dr, train_ent_in_ent, train_ent_out_ent, train_rel_ent_in_ent, train_rel_ent_out_ent = construct_graph(base_path, indexified_files[:1])  # ent_in
    if gen_valid or gen_test:
        valid_ent_in, valid_ent_out, valid_rel_dr, valid_ent_in_ent, valid_ent_out_ent, valid_rel_ent_in_ent, valid_rel_ent_out_ent = construct_graph(base_path, indexified_files[:2])
        valid_only_ent_in, valid_only_ent_out, valid_only_rel_dr, valid_only_ent_in_ent, valid_only_ent_out_ent, valid_only_rel_ent_in_ent, valid_only_rel_ent_out_ent = construct_graph(base_path, indexified_files[1:2])
    if gen_test:
        test_ent_in, test_ent_out, test_rel_dr, test_ent_in_ent, test_ent_out_ent, test_rel_ent_in_ent, test_rel_ent_out_ent = construct_graph(base_path, indexified_files[:3])
        test_only_ent_in, test_only_ent_out, test_only_rel_dr, test_only_ent_in_ent, test_only_ent_out_ent, test_only_rel_ent_in_ent, test_only_rel_ent_out_ent = construct_graph(base_path, indexified_files[2:3])

    ent2id = pickle.load(open(os.path.join(base_path, "ent2id.pkl"), 'rb'))
    rel2id = pickle.load(open(os.path.join(base_path, "rel2id.pkl"), 'rb'))

    train_queries = defaultdict(set)
    train_tp_answers = defaultdict(set)
    train_fp_answers = defaultdict(set)
    train_fn_answers = defaultdict(set)
    valid_queries = defaultdict(set)
    valid_tp_answers = defaultdict(set)
    valid_fp_answers = defaultdict(set)
    valid_fn_answers = defaultdict(set)
    test_queries = defaultdict(set)
    test_answers = defaultdict(set)
    test_tp_answers = defaultdict(set)
    test_fp_answers = defaultdict(set)
    test_fn_answers = defaultdict(set)

    t1, t2, t3, t4, t5, t6 = 0, 0, 0, 0, 0, 0
    query_structure = query_structures[idx]
    query_name = query_names[idx] if save_name else str(idx)
    print('general structure is', query_structure, "with name", query_name)
    name_to_save = query_name
    set_logger("./data/{}/".format(dataset), name_to_save)

    num_sampled, num_try, num_repeat, num_more_answer, num_broken, num_empty = 0, 0, 0, 0, 0, 0
    train_ans_num = []
    s0 = time.time()
    if gen_train:
        train_queries, train_tp_answers, train_fp_answers, train_fn_answers = ground_queries(dataset, query_structure,
                                                                                             train_ent_in,
                                                                                             train_ent_out, defaultdict(lambda: defaultdict(set)), defaultdict(lambda: defaultdict(set)),
                                                                                             gen_num[0], max_ans_num,
                                                                                             query_name, 'train',
                                                                                             ent2id, rel2id, train_rel_dr, defaultdict(lambda: defaultdict(set)),
                                                                                             train_ent_in_ent, train_ent_out_ent, defaultdict(lambda: defaultdict(set)), defaultdict(lambda: defaultdict(set)),
                                                                                             train_rel_ent_in_ent, train_rel_ent_out_ent, defaultdict(lambda: defaultdict(set)), defaultdict(lambda: defaultdict(set)))
    if gen_valid:
        valid_queries, valid_tp_answers, valid_fp_answers, valid_fn_answers = ground_queries(dataset, query_structure,
                                                                                             valid_ent_in,
                                                                                             valid_ent_out,
                                                                                             train_ent_in,
                                                                                             train_ent_out, gen_num[1],
                                                                                             max_ans_num, query_name,
                                                                                             'valid', ent2id, rel2id, valid_rel_dr, train_rel_dr,
                                                                                             valid_ent_in_ent, valid_ent_out_ent, train_ent_in_ent, train_ent_out_ent,
                                                                                             valid_rel_ent_in_ent, valid_rel_ent_out_ent, train_rel_ent_in_ent, train_rel_ent_out_ent)
    if gen_test:
        test_queries, test_tp_answers, test_fp_answers, test_fn_answers = ground_queries(dataset, query_structure,
                                                                                         test_ent_in, test_ent_out,
                                                                                         valid_ent_in, valid_ent_out,
                                                                                         gen_num[2], max_ans_num,
                                                                                         query_name, 'test', ent2id,
                                                                                         rel2id, test_rel_dr, valid_rel_dr,
                                                                                         test_ent_in_ent, test_ent_out_ent, valid_ent_in_ent, valid_ent_out_ent,
                                                                                         test_rel_ent_in_ent, test_rel_ent_out_ent, valid_rel_ent_in_ent, valid_rel_ent_out_ent)
    print('%s queries generated with structure %s' % (gen_num, query_structure))


def fill_query(query_structure, ent_in, ent_out, answer, ent2id, rel2id,
               ent_in_ent, ent_out_ent, rel_ent_in_ent, rel_ent_out_ent):
    ## projections
    assert type(query_structure[-1]) == list
    all_relation_flag = True
    for ele in query_structure[-1]:
        if ele not in ['r', 'n']:
            all_relation_flag = False
            break
    if all_relation_flag:
        r = -1
        for i in range(len(query_structure[-1]))[::-1]:
            if query_structure[-1][i] == 'n':
                query_structure[-1][i] = -2
                continue
            found = False
            for j in range(40):
                r_tmp = random.sample(ent_in[answer].keys(), 1)[0]
                if r_tmp // 2 != r // 2 or r_tmp == r:
                    r = r_tmp
                    found = True
                    break
            if not found:
                return True
            query_structure[-1][i] = r
            #answer = random.sample(ent_in[answer][r], 1)[0]

            answer_pool = set()
            for i in range(2, max(rel_ent_in_ent.keys()) + 1):
                if rel_ent_in_ent.get(i) is not None:
                    if rel_ent_in_ent.get(i).get(answer) is not None:
                        answer_pool = answer_pool.union(rel_ent_in_ent[i][answer])
            if len(ent_in[answer][r].intersection(answer_pool)) != 0:
                answer = random.sample(ent_in[answer][r].intersection(answer_pool), 1)[0]
            else:
                return True

        if query_structure[0] == 'e':
            query_structure[0] = answer
        else:
            return fill_query(query_structure[0], ent_in, ent_out, answer, ent2id, rel2id, ent_in_ent, ent_out_ent, rel_ent_in_ent, rel_ent_out_ent)
    else:
        if len(query_structure[-1]) == 1 and query_structure[-1][0] == 'u':
            ## union
            query_structure[-1][0] = -1
            for i in range(len(query_structure)-1):
                broken_flag = fill_query(query_structure[i], ent_in, ent_out, answer, ent2id, rel2id, ent_in_ent, ent_out_ent, rel_ent_in_ent, rel_ent_out_ent)
                if broken_flag:
                    return True

        elif query_structure[-1][-1] == 's':
            ## split
            query_structure[-1][-1] = -3
            found = False
                #answer_tmp = random.sample(ent_in_ent[answer].keys(),1)[0]
            answer_pool = set()
            for i in range(len(query_structure[-1])-1, max(rel_ent_in_ent.keys())+1):
                if rel_ent_in_ent.get(i) is not None:
                    if rel_ent_in_ent.get(i).get(answer) is not None:
                        answer_pool = answer_pool.union(rel_ent_in_ent.get(i).get(answer))
            if len(answer_pool) >= 1:
                answer_tmp = random.sample(answer_pool, 1)[0]
                r_tmps = random.sample(ent_in_ent[answer][answer_tmp], len(query_structure[-1])-1) ## give it an order
                r_tmps.sort()
                answer = answer_tmp
                for i in range(len(query_structure[-1]) - 1):
                    if query_structure[-1][i] not in ['r']:
                        ## [r,n]
                        assert len(query_structure[-1][i]) == 2
                        query_structure[-1][i][-1] = -2
                        query_structure[-1][i][0] = r_tmps[i]
                    else:
                        query_structure[-1][i] = r_tmps[i]
                found = True
            '''
            overlap = set(ent_out.keys())
            for j in range(40):
                if len(ent_in[answer].keys()) >= len(query_structure[-1])-1:
                    r_tmps = random.sample(ent_in[answer].keys(), len(query_structure[-1])-1)
                    for x in r_tmps:
                        overlap = overlap.intersection(ent_in[answer][x])
                    if len(overlap) != 0:
                        for i in range(len(query_structure[-1]) - 1):
                            if query_structure[-1][i] not in ['r']:
                                ## [r,n]
                                assert len(query_structure[-1][i]) == 2
   0                             query_structure[-1][i][-1] = -2
                                query_structure[-1][i][0] = r_tmps[i]
                            else:
                                query_structure[-1][i] = r_tmps[i]
                        found = True
                        break
            '''
            if not found:
                return True
            #answer = random.sample(overlap, 1)[0]
            if query_structure[0] == 'e':
                query_structure[0] = answer
            else:
                return fill_query(query_structure[0], ent_in, ent_out, answer, ent2id, rel2id, ent_in_ent, ent_out_ent, rel_ent_in_ent, rel_ent_out_ent)

        else: ## intersection query
            for i in range(len(query_structure)):
                broken_flag = fill_query(query_structure[i], ent_in, ent_out, answer, ent2id, rel2id, ent_in_ent, ent_out_ent, rel_ent_in_ent, rel_ent_out_ent)
                if broken_flag:
                    return True

        #same_structure = defaultdict(list)
        #for i in range(len(query_structure)):
        #    same_structure[list2tuple(query_structure[i])].append(i)
        ## union query
        '''
        for i in range(len(query_structure)):
            if len(query_structure[i]) == 1 and query_structure[i][0] == 'u':
                assert i == len(query_structure) - 1
                query_structure[i][0] = -1
                continue
            broken_flag = fill_query(query_structure[i], ent_in, ent_out, answer, ent2id, rel2id)
            if broken_flag:
                return True

        for structure in same_structure:
            if len(same_structure[structure]) != 1:
                structure_set = set()
                for i in same_structure[structure]:
                    structure_set.add(list2tuple(query_structure[i]))
                if len(structure_set) < len(same_structure[structure]):
                    return True
        '''

'''
def achieve_answer(query, ent_in, ent_out, rel_dr, ent_in_ent, ent_out_ent):
    assert type(query[-1]) == list
    all_relation_flag = True
    for ele in query[-1]:
        if (type(ele) != int) or (ele in [-1, -3]):
            all_relation_flag = False
            break
    ## projections
    if all_relation_flag:
        if type(query[0]) == int:
            ent_set = set([query[0]])
        else:
            ent_set = achieve_answer(query[0], ent_in, ent_out, rel_dr, ent_in_ent, ent_out_ent)
        for i in range(len(query[-1])):
            if query[-1][i] == -2:
                ent_set = set(range(len(ent_in))) - ent_set
            else:
                ent_set_traverse = set()
                for ent in ent_set:
                    ent_set_traverse = ent_set_traverse.union(ent_out[ent][query[-1][i]])
                ent_set = ent_set_traverse
    else:
        ent_set = achieve_answer(query[0], ent_in, ent_out, rel_dr, ent_in_ent, ent_out_ent)
        ## split
        # find real intermediate ent_set
        if query[-1][-1] == -3:
            real_ent_set = set(ent_out.keys())
            for rel in query[-1][:-1]:
                if type(rel) == int:
                    real_ent_set = real_ent_set.intersection(rel_dr[rel]['dom'])
                else:
                    assert len(rel) == 2
                    real_ent_set = real_ent_set.intersection(rel_dr[rel[0]]['dom'])
            #assert len(real_ent_set) != 0
            ent_set_traverse = set(ent_out.keys())
            for ent in ent_set:
                for rel in query[-1][:-1]:
                    if type(rel) == int:
                        if ent_out.get(ent).get(rel) is not None:
                            ent_set_traverse = ent_set_traverse.intersection(ent_out[ent][rel])
                        else:
                            ent_set_traverse = set()
                    else:
                        assert rel[1] == -2
                        if ent_out.get(ent).get(rel[0]) is not None:
                            ent_set_traverse = ent_set_traverse.intersection(set(ent_out.keys())-ent_out[ent][rel[0]])
                        else:
                            ent_set_traverse = ent_set_traverse.intersection(set(ent_out.keys()))
            ent_set = real_ent_set.intersection(ent_set_traverse)

        ## union
        elif len(query[-1]) == 1 and query[-1][0] == -1:
            for i in range(1, len(query)-1):
                ent_set = ent_set.union(achieve_answer(query[i], ent_in, ent_out, rel_dr, ent_in_ent, ent_out_ent))

        else:
            for i in range(1, len(query)):
                ent_set = ent_set.intersection(achieve_answer(query[i], ent_in, ent_out, rel_dr, ent_in_ent, ent_out_ent))

    return ent_set
'''
def achieve_answer(query, ent_in, ent_out, rel_dr, ent_in_ent, ent_out_ent):
    #assert type(query[-1]) == list
    all_relation_flag = True
    for ele in query[-1]:
        if (type(ele) != int) or (ele in [-1, -3]):
            all_relation_flag = False
            break
    ## projections
    if all_relation_flag:
        if type(query[0]) == int:
            ent_set = set([query[0]])
        else:
            ent_set = achieve_answer(query[0], ent_in, ent_out, rel_dr, ent_in_ent, ent_out_ent)
        for i in range(len(query[-1])):
            if query[-1][i] == -2:
                ent_set = set(range(len(ent_in))) - ent_set
            else:
                ent_set_traverse = set()
                for ent in ent_set:
                    ent_set_traverse = ent_set_traverse.union(ent_out[ent][query[-1][i]])
                ent_set = ent_set_traverse
    else:
        if type(query[0]) == list:
            ent_set = achieve_answer(query[0], ent_in, ent_out, rel_dr, ent_in_ent, ent_out_ent)
        else:
            ent_set = set([query[0]])
        ## split
        # find real intermediate ent_set
        if query[-1][-1] == -3:
            real_ent_set = set(ent_out.keys())
            for rel in query[-1][:-1]:
                if type(rel) == int:
                    real_ent_set = real_ent_set.intersection(rel_dr[rel]['dom'])
                else:
                    assert len(rel) == 2
                    ## negation no dom
                    #real_ent_set = real_ent_set.intersection(rel_dr[rel[0]]['dom'])
            ent_set = ent_set.intersection(real_ent_set)
            #assert len(real_ent_set) != 0
            ent_set_traverse = set(ent_out.keys()) if len(ent_set) != 0 else ent_set
            for ent in ent_set:
                for rel in query[-1][:-1]:
                    if type(rel) == int:
                        if ent_out.get(ent).get(rel) is not None:
                            ent_set_traverse = ent_set_traverse.intersection(ent_out[ent][rel])
                        else:
                            ent_set_traverse = set()
                    else:
                        assert rel[1] == -2
                        if ent_out.get(ent).get(rel[0]) is not None:
                            ent_set_traverse = ent_set_traverse.intersection(set(ent_out.keys())-ent_out[ent][rel[0]])
                        else:
                            ent_set_traverse = ent_set_traverse.intersection(set(ent_out.keys()))
            ent_set = ent_set_traverse
            #ent_set = real_ent_set.intersection(ent_set_traverse)

        ## union
        elif len(query[-1]) == 1 and query[-1][0] == -1:
            for i in range(1, len(query)-1):
                ent_set = ent_set.union(achieve_answer(query[i], ent_in, ent_out, rel_dr, ent_in_ent, ent_out_ent))

        else:
            for i in range(1, len(query)):
                ent_set = ent_set.intersection(achieve_answer(query[i], ent_in, ent_out, rel_dr, ent_in_ent, ent_out_ent))

    return ent_set

def main(args):
    train_num_dict = {'FB15k': 273710, "FB15k-237": 149689, "NELL": 107982}
    valid_num_dict = {'FB15k': 8000, "FB15k-237": 5000, "NELL": 4000}
    test_num_dict = {'FB15k': 8000, "FB15k-237": 5000, "NELL": 4000}
    if args.gen_train and args.gen_train_num == 0:
        if 'FB15k-237' in args.dataset:
            args.gen_train_num = 149689
        elif 'FB15k' in args.dataset:
            args.gen_train_num = 273710
        elif 'NELL' in args.dataset:
            args.gen_train_num = 10000
        else:
            args.gen_train_num = train_num_dict[args.dataset]
    if args.gen_valid and args.gen_valid_num == 0:
        if 'FB15k-237' in args.dataset:
            args.gen_valid_num = 5000
        elif 'FB15k' in args.dataset:
            args.gen_valid_num = 8000
        elif 'NELL' in args.dataset:
            args.gen_valid_num = 1000
        else:
            args.gen_valid_num = valid_num_dict[args.dataset]
    if args.gen_test and args.gen_test_num == 0:
        if 'FB15k-237' in args.dataset:
            args.gen_test_num = 5000
        elif 'FB15k' in args.dataset:
            args.gen_test_num = 8000
        elif 'NELL' in args.dataset:
            args.gen_test_num = 1000
        else:
            args.gen_test_num = test_num_dict[args.dataset]
    if args.index_only:
        index_dataset(args.base_path, args.dataset, args.reindex)
        exit(-1)

    e = 'e'
    r = 'r'
    n = 'n'
    u = 'u'
    s = 's'

    query_structures = [
        ## 2s
        [[e,[r]],[r,r,s]],
        ## 3s
        [[e,[r]],[r,r,r,s]],
        ## sp
        [[[e,[r]],[r,r,s]],[r]],
        ## us
        [[[e,[r]],[e,[r]],[u]],[r,r,s]],
        ## is
        [[[e,[r]],[e,[r]]],[r,r,s]],
        ## ins
        [[[e,[r,n]],[e,[r]]],[r,r,s]],
        ## 2rs
        [e, [r, r, s]],
        ## 3rs
        [e, [r, r, r, s]]
    ]

    query_names = ['2s', '3s', 'sp', 'us', 'is', 'ins', '2rs', '3rs']
    gen_id_strings = args.gen_id.split('.')

    for gen_id in gen_id_strings:
        generate_queries(args.base_path, args.dataset, query_structures,
                         [args.gen_train_num, args.gen_valid_num, args.gen_test_num], args.max_ans_num, args.gen_train,
                         args.gen_valid, args.gen_test, query_names[int(gen_id):int(gen_id) + 1], args.save_name, int(gen_id))

if __name__ == '__main__':
    main(parse_args())