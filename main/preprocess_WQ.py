#!/usr/bin/env python3
import argparse
from pathlib import Path
import pickle


def fix_none(datapoint):
    """
    This function fixes the graphs containing none entities by
    merging triples that contain none entities together;
    Triples missing a head will be merged with triples missing a tail and vice versa.
    """
    # split graph in full triples, triples missing head and missing tail.
    miss_tail = []
    miss_head = []
    okay = []
    for trip in datapoint:
        if trip[0] == 'none' and trip[2] == 'none':
            pass
        elif trip[2] == 'none':
            miss_tail.append(trip)
        elif trip[0] == 'none':
            miss_head.append(trip)
        else:
            okay.append(trip)

    # paste the the missing heads and missing tails to other missing triples
    new_triple = []

    if len(miss_tail) != 0 and len(miss_head) == 0:
        pass
    elif len(miss_tail) == 0 and len(miss_head) != 0:
        pass
    elif len(miss_tail) == 0 and len(miss_head) == 0:
        pass

    elif len(miss_tail) < len(miss_head):
        for i in range(len(miss_head)):
            new = miss_head[i][1:]
            try:
                new.insert(0, miss_tail[i][0])
            except IndexError:
                new.insert(0, miss_tail[-1][0])
            new_triple.append(new)

    elif (len(miss_tail) > len(miss_head)) or (len(miss_tail) == len(miss_head)):
        for i in range(len(miss_tail)):
            new = miss_tail[i][0:2]
            try:
                new.append(miss_head[i][2])
            except IndexError:
                new.append(miss_head[-1][2])
            new_triple.append(new)

    new_triple.extend(okay)

    return new_triple


def fix_rel(rel):
    """
    Convert the naming of the relations to natural language
    """
    words = rel.split('/')[-1].split('_')
    if words[-1] == 's':
        words = words[:-1]
    return ' '.join(words)


def linearize_graphs(dataset, with_answer=False, answer_list=None):
    """
    Iterate through the triples in the graphs and linearize in
    the following way: <H> {head entity} <R> {relation} <T> {tail entity}
    """
    graphs = []
    for i, graph in enumerate(dataset):
        lin_graph = ''
        if with_answer:
            lin_graph += f'<A> {answer_list[i]} '
        for triple in graph:
            head = triple[0]
            relation = fix_rel(triple[1])
            tail = triple[2]
            lin_graph += f'<H> {head} <R> {relation} <T> {tail} '
        graphs.append(lin_graph.strip())
    return graphs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--with_answer", help="Add answers to source graphs.",
                        action="store_true")
    args = parser.parse_args()

    # load data
    data = pickle.load(open('data/raw_data/WQ/subgraph_nl.pkl', 'rb'))

    # fix the none entities
    new_data = []
    for point in data:
        new_data.append(fix_none(point))

    # linearize the graphs
    if args.with_answer:
        # load list of answers
        with open('data/raw_data/WQ/answer_list.txt', 'r', encoding='utf-8') as f:
            answer_file = f.read().splitlines()
        answers = [' '.join(e.split(' | ')) for e in answer_file]

        lin_graphs = linearize_graphs(new_data, args.with_answer, answers)

    else:
        lin_graphs = linearize_graphs(new_data)

    # write files, 18989 2000 2000 split
    path = Path.cwd() / 'data' / 'WQ'
    path.mkdir(parents=True, exist_ok=True)
    dir_ = 'data/WQ/'
    a = ''
    if args.with_answer:
        a = '_a'

    with open(f'{dir_}train{a}_src.txt', "w", encoding="utf-8") as outfile:
        outfile.write("\n".join(lin_graphs[:18989]))

    with open(f'{dir_}test{a}_src.txt', "w", encoding="utf-8") as outfile:
        outfile.write("\n".join(lin_graphs[18989:20989]))

    with open(f'{dir_}dev{a}_src.txt', "w", encoding="utf-8") as outfile:
        outfile.write("\n".join(lin_graphs[20989:]))
