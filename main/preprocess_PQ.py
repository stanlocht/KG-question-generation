#! /usr/bin/env python3
import argparse
import json
from pathlib import Path


def add_spaces(string):
    return ' '.join(string.split('_'))


def linearize_graphs(graphs, with_answer=False):
    """
    Iterate through the triples in the graphs and linearize in
    the following way: <H> {head entity} <R> {relation} <T> {tail entity}
    """
    lin_graphs = []
    for graph in graphs:
        linearized_graph = ''
        if with_answer:
            answer = graph['answers'][0]
            linearized_graph += f'<A> {answer} '
        for head in graph['inGraph']['g_adj']:
            head_ = add_spaces(head)
            for tail in graph['inGraph']['g_adj'][head]:
                tail_ = add_spaces(tail)
                for relation in graph['inGraph']['g_adj'][head][tail]:
                    relation_ = add_spaces(relation.split('/')[-1])

                    # add triple to graph
                    triple = f'<H> {head_} <R> {relation_} <T> {tail_} '
                    linearized_graph += triple

        lin_graphs.append(linearized_graph.strip())

    return lin_graphs


def convert_file(datafile, save_path, with_answer):
    # load in file
    graphs = []
    for line in open(datafile, 'r'):
        graphs.append(json.loads(line))

    # linearize the graphs
    lin_graphs = linearize_graphs(graphs, with_answer)

    # write file
    with open(save_path, 'w', encoding="utf-8") as f:
        f.write('\n'.join(lin_graphs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--with_answer", help="Add answers to source graphs.",
                        action="store_true")
    args = parser.parse_args()

    a = ''
    if args.with_answer:
        a = 'a_'

    splits = ['train', 'dev', 'test']

    for split in splits:
        file = f'data/raw_data/PQ/{split}.json'
        path = Path.cwd() / 'data' / 'PQ'
        path.mkdir(parents=True, exist_ok=True)

        convert_file(file, f'data/PQ/{split}_{a}src.txt', args.with_answer)
