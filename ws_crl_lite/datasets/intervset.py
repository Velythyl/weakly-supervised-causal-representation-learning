import dataclasses
import numpy as np
import torch

from itertools import chain, combinations

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


@dataclasses.dataclass
class Interv:
    self: torch.Tensor
    history: torch.Tensor

    def march_history(self, markov):
        ret = torch.hstack((self.history, self.self[:,None]))
        if ret.shape[1] > markov:
            ret = ret[:,-markov:]
        return ret


def to_tensor(arr):
    if isinstance(arr, torch.Tensor):
        return arr.int()
    assert isinstance(arr, np.ndarray)
    return torch.from_numpy(arr).int()

def to_np(arr):
    if isinstance(arr, np.ndarray):
        return arr.astype(int)
    assert isinstance(arr, torch.Tensor)
    return arr.detach().cpu().numpy().astype(int)

def dict_to_tensor(dico):
    temp = {}
    for k, c in dico.items():
        temp[k] = to_tensor(c)
    return temp


class _IntervDist:
    def __init__(self):
        self.interv_ids = None

    def set_interv_ids(self, interv_ids):
        self.interv_ids = interv_ids
        return self

    def __call__(self, history):
        if self.interv_ids is None:
            raise AssertionError("You should set the interv_ids in the IntervSet's constructor. You done goofed.")

        if isinstance(history, int) or ( isinstance(history, float) and int(history) == history):
            history = np.array([[]]*history)
        BATCH_SIZE = history.shape[0]
        HISTORY_LENGTH = history.shape[1]

        return self.call(BATCH_SIZE, HISTORY_LENGTH, to_np(history))

    def call(self, BATCH_SIZE, HISTORY_LENGTH, history):
        raise NotImplemented()


class Uniform(_IntervDist):
    def __init__(self, no_replace):
        super().__init__()
        self.no_replace = no_replace

    def call(self, BATCH_SIZE, HISTORY_LENGTH, history):
        DO_REPLACE = not self.no_replace
        if HISTORY_LENGTH == 0 or DO_REPLACE:    # if we ARE replacing, it means that we can pick an intervention more than once
            return np.random.choice(self.interv_ids, size=BATCH_SIZE, replace=True)

        # can't pick an intervention more than once (?)
        ret = []
        for i in range(BATCH_SIZE):
            h = set(history[i].detach().cpu().numpy().tolist())
            leftovers = list(set(self.interv_ids) - h)

            subret = np.random.choice(leftovers, size=(1,), replace=False)  # Note: the replace=False has nothing to do with self.no_replace
            ret.append(subret)

        ret = np.hstack(ret)
        return ret

class Table(_IntervDist):

    def __init__(self, dict_of_tables, dict_of_alphas):
        super().__init__()
        self.dict_of_tables = dict_of_tables
        # {
        #   0: (num_nodes,)  # chosen at timestep 0 (ONLY HAPPENS ONCE)
        #   1: (num_nodes, num_nodes) # chosen at order 1
        #   2: (num_nodes, num_nodes) # chosen at order 2
        #   ...
        # }
        self.dict_of_alphas = dict_of_alphas
        # {
        #   0: 1    # never used
        #   1: 1    # used, but useless (tragic)
        #   2: (2,) # [0] * dict_of_tables [1] + [1] * dict_of_tables[2]
        #   3: (3,) # [0] * dict_of_tables [1] + [1] * dict_of_tables[2] + [2] * dict_of_tables[3]
        # }

    @property
    def num_interv_ids(self):
        return self.dict_of_tables[0].shape[0]

    def call(self, BATCH_SIZE, HISTORY_LENGTH, history):
        def maybe_squeeze(arr):
            if len(arr.shape) == 2:
                return arr.squeeze()
            else:
                return arr

        def weighted_summon(weights):
            if len(weights.shape) == 1:
                sum = weights.sum()
                weights = weights / sum
                return np.random.choice(self.interv_ids, size=(BATCH_SIZE,), replace=True, p=weights)

            sum = weights.sum(axis=1)
            sum = np.broadcast_to(sum[:,None], weights.shape)
            weights = weights / sum

            ret = []
            for i in range(BATCH_SIZE):
                ret.append( np.random.choice(self.interv_ids, size=(1,), replace=True, p=weights[i]) )
            ret = np.vstack(ret)
            return ret

        if HISTORY_LENGTH == 0:
            weights = self.dict_of_tables[0]
            return maybe_squeeze(weighted_summon(weights))

        def resolve_weights():
            final_weights = np.zeros((BATCH_SIZE, self.num_interv_ids))
            for i in range(HISTORY_LENGTH):
                weights_for_past = []
                current_alpha = self.dict_of_alphas[HISTORY_LENGTH][i]
                for l in range(BATCH_SIZE):
                    past_node = history[l,i]
                    weights_for_past.append(self.dict_of_tables[i+1][past_node] * current_alpha)
                weights_for_past = np.vstack(weights_for_past)
                final_weights += weights_for_past
            return final_weights
        return maybe_squeeze(weighted_summon(resolve_weights()))

class IntervSet:
    def __init__(self, adj_mat, markov=0):
        self.markov = markov

        self.num_nodes = adj_mat.shape[0]
        self.adj_mat = adj_mat

        self.set_of_all_intervs = list(sorted(list(powerset(list(range(self.num_nodes))))))


        self.interv_ids = np.arange(len(self.set_of_all_intervs))
        self.switch_case = None

    def id2interv(self, id):
        return self.set_of_all_intervs[id]

    @property
    def num_interv_ids(self):
        return self.interv_ids.shape[0]

    def set_switch_case(self, switch_case):
        if switch_case is None:
            self.switch_case = {
                0: Uniform(no_replace=False).set_interv_ids(self.interv_ids)
            }
        else:
            self.switch_case = switch_case
        temp = {}
        for k, c in self.switch_case.items():
            c = c.set_interv_ids(self.interv_ids)
            temp[k] = lambda h: to_tensor(c(h))
        self.switch_case = temp

    def init(self, batch_size):
        if self.switch_case is None:
            raise Exception("how dare you")

        return Interv(self.switch_case[0](batch_size), to_tensor(np.array([[]]*batch_size)))

    def pick(self, past_interv):
        if self.switch_case is None:
            raise Exception("how dare you")

        history = past_interv.march_history(self.markov)
        h_len = history.shape[1]

        if len(self.switch_case) == 1:
            return Interv(self.switch_case[0](history), history)

        return Interv(self.switch_case[h_len](history), history)



if __name__ == "__main__":

    import networkx as nx
    G = nx.DiGraph()

    # Add edges to the graph based on the structure you provided
    edges = [('A', 'B'), ('B', 'C'), ('A', 'C')]
    G.add_edges_from(edges)

    adj_mat = nx.adjacency_matrix(G)
    # Convert the adjacency matrix to a NumPy array (if needed)
    adj_mat = adj_mat.toarray()


    x = IntervSet(adj_mat, 2)

    dict_of_tables = {
        0: np.ones(x.num_interv_ids),
        1: np.random.uniform(0, 10, size=(x.num_interv_ids, x.num_interv_ids)),
        2: np.random.uniform(0, 10, size=(x.num_interv_ids, x.num_interv_ids))
    }
    dict_of_alphas = {
        0: [1],
        1: [1],
        2: [0.5, 1]
    }

    switch_case = {
        0: Uniform(no_replace=True),
        1: Table(dict_of_tables, dict_of_alphas),
        2: Table(dict_of_tables, dict_of_alphas)
    }

    x.set_switch_case(switch_case)

    t0 = x.init(100)
    t1 = x.pick(t0)
    t2 = x.pick(t1)
    t3 = x.pick(t2)

