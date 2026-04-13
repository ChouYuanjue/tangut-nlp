from abc import ABC, abstractmethod
from typing import Optional


class Constraint(ABC):
    def __init__(self):
        self.test()

    def test(self):
        counter = 0
        completed = False
        while not completed:
            if counter == 1:
                self.reset()
            advance = self.advance()
            if not self.does_advance(advance):
                raise Exception(
                    "Custom Constraint is not defined correctly. self.does_advance(self.advance()) must be true."
                )

            stepped, completed, reset = self.update(advance)
            counter += 1

            if counter > 10000:
                raise Exception("update() does not fulfill the constraint.")

        if self.remaining() != 0:
            raise Exception("Custom Constraint is not defined correctly.")

    @abstractmethod
    def advance(self):
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    @abstractmethod
    def does_advance(self, token_id: int):
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    @abstractmethod
    def update(self, token_id: int):
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    @abstractmethod
    def reset(self):
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    @abstractmethod
    def remaining(self):
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    @abstractmethod
    def copy(self, stateful=False):
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


class PhrasalConstraint(Constraint):
    def __init__(self, token_ids: list[int]):
        super(Constraint, self).__init__()

        if not isinstance(token_ids, list) or len(token_ids) == 0:
            raise ValueError(f"`token_ids` has to be a non-empty list, but is {token_ids}.")
        if any((not isinstance(token_id, int) or token_id < 0) for token_id in token_ids):
            raise ValueError(f"Each list in `token_ids` has to be a list of positive integers, but is {token_ids}.")

        self.token_ids = token_ids
        self.seqlen = len(self.token_ids)
        self.fulfilled_idx = -1
        self.completed = False

    def advance(self):
        if self.completed:
            return None
        return self.token_ids[self.fulfilled_idx + 1]

    def does_advance(self, token_id: int):
        if not isinstance(token_id, int):
            raise TypeError(f"`token_id` has to be an `int`, but is {token_id} of type {type(token_id)}")
        if self.completed:
            return False
        return token_id == self.token_ids[self.fulfilled_idx + 1]

    def update(self, token_id: int):
        if not isinstance(token_id, int):
            raise TypeError(f"`token_id` has to be an `int`, but is {token_id} of type {type(token_id)}")

        stepped = False
        completed = False
        reset = False

        if self.does_advance(token_id):
            self.fulfilled_idx += 1
            stepped = True
            if self.fulfilled_idx == (self.seqlen - 1):
                completed = True
            self.completed = completed
        else:
            reset = True
            self.reset()
        return stepped, completed, reset

    def reset(self):
        self.completed = False
        self.fulfilled_idx = 0

    def remaining(self):
        return self.seqlen - (self.fulfilled_idx + 1)

    def copy(self, stateful=False):
        new_constraint = PhrasalConstraint(self.token_ids)
        if stateful:
            new_constraint.seq_len = self.seqlen
            new_constraint.fulfilled_idx = self.fulfilled_idx
            new_constraint.completed = self.completed
        return new_constraint


class DisjunctiveTrie:
    def __init__(self, nested_token_ids: list[list[int]], no_subsets=True):
        self.max_height = max([len(one) for one in nested_token_ids])

        root = {}
        for token_ids in nested_token_ids:
            level = root
            for token_id in token_ids:
                if token_id not in level:
                    level[token_id] = {}
                level = level[token_id]

        if no_subsets and self.has_subsets(root, nested_token_ids):
            raise ValueError(
                "Each list in `nested_token_ids` can't be a complete subset of another list, but is"
                f" {nested_token_ids}."
            )

        self.trie = root

    def next_tokens(self, current_seq):
        start = self.trie
        for current_token in current_seq:
            start = start[current_token]
        return list(start.keys())

    def reached_leaf(self, current_seq):
        next_tokens = self.next_tokens(current_seq)
        return len(next_tokens) == 0

    def count_leaves(self, root):
        next_nodes = list(root.values())
        if len(next_nodes) == 0:
            return 1
        return sum([self.count_leaves(node) for node in next_nodes])

    def has_subsets(self, trie, nested_token_ids):
        leaf_count = self.count_leaves(trie)
        return len(nested_token_ids) != leaf_count


class DisjunctiveConstraint(Constraint):
    def __init__(self, nested_token_ids: list[list[int]]):
        super(Constraint, self).__init__()

        if not isinstance(nested_token_ids, list) or len(nested_token_ids) == 0:
            raise ValueError(f"`nested_token_ids` has to be a non-empty list, but is {nested_token_ids}.")
        if any(not isinstance(token_ids, list) for token_ids in nested_token_ids):
            raise ValueError(f"`nested_token_ids` has to be a list of lists, but is {nested_token_ids}.")
        if any(
            any((not isinstance(token_id, int) or token_id < 0) for token_id in token_ids)
            for token_ids in nested_token_ids
        ):
            raise ValueError(
                f"Each list in `nested_token_ids` has to be a list of positive integers, but is {nested_token_ids}."
            )

        self.trie = DisjunctiveTrie(nested_token_ids)
        self.token_ids = nested_token_ids
        self.seqlen = self.trie.max_height
        self.current_seq = []
        self.completed = False

    def advance(self):
        token_list = self.trie.next_tokens(self.current_seq)
        if len(token_list) == 0:
            return None
        return token_list

    def does_advance(self, token_id: int):
        if not isinstance(token_id, int):
            raise TypeError(f"`token_id` is supposed to be type `int`, but is {token_id} of type {type(token_id)}")
        next_tokens = self.trie.next_tokens(self.current_seq)
        return token_id in next_tokens

    def update(self, token_id: int):
        if not isinstance(token_id, int):
            raise TypeError(f"`token_id` is supposed to be type `int`, but is {token_id} of type {type(token_id)}")

        stepped = False
        completed = False
        reset = False

        if self.does_advance(token_id):
            self.current_seq.append(token_id)
            stepped = True
        else:
            reset = True
            self.reset()

        completed = self.trie.reached_leaf(self.current_seq)
        self.completed = completed
        return stepped, completed, reset

    def reset(self):
        self.completed = False
        self.current_seq = []

    def remaining(self):
        if self.completed:
            return 0
        return self.seqlen - len(self.current_seq)

    def copy(self, stateful=False):
        new_constraint = DisjunctiveConstraint(self.token_ids)
        if stateful:
            new_constraint.seq_len = self.seqlen
            new_constraint.current_seq = self.current_seq
            new_constraint.completed = self.completed
        return new_constraint


class ConstraintListState:
    def __init__(self, constraints: list[Constraint]):
        self.constraints = constraints
        self.max_seqlen = max([c.seqlen for c in constraints])
        self.n_constraints = len(constraints)
        self.completed = False
        self.init_state()

    def init_state(self):
        self.complete_constraints = []
        self.inprogress_constraint = None
        self.pending_constraints = [constraint.copy(stateful=False) for constraint in self.constraints]

    def get_bank(self):
        add = 0
        if self.inprogress_constraint:
            add += self.max_seqlen - self.inprogress_constraint.remaining()
        return (len(self.complete_constraints) * self.max_seqlen) + add

    def advance(self):
        token_list = []
        if self.inprogress_constraint is None:
            for constraint in self.pending_constraints:
                advance = constraint.advance()
                if isinstance(advance, int):
                    token_list.append(advance)
                elif isinstance(advance, list):
                    token_list.extend(advance)
        else:
            advance = self.inprogress_constraint.advance()
            if isinstance(advance, int):
                token_list.append(advance)
            elif isinstance(advance, list):
                token_list.extend(advance)

        if len(token_list) == 0:
            return None
        return token_list

    def reset(self, token_ids: Optional[list[int]]):
        self.init_state()
        if token_ids is not None:
            for token in token_ids:
                complete, stepped = self.add(token)
                if self.completed:
                    break

    def add(self, token_id: int):
        if not isinstance(token_id, int):
            raise TypeError(f"`token_id` should be an `int`, but is `{token_id}`.")

        complete, stepped = False, False

        if self.completed:
            complete = True
            stepped = False
            return complete, stepped

        if self.inprogress_constraint is not None:
            stepped, complete, reset = self.inprogress_constraint.update(token_id)
            if reset:
                self.pending_constraints.append(self.inprogress_constraint.copy(stateful=False))
                self.inprogress_constraint = None

            if complete:
                self.complete_constraints.append(self.inprogress_constraint)
                self.inprogress_constraint = None
                if len(self.pending_constraints) == 0:
                    self.completed = True
        else:
            for cidx, pending_constraint in enumerate(self.pending_constraints):
                if pending_constraint.does_advance(token_id):
                    stepped, complete, reset = pending_constraint.update(token_id)

                    if not stepped:
                        raise Exception(
                            "`constraint.update(token_id)` is not yielding incremental progress, "
                            "even though `constraint.does_advance(token_id)` is true."
                        )

                    if complete:
                        self.complete_constraints.append(pending_constraint)
                        self.inprogress_constraint = None

                    if not complete and stepped:
                        self.inprogress_constraint = pending_constraint

                    if complete or stepped:
                        self.pending_constraints = (
                            self.pending_constraints[:cidx] + self.pending_constraints[cidx + 1 :]
                        )

                        if len(self.pending_constraints) == 0 and self.inprogress_constraint is None:
                            self.completed = True
                        break

        return complete, stepped

    def copy(self, stateful=True):
        new_state = ConstraintListState(self.constraints)
        if stateful:
            new_state.complete_constraints = [
                constraint.copy(stateful=True) for constraint in self.complete_constraints
            ]
            if self.inprogress_constraint is not None:
                new_state.inprogress_constraint = self.inprogress_constraint.copy(stateful=True)
            new_state.pending_constraints = [constraint.copy() for constraint in self.pending_constraints]
        return new_state
