# infinite_simples

For now it requires permuta and automata-lib (modified to support union and intersection)

The DFA class must additionally have these functions implemented:

```python
    def _cross_product(self, other):
        """
        Creates a new DFA which is the cross product of DFAs self and other
        with an empty set of final states.
        """
        assert self.input_symbols == other.input_symbols
        states_a = list(self.states)
        states_b = list(other.states)
        new_states = {self._stringify_states_unsorted((a, b)) for a in states_a for b in states_b}
        new_transitions = dict()
        for state_a, transitions_a in self.transitions.items():
            for state_b, transitions_b in other.transitions.items():
                new_state = self._stringify_states_unsorted((state_a, state_b))
                new_transitions[new_state] = dict()
                for symbol in self.input_symbols:
                    new_transitions[new_state][symbol] = self._stringify_states_unsorted((transitions_a[symbol], transitions_b[symbol]))
        new_initial_state = self._stringify_states_unsorted((self.initial_state, other.initial_state))

        return DFA(
                states=new_states,
                input_symbols=self.input_symbols,
                transitions=new_transitions,
                initial_state=new_initial_state,
                final_states=set()
                )


    def union(self, other):
        new_dfa = self._cross_product(other)
        for state_a in self.states:
            for state_b in other.states:
                if state_a in self.final_states or state_b in other.final_states:
                    new_dfa.final_states.add(self._stringify_states_unsorted((state_a, state_b)))
        new_dfa.validate()
        return new_dfa

    def intersect(self, other):
        new_dfa = self._cross_product(other)
        for state_a in self.final_states:
            for state_b in other.final_states:
                new_dfa.final_states.add(self._stringify_states_unsorted((state_a, state_b)))
        return new_dfa

    def complement(self):
        new_dfa = self.copy()
        new_dfa.final_states = self.states - self.final_states
        return new_dfa

    @staticmethod
    def _stringify_states_unsorted(states):
        """Stringify the given set of states as a single state name."""
        return '{{{}}}'.format(','.join(states))
```
