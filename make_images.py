from itertools import combinations
from pinperms import *
from pathlib import Path


if __name__ == "__main__":

    n = int(sys.argv[1])
    k = int(sys.argv[2])
    for basis in combinations(Perm.of_length(n), k):
        basis_str = '_'.join(''.join(str(x) for x in p) for p in basis)
        L = make_DFA_for_basis(basis, use_db=True, verbose=False)
        L = L.complement()
        L = make_DFA_for_M().intersect(L).minify(False)
        directory = "images/S{}/length{}".format(n, k)
        p = Path(directory)
        p.mkdir(parents=True, exist_ok=True)
        filename = "{}.png".format(basis_str)
        p = p / filename
        L.show_diagram(p)
    
