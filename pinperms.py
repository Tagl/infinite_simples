from bisect import bisect_left
from collections import defaultdict
from fractions import Fraction
from functools import lru_cache
from permuta import *

from typing import (
    TYPE_CHECKING,
    Callable,
    ClassVar,
    Deque,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

DIRS = "ULDR"
QUADS = "1234"

def pinword_to_perm(word: str) -> "Perm":
    """
    Returns the permutation associated with the given pinword.

    Examples:
        >>> pinword_to_perm("31")
        Perm((0, 1))
        >>> pinword_to_perm("4R")
        Perm((0, 1))
        >>> pinword_to_perm("3DL2UR")
        Perm((3, 5, 1, 2, 0, 4))
        >>> pinword_to_perm("14L2UR")
        Perm((3, 5, 1, 2, 0, 4))
    """
    ZERO = Fraction(0,1)
    ONE = Fraction(1,1)
    HALF = Fraction(1,2)
    p = [(ZERO, ZERO)]
    def min_x(p):
        return min(p, key=lambda x: x[0])[0]
    def max_x(p):
        return max(p, key=lambda x: x[0])[0]
    def min_y(p):
        return min(p, key=lambda x: x[1])[1]
    def max_y(p):
        return max(p, key=lambda x: x[1])[1]
    # Go through each character handling the different cases
    # For numbers we add a point in one of 4 quadrants
    # - 1: Northeast
    # - 2: Northwest
    # - 3: Southwest
    # - 4: Southeast
    # For letters when adding the nth value we are separating [p_0, ..., p_{n-2}] from p_{n-1}
    # and placing p_{n} between those two entities at the extreme of the specified direction.
    for c in word:
        if c == "1":
            next_x = max_x(p) + ONE
            next_y = max_y(p) + ONE
        elif c == "2":
            next_x = min_x(p) - ONE
            next_y = max_y(p) + ONE
        elif c == "3":
            next_x = min_x(p) - ONE
            next_y = min_y(p) - ONE
        elif c == "4":
            next_x = max_x(p) + ONE
            next_y = min_y(p) - ONE
        elif c == "U":
            last_x, last_y = p[-1]
            if last_x > max_x(p[:-1]):
                next_x = HALF * (last_x + max_x(p[:-1]))
                next_y = max_y(p) + ONE
            elif last_x < min_x(p[:-1]):
                next_x = HALF * (last_x + min_x(p[:-1]))
                next_y = max_y(p) + ONE
            else:
                assert False
        elif c == "L":
            last_x, last_y = p[-1]
            if last_y > max_y(p[:-1]):
                next_x = min_x(p) - ONE
                next_y = HALF * (last_y + max_y(p[:-1]))
            elif last_y < min_y(p[:-1]):
                next_x = min_x(p) - ONE
                next_y = HALF * (last_y + min_y(p[:-1]))
            else:
                assert False
        elif c == "D":
            last_x, last_y = p[-1]
            if last_x > max_x(p[:-1]):
                next_x = HALF * (last_x + max_x(p[:-1]))
                next_y = min_y(p) - ONE
            elif last_x < min_x(p[:-1]):
                next_x = HALF * (last_x + min_x(p[:-1]))
                next_y = min_y(p) - ONE
            else:
                assert False
        elif c == "R":
            last_x, last_y = p[-1]
            if last_y > max_y(p[:-1]):
                next_x = max_x(p) + ONE
                next_y = HALF * (last_y + max_y(p[:-1]))
            elif last_y < min_y(p[:-1]):
                next_x = max_x(p) + ONE
                next_y = HALF * (last_y + min_y(p[:-1]))
            else:
                assert False
        else:
            assert False
        p.append((next_x, next_y))

    # At this point we have obtained a geometric description of the permutation
    # First remove the origin
    p.pop(0)
    # Sort to get x-coordinates in order, like in a permutation
    p.sort() 
    # Obtain a sorted list of the y-coordinates
    sp = sorted(x[1] for x in p)
    # Standardize to obtain a classical permutation
    perm = tuple(bisect_left(sp, x[1]) for x in p)
    return Perm(perm)

def pinwords_of_length(n: int) -> Iterator[str]:
    """
    Generates all pinwords of length n.
    Note that pinwords cannot contain any occurrence of:
    UU, UD, DU, DD, LL, LR, RL, RR
    """
    if n == 0:
        yield ""
    else:
        for word in pinwords_of_length(n-1):
            if len(word) > 0 and word[-1] != "U" and word[-1] != "D":
                yield word + "U"
                yield word + "D"
            if len(word) > 0 and word[-1] != "R" and word[-1] != "L":
                yield word + "L"
                yield word + "R"
            for c in QUADS:
                yield word + c

@lru_cache(maxsize=None)
def pinword_to_perm_mapping(n: int) -> Dict:
    return {pinword:pinword_to_perm(pinword) for pinword in pinwords_of_length(n)}

@lru_cache(maxsize=None)
def perm_to_pinword_mapping(n: int) -> Dict:
    res = defaultdict(set)
    for k,v in pinword_to_perm_mapping(n).items():
        res[v].add(k)
    return res

def is_strict_pinword(w: str) -> bool:
    """
    Returns True if w is a strict pinword, False otherwise
    """
    if w == "":
        return True # paper does not mention the empty pinword
    return w[0] in QUADS and all(w[i] in DIRS for i in range(1, len(w)))

@lru_cache(maxsize=None)
def perm_to_strict_pinword_mapping(n: int) -> Dict:
    original = perm_to_pinword_mapping(n)
    filtered = {k:{x for x in v if is_strict_pinword(x)} for k,v in original.items()}
    return filtered

def perm_to_pinword(perm: "Perm", origin) -> str:
    pass

def perm_to_pinwords(perm: "Perm") -> List[str]:
    pass

def factor_pinword(word: str) -> List[str]:
    """
    Factors a pinword into its strong numeral led factor decomposition.
    
    Examples:
        >>> factor_pinword("14L2UR")
        ['1', '4L', '2UR']
    """
    at = 0
    factor_list = []
    while at < len(word):
        cur = at+1
        while cur < len(word) and word[cur] in DIRS:
            cur += 1
        factor_list.append(word[at:cur])
        at = cur
    return factor_list

def SP_to_M(word: str) -> Tuple[str]:
    """
    The bijection phi in Definition 3.9 mapping words in SP to words in M.
    Input must be a strict pin word. This implementation includes the extra
    definition given in Remark 3.11, mapping words in M to words in M.

    Examples:
        >>> SP_to_M("1R")
        ('RUR',)
        >>> SP_to_M("2UL")
        ('ULUL',)
        >>> SP_to_M("3")
        ('LD', 'DL')
        >>> SP_to_M("4D")
        ('DRD',)
    """
    if word == "":
        return ("",)
    if word[0] in QUADS:
        letter_dict = {"1":"RU", "2":"LU", "3": "LD", "4": "RD"}
        letters = letter_dict[word[0]]
        if len(word) == 1:
            return (letters, letters[::-1])
        if letters[1] == word[1]:
            letters = letters[::-1]
        return (letters + word[1:],)
    return word

def M_to_SP(word: str) -> str:
    """
    The bijection phi in Definition 3.9 mapping words in M to words in SP.
    
    Examples:
        >>> M_to_SP("RUR")
        '1R'
        >>> M_to_SP("ULUL")
        '2UL'
        >>> M_to_SP("DL")
        '3'
        >>> M_to_SP("LD")
        '3'
        >>> M_to_SP("DRD")
        '4D'
    """
    letter_dict = {"1":"RU", "2":"LU", "3": "LD", "4": "RD"}
    rev_letter_dict = dict()
    for k,v in letter_dict.items():
        rev_letter_dict[v] = k
        rev_letter_dict[v[::-1]] = k
    return rev_letter_dict[word[0:2]] + word[2:]

def quadrant(word: str, ind: int) -> str:
    """
    Determines the quadrant which point p_i in the pin representation resides in 
    with respect to the origin p_0. (Lemma 3.10)

    Examples:
        >>> quadrant("2RU4LULURD4L", 2)
        '1'
        >>> quadrant("2RU4LULURD4L", 3)
        '4'
        >>> quadrant("2RU4LULURD4L", 6)
        '2'
    """
    if word[ind] in QUADS:
        return word[ind]
    elif word[ind-1] in QUADS:
        return M_to_SP(SP_to_M(word[ind-1:ind+1])[0][1:])
    else:
        return M_to_SP(word[ind-1:ind+1])


def pinword_occurrences_SP(w: str, u: str, start_index: int=0) -> Iterator[int]:
    """
    Yields all occurrences (starting indices) of strict pinword u in pinword w (Lemma 3.12)
    """
    k = len(u)
    for i in range(start_index, len(w)):
        if quadrant(w, i) == quadrant(u, 0) and w[i+1:i+k] == u[1:]:
            yield i

def pinword_contains_SP(w: str, u: str) -> bool:
    return next(pinword_occurrences_SP(w, u), False) != False

def pinword_occurrences(w: str, u: str) -> Iterator[Tuple[int]]:
    """
    Yields all occurrences (starting indices of pinword u in pinword w (Theorem 3.13)
    """
    def rec(w: str, u: List[str], i: int, j: int, res: List[int]) -> bool:
        """
        Recursive helper function used to check for multiple sequential occurrences.
        """
        if j == len(u):
            yield tuple(res)
        elif i >= len(w):
            return
        else:
            for occ in pinword_occurrences_SP(w, u[j], i):
                res.append(occ)
                for x in rec(w, u, occ+len(u[j]), j+1, res):
                    yield x
                res.pop()
    return rec(w, factor_pinword(u), 0, 0, [])

def pinword_contains(w: str, u: str):
    return next(pinword_occurrences(w, u), False) != False

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    for k,v in perm_to_pinword_mapping(3).items():
        print(k, v)
    print()
    for k,v in perm_to_strict_pinword_mapping(3).items():
        print(k, v)
    
