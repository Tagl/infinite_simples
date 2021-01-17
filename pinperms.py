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
    # For numbers we add a point in one of 4 corners
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

def pinwords_of_length(n: int):
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
def pinword_to_perm_mapping(n: int):
    return {pinword:pinword_to_perm(pinword) for pinword in pinwords_of_length(n)}

@lru_cache(maxsize=None)
def perm_to_pinword_mapping(n: int):
    res = defaultdict(set)
    for k,v in pinword_to_perm_mapping(n).items():
        res[v].add(k)
    return res

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
        return M_to_SP(SP_to_M(word[ind-1:ind+1])[1:])
    else:
        return M_to_SP(word[ind-1:ind+1])


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    print(SP_to_M(
