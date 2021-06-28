from pinperms import *

def all_classes_with_infinite_simples(n, filter_ins_enc=True):
    finite = dict()
    perms = list(Perm.of_length(n))
    q = deque()
    for p in perms:
        cur = (p,)
        if cur != lex_min(cur):
            continue
        L = load_DFA_for_perm(p)
        q.append((cur, L))
        val = True
        if not filter_ins_enc or not is_insertion_encodable(cur):
            val = has_finite_special_simples(cur) and has_finite_pinperms(cur, L=L)
        finite[cur] = val
        if val == False:
            yield cur

    
    with tqdm(total=len(q)) as pbar:
        while len(q) > 0:
            cur,L = q.popleft()
            pbar.set_postfix({"B":"_".join(str(i) for i in cur)})
            for p in perms:
                if p in cur:
                    continue
                nxt = tuple(sorted(cur + (p,)))
                if nxt != lex_min(nxt):
                    continue
                if nxt not in finite:
                    nxtL = DFA_name_reset(L.union(load_DFA_for_perm(p)))
                    val = True
                    if not filter_ins_enc or not is_insertion_encodable(nxt):
                        val = has_finite_special_simples(nxt) and has_finite_pinperms(nxt, L=nxtL)
                    finite[nxt] = val
                    if val == False:
                        yield nxt
                        q.append((nxt, nxtL))
                        pbar.total = pbar.total+1
                        pbar.refresh()
            pbar.update(1)





if __name__ == "__main__":
    #import doctest
    #doctest.testmod()
    n = int(sys.argv[1])
    
    for basis in all_classes_with_infinite_simples(n, True):
        print('_'.join(''.join(str(d) for d in p) for p in basis), flush=True)
