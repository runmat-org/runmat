//! Structural decomposition.
//!
//! 1. **Connected components** (union-find over the variables each constraint reads): disjoint
//!    sub-problems that solve and diagnose independently.
//! 2. **Dulmage–Mendelsohn** inside a component (maximum matching of equations to variables,
//!    then strongly connected components of the square part): an over-determined block, a
//!    sequence of square blocks in dependency order, and an under-determined block. Solving
//!    them in that order replaces one large Newton system with several small ones.
//!
//! All traversals use index order, so the decomposition is deterministic.

const NONE: usize = usize::MAX;

/// A connected sub-problem: local row indices and variable indices, both ascending.
#[derive(Clone, Debug)]
pub(crate) struct Component {
    pub rows: Vec<usize>,
    pub vars: Vec<usize>,
}

fn find(p: &mut [usize], mut i: usize) -> usize {
    while p[i] != i {
        p[i] = p[p[i]];
        i = p[i];
    }
    i
}

/// Splits rows and variables into connected components. `row_vars[r]` lists the variables row
/// `r` reads (indices `< nvars`). Rows that read no variable are returned separately.
pub(crate) fn components(nvars: usize, row_vars: &[Vec<usize>]) -> (Vec<Component>, Vec<usize>) {
    let mut parent: Vec<usize> = (0..nvars).collect();
    for vars in row_vars {
        if let Some((&first, rest)) = vars.split_first() {
            for &v in rest {
                let (a, b) = (find(&mut parent, first), find(&mut parent, v));
                if a != b {
                    // Union toward the smaller root so roots are canonical.
                    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                    parent[hi] = lo;
                }
            }
        }
    }
    let mut slot = vec![NONE; nvars];
    let mut out: Vec<Component> = Vec::new();
    for v in 0..nvars {
        let r = find(&mut parent, v);
        if slot[r] == NONE {
            slot[r] = out.len();
            out.push(Component {
                rows: Vec::new(),
                vars: Vec::new(),
            });
        }
        out[slot[r]].vars.push(v);
    }
    let mut orphan = Vec::new();
    for (r, vars) in row_vars.iter().enumerate() {
        match vars.first() {
            Some(&v) => {
                let root = find(&mut parent, v);
                out[slot[root]].rows.push(r);
            }
            None => orphan.push(r),
        }
    }
    (out, orphan)
}

/// What a DM block is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BlockKind {
    Over,
    Square,
    Under,
}

/// A block of the DM decomposition (indices into the component's local rows and vars).
#[derive(Clone, Debug)]
pub(crate) struct Block {
    #[cfg_attr(not(test), allow(dead_code))]
    pub kind: BlockKind,
    pub rows: Vec<usize>,
    pub vars: Vec<usize>,
}

/// Dulmage–Mendelsohn decomposition of a bipartite structure with `nrows` equations and `nvars`
/// unknowns. Returns blocks in solve order: over-determined, square blocks (dependencies first),
/// under-determined. Empty blocks are omitted.
pub(crate) fn dulmage_mendelsohn(
    nrows: usize,
    nvars: usize,
    row_vars: &[Vec<usize>],
) -> Vec<Block> {
    // Variable → rows.
    let mut var_rows: Vec<Vec<usize>> = vec![Vec::new(); nvars];
    for (r, vs) in row_vars.iter().enumerate() {
        for &v in vs {
            var_rows[v].push(r);
        }
    }
    // Maximum matching: greedy start, then augmenting paths (iterative DFS).
    let mut row_match = vec![NONE; nrows];
    let mut var_match = vec![NONE; nvars];
    for r in 0..nrows {
        for &v in &row_vars[r] {
            if var_match[v] == NONE {
                var_match[v] = r;
                row_match[r] = v;
                break;
            }
        }
    }
    let mut seen = vec![usize::MAX; nvars];
    for r0 in 0..nrows {
        if row_match[r0] != NONE {
            continue;
        }
        // Stack of (row, next edge index): the current alternating path.
        let mut stack: Vec<(usize, usize)> = vec![(r0, 0)];
        let mut found = NONE;
        while let Some(&mut (r, ref mut e)) = stack.last_mut() {
            if *e >= row_vars[r].len() {
                stack.pop();
                continue;
            }
            let v = row_vars[r][*e];
            *e += 1;
            if seen[v] == r0 {
                continue;
            }
            seen[v] = r0;
            if var_match[v] == NONE {
                found = v;
                break;
            }
            stack.push((var_match[v], 0));
        }
        if found != NONE {
            // Walk back along the stack: each stacked row takes the var it reached next.
            let rows_on_path: Vec<usize> = stack.iter().map(|s| s.0).collect();
            let mut v = found;
            for &r in rows_on_path.iter().rev() {
                let prev = row_match[r];
                row_match[r] = v;
                var_match[v] = r;
                v = prev;
                if v == NONE {
                    break;
                }
            }
        }
    }

    // Over-determined part: alternating reach from unmatched rows.
    let mut in_over_row = vec![false; nrows];
    let mut in_over_var = vec![false; nvars];
    let mut queue: Vec<usize> = (0..nrows).filter(|&r| row_match[r] == NONE).collect();
    for &r in &queue {
        in_over_row[r] = true;
    }
    while let Some(r) = queue.pop() {
        for &v in &row_vars[r] {
            if !in_over_var[v] {
                in_over_var[v] = true;
                let m = var_match[v];
                if m != NONE && !in_over_row[m] {
                    in_over_row[m] = true;
                    queue.push(m);
                }
            }
        }
    }
    // Under-determined part: alternating reach from unmatched vars.
    let mut in_under_row = vec![false; nrows];
    let mut in_under_var = vec![false; nvars];
    let mut queue: Vec<usize> = (0..nvars).filter(|&v| var_match[v] == NONE).collect();
    for &v in &queue {
        in_under_var[v] = true;
    }
    while let Some(v) = queue.pop() {
        for &r in &var_rows[v] {
            if !in_under_row[r] {
                in_under_row[r] = true;
                let m = row_match[r];
                if m != NONE && !in_under_var[m] {
                    in_under_var[m] = true;
                    queue.push(m);
                }
            }
        }
    }

    let mut blocks = Vec::new();
    let over_rows: Vec<usize> = (0..nrows).filter(|&r| in_over_row[r]).collect();
    let over_vars: Vec<usize> = (0..nvars).filter(|&v| in_over_var[v]).collect();
    if !over_rows.is_empty() {
        blocks.push(Block {
            kind: BlockKind::Over,
            rows: over_rows,
            vars: over_vars,
        });
    }

    // Square part: Tarjan SCC over matched rows, edge r → match(v) for each square var v of r.
    let square: Vec<usize> = (0..nrows)
        .filter(|&r| !in_over_row[r] && !in_under_row[r])
        .collect();
    let mut is_square = vec![false; nrows];
    for &r in &square {
        is_square[r] = true;
    }
    let succ = |r: usize| -> Vec<usize> {
        row_vars[r]
            .iter()
            .filter_map(|&v| {
                let m = var_match[v];
                (m != NONE && is_square[m] && m != r).then_some(m)
            })
            .collect()
    };
    let mut index = vec![NONE; nrows];
    let mut low = vec![0usize; nrows];
    let mut on_stack = vec![false; nrows];
    let mut tstack: Vec<usize> = Vec::new();
    let mut next = 0usize;
    for &root in &square {
        if index[root] != NONE {
            continue;
        }
        let mut call: Vec<(usize, Vec<usize>, usize)> = vec![(root, succ(root), 0)];
        index[root] = next;
        low[root] = next;
        next += 1;
        tstack.push(root);
        on_stack[root] = true;
        while let Some((v, ws, i)) = call.last_mut() {
            let v = *v;
            if *i < ws.len() {
                let w = ws[*i];
                *i += 1;
                if index[w] == NONE {
                    index[w] = next;
                    low[w] = next;
                    next += 1;
                    tstack.push(w);
                    on_stack[w] = true;
                    let s = succ(w);
                    call.push((w, s, 0));
                } else if on_stack[w] {
                    low[v] = low[v].min(index[w]);
                }
            } else {
                call.pop();
                if let Some((u, _, _)) = call.last() {
                    let u = *u;
                    low[u] = low[u].min(low[v]);
                }
                if low[v] == index[v] {
                    let mut rows = Vec::new();
                    loop {
                        let w = tstack.pop().unwrap();
                        on_stack[w] = false;
                        rows.push(w);
                        if w == v {
                            break;
                        }
                    }
                    rows.sort_unstable();
                    let mut vars: Vec<usize> = rows.iter().map(|&r| row_match[r]).collect();
                    vars.sort_unstable();
                    blocks.push(Block {
                        kind: BlockKind::Square,
                        rows,
                        vars,
                    });
                }
            }
        }
    }

    let under_rows: Vec<usize> = (0..nrows).filter(|&r| in_under_row[r]).collect();
    let under_vars: Vec<usize> = (0..nvars).filter(|&v| in_under_var[v]).collect();
    if !under_vars.is_empty() {
        blocks.push(Block {
            kind: BlockKind::Under,
            rows: under_rows,
            vars: under_vars,
        });
    }
    blocks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn components_split_independent_parts() {
        let rows = vec![vec![0, 1], vec![2], vec![1, 3], vec![]];
        let (c, orphan) = components(5, &rows);
        assert_eq!(c.len(), 3);
        assert_eq!(c[0].vars, vec![0, 1, 3]);
        assert_eq!(c[0].rows, vec![0, 2]);
        assert_eq!(c[1].vars, vec![2]);
        assert_eq!(c[2].vars, vec![4]);
        assert!(c[2].rows.is_empty());
        assert_eq!(orphan, vec![3]);
    }

    #[test]
    fn dm_orders_triangular_chain() {
        // r0: x0; r1: x0, x1; r2: x1, x2 → three 1×1 blocks in order, then x3 under.
        let rows = vec![vec![0], vec![0, 1], vec![1, 2], vec![2, 3, 4]];
        let b = dulmage_mendelsohn(4, 5, &rows);
        let kinds: Vec<_> = b.iter().map(|b| b.kind).collect();
        assert_eq!(
            kinds,
            vec![
                BlockKind::Square,
                BlockKind::Square,
                BlockKind::Square,
                BlockKind::Under
            ]
        );
        assert_eq!(b[0].vars, vec![0]);
        assert_eq!(b[1].vars, vec![1]);
        assert_eq!(b[2].vars, vec![2]);
        assert_eq!(b[3].rows, vec![3]);
    }

    #[test]
    fn dm_finds_over_part() {
        // Two rows on x0 only, then a row that settles x1 given x0, then x2 left free.
        let rows = vec![vec![0], vec![0], vec![0, 1], vec![1, 2, 3]];
        let b = dulmage_mendelsohn(4, 4, &rows);
        assert_eq!(b[0].kind, BlockKind::Over);
        assert_eq!(b[0].rows, vec![0, 1]);
        assert_eq!(b[1].kind, BlockKind::Square);
        assert_eq!(b[1].vars, vec![1]);
        assert_eq!(b[2].kind, BlockKind::Under);
        assert_eq!(b[2].rows, vec![3]);
    }
}
