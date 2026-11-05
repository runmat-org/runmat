//! Minimum-degree ordering on the symmetrized pattern `A + Aᵀ`.
//!
//! Plain elimination-graph minimum degree with lazy priority updates; ties go to the lowest
//! index, so the ordering is deterministic. Circuit graphs are sparse with small fill, which this
//! handles well into the tens of thousands of unknowns.

use super::Pattern;
use std::cmp::Reverse;
use std::collections::{BTreeSet, BinaryHeap};

/// The elimination order: `order[k]` is the column eliminated at step `k`.
pub fn minimum_degree(p: &Pattern) -> Vec<usize> {
    let n = p.n;
    let mut adj: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];
    for (r, c) in p.entries() {
        if r != c {
            adj[r].insert(c);
            adj[c].insert(r);
        }
    }
    let mut heap: BinaryHeap<Reverse<(usize, usize)>> =
        (0..n).map(|v| Reverse((adj[v].len(), v))).collect();
    let mut done = vec![false; n];
    let mut order = Vec::with_capacity(n);
    while let Some(Reverse((deg, v))) = heap.pop() {
        if done[v] || deg != adj[v].len() {
            continue;
        }
        done[v] = true;
        order.push(v);
        let nb: Vec<usize> = std::mem::take(&mut adj[v]).into_iter().collect();
        for &u in &nb {
            adj[u].remove(&v);
        }
        for (i, &u) in nb.iter().enumerate() {
            for &w in &nb[i + 1..] {
                adj[u].insert(w);
                adj[w].insert(u);
            }
        }
        for &u in &nb {
            heap.push(Reverse((adj[u].len(), u)));
        }
    }
    order
}
