//! Undirected weighted graphs used by the orderings.

use std::collections::HashMap;

use crate::csr::CsrRef;
use crate::error::{SolverError, SolverResult};
use crate::scalar::Scalar;

/// Undirected graph in adjacency (CSR) form without self loops. Neighbor lists are sorted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Graph {
    pub xadj: Vec<usize>,
    pub adjncy: Vec<usize>,
    /// Edge weights aligned with `adjncy` (multiplicity after coarsening).
    pub adjwgt: Vec<u32>,
    /// Vertex weights (supervariable sizes / coarsened vertex counts).
    pub vwgt: Vec<u32>,
}

impl Graph {
    pub fn vertex_count(&self) -> usize {
        self.vwgt.len()
    }
    pub fn edge_count(&self) -> usize {
        self.adjncy.len() / 2
    }
    pub fn neighbors(&self, v: usize) -> &[usize] {
        &self.adjncy[self.xadj[v]..self.xadj[v + 1]]
    }
    pub fn edge_weights(&self, v: usize) -> &[u32] {
        &self.adjwgt[self.xadj[v]..self.xadj[v + 1]]
    }
    pub fn total_weight(&self) -> u64 {
        self.vwgt.iter().map(|&w| w as u64).sum()
    }

    /// Adjacency of `A + Aᵀ` without the diagonal.
    pub fn from_matrix<T: Scalar>(matrix: CsrRef<'_, T>) -> Self {
        let (xadj, adjncy) = matrix.symmetric_adjacency();
        let n = matrix.nrows();
        Self {
            adjwgt: vec![1; adjncy.len()],
            vwgt: vec![1; n],
            xadj,
            adjncy,
        }
    }

    /// From a symmetric adjacency list (both directions present, no self loops).
    pub fn from_adjacency(xadj: Vec<usize>, adjncy: Vec<usize>) -> Self {
        let n = xadj.len() - 1;
        Self {
            adjwgt: vec![1; adjncy.len()],
            vwgt: vec![1; n],
            xadj,
            adjncy,
        }
    }

    /// Groups vertices whose closed neighborhoods `{v} ∪ N(v)` are identical (supervariables)
    /// and returns the quotient graph with the member lists (in ascending original order).
    pub fn compress_supervariables(&self) -> (Graph, Vec<Vec<usize>>) {
        let n = self.vertex_count();
        let hash_of = |v: usize| -> u64 {
            let mut h: u64 = 0xcbf2_9ce4_8422_2325;
            let mut mix = |x: usize| {
                h ^= x as u64;
                h = h.wrapping_mul(0x100_0000_01b3);
            };
            // Closed neighborhood in sorted order.
            let neighbors = self.neighbors(v);
            let mut inserted = false;
            for &u in neighbors {
                if !inserted && v < u {
                    mix(v);
                    inserted = true;
                }
                mix(u);
            }
            if !inserted {
                mix(v);
            }
            h ^ (neighbors.len() as u64).rotate_left(17)
        };
        let closed_equal = |a: usize, b: usize| -> bool {
            let (na, nb) = (self.neighbors(a), self.neighbors(b));
            if na.len() != nb.len() {
                return false;
            }
            // {a} ∪ N(a) == {b} ∪ N(b) with a ∈ N(b) and b ∈ N(a).
            if na.binary_search(&b).is_err() {
                return false;
            }
            let mut ia = na
                .iter()
                .copied()
                .chain(std::iter::once(a))
                .collect::<Vec<_>>();
            let mut ib = nb
                .iter()
                .copied()
                .chain(std::iter::once(b))
                .collect::<Vec<_>>();
            ia.sort_unstable();
            ib.sort_unstable();
            ia == ib
        };
        let mut groups: HashMap<u64, Vec<usize>> = HashMap::new();
        let mut representative = vec![usize::MAX; n];
        let mut supervariable_of = vec![usize::MAX; n];
        let mut members: Vec<Vec<usize>> = Vec::new();
        for v in 0..n {
            let h = hash_of(v);
            let bucket = groups.entry(h).or_default();
            let mut placed = false;
            for &rep in bucket.iter() {
                if closed_equal(rep, v) {
                    let s = supervariable_of[rep];
                    supervariable_of[v] = s;
                    members[s].push(v);
                    placed = true;
                    break;
                }
            }
            if !placed {
                let s = members.len();
                members.push(vec![v]);
                supervariable_of[v] = s;
                representative[v] = v;
                bucket.push(v);
            }
        }
        if members.len() == n {
            return (self.clone(), members);
        }
        let q = members.len();
        let mut xadj = vec![0usize; q + 1];
        let mut adjncy = Vec::new();
        let mut adjwgt = Vec::new();
        let mut marker = vec![usize::MAX; q];
        for (s, group) in members.iter().enumerate() {
            let rep = group[0];
            let start = adjncy.len();
            for &u in self.neighbors(rep) {
                let t = supervariable_of[u];
                if t != s && marker[t] != s {
                    marker[t] = s;
                    adjncy.push(t);
                    adjwgt.push(1);
                }
            }
            adjncy[start..].sort_unstable();
            xadj[s + 1] = adjncy.len();
        }
        let vwgt = members.iter().map(|m| m.len() as u32).collect();
        (
            Graph {
                xadj,
                adjncy,
                adjwgt,
                vwgt,
            },
            members,
        )
    }

    /// Approximate minimum degree order of this graph (`order[new] = old`).
    pub fn amd_order(&self) -> SolverResult<Vec<usize>> {
        let n = self.vertex_count();
        if n == 0 {
            return Ok(Vec::new());
        }
        let mut perm = vec![0usize; n];
        let mut inverse = vec![0usize; n];
        let symbolic = faer::sparse::SymbolicSparseColMatRef::new_checked(
            n,
            n,
            &self.xadj,
            None,
            &self.adjncy,
        );
        let mut memory = faer::dyn_stack::MemBuffer::try_new(
            faer::sparse::linalg::amd::order_scratch::<usize>(n, self.adjncy.len()),
        )
        .map_err(|_| SolverError::OutOfMemory {
            bytes: (self.adjncy.len() * 24) as u64,
        })?;
        faer::sparse::linalg::amd::order(
            &mut perm,
            &mut inverse,
            symbolic,
            Default::default(),
            faer::dyn_stack::MemStack::new(&mut memory),
        )?;
        Ok(perm)
    }

    /// Induced subgraph on `vertices` (local numbering follows `vertices`).
    pub fn induced(&self, vertices: &[usize], local: &mut [usize]) -> Graph {
        for (i, &v) in vertices.iter().enumerate() {
            local[v] = i;
        }
        let mut xadj = Vec::with_capacity(vertices.len() + 1);
        xadj.push(0);
        let mut adjncy = Vec::new();
        let mut adjwgt = Vec::new();
        for &v in vertices {
            for (&u, &w) in self.neighbors(v).iter().zip(self.edge_weights(v)) {
                let lu = local[u];
                if lu != usize::MAX {
                    adjncy.push(lu);
                    adjwgt.push(w);
                }
            }
            xadj.push(adjncy.len());
        }
        let vwgt = vertices.iter().map(|&v| self.vwgt[v]).collect();
        for &v in vertices {
            local[v] = usize::MAX;
        }
        Graph {
            xadj,
            adjncy,
            adjwgt,
            vwgt,
        }
    }
}
