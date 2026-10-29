//! Fill-reducing orderings for symmetric (or structurally symmetric) sparse matrices.
//!
//! - [`Ordering::Amd`]: approximate minimum degree (faer's implementation); best for small and 2D
//!   problems.
//! - [`Ordering::NestedDissection`]: multilevel nested dissection (heavy-edge coarsening, greedy
//!   graph-growing bisection, Fiduccia–Mattheyses refinement, minimum vertex cover separators,
//!   minimum degree on leaves). Asymptotically optimal fill for 3D meshes, and the separator tree
//!   exposes parallelism to the supernodal factorization.
//! - [`Ordering::Auto`]: nested dissection above [`ND_THRESHOLD`] vertices, AMD below.
//!
//! Both work on the compressed quotient graph when rows share identical patterns (vector degrees of
//! freedom at a node), which shrinks the ordering problem by the block size and keeps block rows
//! contiguous.

mod graph;
mod nd;

pub use graph::Graph;
pub(crate) use nd::edge_bisection;

use crate::csr::CsrRef;
use crate::error::{SolverError, SolverResult};
use crate::scalar::Scalar;

/// Vertex count at which `Auto` switches from AMD to nested dissection.
pub const ND_THRESHOLD: usize = 20_000;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum Ordering {
    #[default]
    Auto,
    Amd,
    NestedDissection,
    /// Keep the given order.
    Identity,
    /// `perm[new] = old`.
    Custom(Vec<usize>),
}

/// A permutation with `perm[new] = old` and `inverse[old] = new`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Permutation {
    pub perm: Vec<usize>,
    pub inverse: Vec<usize>,
}

impl Permutation {
    pub fn identity(n: usize) -> Self {
        Self {
            perm: (0..n).collect(),
            inverse: (0..n).collect(),
        }
    }
    pub fn from_perm(perm: Vec<usize>) -> SolverResult<Self> {
        let n = perm.len();
        let mut inverse = vec![usize::MAX; n];
        for (new, &old) in perm.iter().enumerate() {
            if old >= n || inverse[old] != usize::MAX {
                return Err(SolverError::InvalidStructure(
                    "ordering is not a permutation".into(),
                ));
            }
            inverse[old] = new;
        }
        Ok(Self { perm, inverse })
    }
    pub fn len(&self) -> usize {
        self.perm.len()
    }
    pub fn is_empty(&self) -> bool {
        self.perm.is_empty()
    }
}

/// Computes a fill-reducing ordering of `A + Aᵀ` (diagonal ignored).
pub fn order<T: Scalar>(matrix: CsrRef<'_, T>, ordering: &Ordering) -> SolverResult<Permutation> {
    if matrix.nrows() != matrix.ncols() {
        return Err(SolverError::Dimension(
            "ordering requires a square matrix".into(),
        ));
    }
    let n = matrix.nrows();
    match ordering {
        Ordering::Identity => Ok(Permutation::identity(n)),
        Ordering::Custom(perm) => {
            if perm.len() != n {
                return Err(SolverError::Dimension("custom ordering length".into()));
            }
            Permutation::from_perm(perm.clone())
        }
        Ordering::Amd | Ordering::NestedDissection | Ordering::Auto => {
            let graph = Graph::from_matrix(matrix);
            order_graph(&graph, ordering)
        }
    }
}

/// Orders an adjacency graph (vertices may carry weights from supervariable compression).
pub fn order_graph(graph: &Graph, ordering: &Ordering) -> SolverResult<Permutation> {
    let n = graph.vertex_count();
    // Compress identical rows (supervariables) first.
    let (quotient, members) = graph.compress_supervariables();
    let use_nd = match ordering {
        Ordering::NestedDissection => true,
        Ordering::Amd => false,
        _ => quotient.vertex_count() >= ND_THRESHOLD / members_average(&members, n),
    };
    let quotient_order = if use_nd {
        nd::nested_dissection(&quotient)
    } else {
        quotient.amd_order()?
    };
    let mut perm = Vec::with_capacity(n);
    for q in quotient_order {
        perm.extend_from_slice(&members[q]);
    }
    Permutation::from_perm(perm)
}

fn members_average(members: &[Vec<usize>], n: usize) -> usize {
    if members.is_empty() {
        1
    } else {
        (n / members.len()).max(1)
    }
}

#[cfg(test)]
mod tests;
