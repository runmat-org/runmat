//! Overlapping additive Schwarz with pluggable subdomain solvers.
//!
//! `M⁻¹ = Σ_i R_iᵀ A_i⁻¹ R_i` (additive, symmetric — usable in PCG) or the restricted variant
//! `Σ_i R̃_iᵀ A_i⁻¹ R_i` (RAS, for GMRES), where `R̃_i` keeps only the owned (non-overlap) rows.
//! Subdomains are solved independently in parallel. A distributed domain-decomposition driver
//! implements [`SubdomainSolver`] for remote subdomains and reuses the partitioning here.

use std::sync::Arc;

use crate::control::SolveControl;
use crate::csr::{CsrMatrix, CsrRef};
use crate::direct::{DirectOptions, DirectSolver, FactorKind, FactorScalar};
use crate::error::{SolverError, SolverResult};
use crate::operator::Preconditioner;
use crate::ordering::{edge_bisection, Graph};
use crate::par;
use runmat_simulation_usage::{UsageCategory, UsageContext};

/// Solves one subdomain problem `A_i z = r` (local numbering).
pub trait SubdomainSolver<T>: Send + Sync {
    fn solve(&self, r: &[T], z: &mut [T]) -> SolverResult<()>;
}

/// Caller-owned subdomain indices, owned-row mask, and shared solve owner.
pub type SubdomainInput<T> = (Vec<usize>, Vec<bool>, Arc<dyn SubdomainSolver<T>>);

/// Local sparse direct factorization of the subdomain matrix.
pub struct DirectSubdomainSolver<T: FactorScalar>(pub DirectSolver<T>);

impl<T: FactorScalar> SubdomainSolver<T> for DirectSubdomainSolver<T> {
    fn solve(&self, r: &[T], z: &mut [T]) -> SolverResult<()> {
        z.copy_from_slice(r);
        self.0.solve_in_place(z, 1)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SchwarzOptions {
    pub subdomains: usize,
    /// Graph layers of overlap added around each owned part.
    pub overlap: usize,
    /// Restricted additive Schwarz (not symmetric; for GMRES/BiCGStab).
    pub restricted: bool,
    /// Local factorization kind when the default direct subdomain solver is used.
    pub local_kind: FactorKind,
}

impl Default for SchwarzOptions {
    fn default() -> Self {
        Self {
            subdomains: 8,
            overlap: 1,
            restricted: false,
            local_kind: FactorKind::Cholesky,
        }
    }
}

/// Balanced graph partition into `parts` pieces by recursive multilevel bisection.
pub fn graph_partition(graph: &Graph, parts: usize) -> Vec<usize> {
    let n = graph.vertex_count();
    let mut assignment = vec![0usize; n];
    fn recurse(graph: &Graph, ids: &[usize], parts: usize, first: usize, assignment: &mut [usize]) {
        if parts <= 1 || ids.len() <= 1 {
            for &v in ids {
                assignment[v] = first;
            }
            return;
        }
        let labels = edge_bisection(graph);
        let left_parts = parts / 2;
        let (mut left, mut right) = (Vec::new(), Vec::new());
        for (local, &label) in labels.iter().enumerate() {
            if label == 0 {
                left.push(local);
            } else {
                right.push(local);
            }
        }
        let mut scratch = vec![usize::MAX; graph.vertex_count()];
        let left_graph = graph.induced(&left, &mut scratch);
        let right_graph = graph.induced(&right, &mut scratch);
        let left_ids: Vec<usize> = left.iter().map(|&v| ids[v]).collect();
        let right_ids: Vec<usize> = right.iter().map(|&v| ids[v]).collect();
        recurse(&left_graph, &left_ids, left_parts, first, assignment);
        recurse(
            &right_graph,
            &right_ids,
            parts - left_parts,
            first + left_parts,
            assignment,
        );
    }
    let ids: Vec<usize> = (0..n).collect();
    recurse(graph, &ids, parts.max(1), 0, &mut assignment);
    assignment
}

struct Subdomain<T> {
    /// Global indices (sorted) of the overlapping subdomain.
    indices: Vec<usize>,
    /// Owned flags aligned with `indices` (used by RAS).
    owned: Vec<bool>,
    solver: Arc<dyn SubdomainSolver<T>>,
}

/// Additive Schwarz preconditioner.
pub struct AdditiveSchwarz<T> {
    n: usize,
    subdomains: Vec<Subdomain<T>>,
    restricted: bool,
}

impl<T: FactorScalar> AdditiveSchwarz<T> {
    /// Partitions the matrix graph and factors every overlapping subdomain matrix directly.
    pub fn new(
        matrix: CsrRef<'_, T>,
        options: &SchwarzOptions,
        control: SolveControl<'_>,
    ) -> SolverResult<Self> {
        let graph = Graph::from_matrix(matrix);
        let assignment = graph_partition(&graph, options.subdomains);
        let sets = overlapping_sets(&graph, &assignment, options.subdomains, options.overlap);
        let usage = UsageContext::capture();
        let local: Vec<SolverResult<Subdomain<T>>> = par::map_range(sets.len(), 1, |part| {
            let _usage_guard = usage.as_ref().map(UsageContext::enter);
            control.check()?;
            let (indices, owned) = &sets[part];
            let sub: CsrMatrix<T> = matrix.submatrix(indices, indices);
            let solver = DirectSolver::new(
                sub.as_ref().into(),
                DirectOptions::new(options.local_kind),
                SolveControl::NONE,
            )?;
            Ok(Subdomain {
                indices: indices.clone(),
                owned: owned.clone(),
                solver: Arc::new(DirectSubdomainSolver(solver)),
            })
        });
        let subdomains = local.into_iter().collect::<SolverResult<Vec<_>>>()?;
        Ok(Self {
            n: matrix.nrows(),
            subdomains,
            restricted: options.restricted,
        })
    }
}

impl<T: crate::scalar::Scalar> AdditiveSchwarz<T> {
    /// Uses caller-supplied subdomain index sets and solvers (e.g. remote workers).
    pub fn from_subdomains(
        n: usize,
        subdomains: Vec<SubdomainInput<T>>,
        restricted: bool,
    ) -> SolverResult<Self> {
        let mut covered = vec![false; n];
        let mut out = Vec::with_capacity(subdomains.len());
        for (indices, owned, solver) in subdomains {
            if indices.len() != owned.len() || indices.iter().any(|&i| i >= n) {
                return Err(SolverError::Dimension("subdomain index sets".into()));
            }
            for &i in &indices {
                covered[i] = true;
            }
            out.push(Subdomain {
                indices,
                owned,
                solver,
            });
        }
        if let Some(missing) = covered.iter().position(|c| !c) {
            return Err(SolverError::InvalidStructure(format!(
                "row {missing} is in no subdomain"
            )));
        }
        Ok(Self {
            n,
            subdomains: out,
            restricted,
        })
    }

    pub fn subdomain_count(&self) -> usize {
        self.subdomains.len()
    }
}

/// Owned parts plus `overlap` layers of neighbors.
fn overlapping_sets(
    graph: &Graph,
    assignment: &[usize],
    parts: usize,
    overlap: usize,
) -> Vec<(Vec<usize>, Vec<bool>)> {
    let n = graph.vertex_count();
    (0..parts)
        .map(|part| {
            let mut member = vec![false; n];
            let mut frontier: Vec<usize> = (0..n).filter(|&v| assignment[v] == part).collect();
            for &v in &frontier {
                member[v] = true;
            }
            for _ in 0..overlap {
                let mut next = Vec::new();
                for &v in &frontier {
                    for &u in graph.neighbors(v) {
                        if !member[u] {
                            member[u] = true;
                            next.push(u);
                        }
                    }
                }
                frontier = next;
            }
            let indices: Vec<usize> = (0..n).filter(|&v| member[v]).collect();
            let owned = indices.iter().map(|&v| assignment[v] == part).collect();
            (indices, owned)
        })
        .filter(|(indices, _)| !indices.is_empty())
        .collect()
}

impl<T: crate::scalar::Scalar> Preconditioner<T> for AdditiveSchwarz<T> {
    fn apply(&self, r: &[T], z: &mut [T]) -> SolverResult<()> {
        let usage = UsageContext::capture();
        let local: Vec<SolverResult<Vec<T>>> = par::map_range(self.subdomains.len(), 1, |s| {
            let _usage_guard = usage.as_ref().map(UsageContext::enter);
            let sub = &self.subdomains[s];
            let rl: Vec<T> = sub.indices.iter().map(|&i| r[i]).collect();
            let mut zl = vec![T::zero(); rl.len()];
            sub.solver.solve(&rl, &mut zl)?;
            Ok(zl)
        });
        z.iter_mut().for_each(|v| *v = T::zero());
        // Fixed subdomain order keeps the accumulation deterministic.
        for (sub, zl) in self.subdomains.iter().zip(local) {
            let zl = zl?;
            let additions = if self.restricted {
                sub.owned.iter().filter(|owned| **owned).count()
            } else {
                sub.indices.len()
            };
            super::usage::measured(
                UsageCategory::Solve,
                super::usage::count(additions, if T::IS_COMPLEX { 2 } else { 1 })?,
                || {
                    for (k, &i) in sub.indices.iter().enumerate() {
                        if !self.restricted || sub.owned[k] {
                            z[i] += zl[k];
                        }
                    }
                    Ok(())
                },
            )?;
        }
        debug_assert_eq!(z.len(), self.n);
        Ok(())
    }
    fn label(&self) -> &'static str {
        if self.restricted {
            "restricted_additive_schwarz"
        } else {
            "additive_schwarz"
        }
    }
    fn is_symmetric_positive_definite(&self) -> bool {
        !self.restricted
    }
}
