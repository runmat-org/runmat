//! Automatic choice between direct and iterative solution, with explicit overrides.
//!
//! The decision uses the problem class (definiteness, symmetry, saddle structure, complex),
//! size, the memory budget, how many solves will reuse a factorization, and whether a
//! near-nullspace is available for multigrid. Every decision carries a human-readable reason for
//! solver provenance.

use crate::direct::FactorKind;
use crate::estimate::{
    amg_bytes, ilu0_bytes, krylov_workspace_bytes, quick_direct_estimate, KrylovMethod,
};

/// Algebraic class of the system matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProblemClass {
    /// Real symmetric (or complex Hermitian) positive definite.
    SymmetricPositiveDefinite,
    /// Real symmetric / Hermitian indefinite (shifted operators, buckling pencils).
    SymmetricIndefinite,
    /// `[A Bᵀ; B −C]` with `A` SPD and `C` PSD: quasi-definite after regularization.
    SaddlePoint,
    /// Complex symmetric (`Aᵀ = A`, not Hermitian): Helmholtz, Maxwell, damped harmonic.
    ComplexSymmetric,
    /// No exploitable symmetry (convection, contact, Newton tangents of non-associative models).
    General,
}

/// Preconditioner choice for iterative strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreconditionerChoice {
    None,
    Jacobi,
    BlockJacobi,
    Ilu0,
    Ilut,
    Amg,
    AdditiveSchwarz,
}

impl PreconditionerChoice {
    pub fn as_str(self) -> &'static str {
        match self {
            PreconditionerChoice::None => "none",
            PreconditionerChoice::Jacobi => "jacobi",
            PreconditionerChoice::BlockJacobi => "block_jacobi",
            PreconditionerChoice::Ilu0 => "ilu0",
            PreconditionerChoice::Ilut => "ilut",
            PreconditionerChoice::Amg => "amg",
            PreconditionerChoice::AdditiveSchwarz => "additive_schwarz",
        }
    }
}

/// A concrete strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverStrategy {
    Direct(FactorKind),
    Iterative {
        method: KrylovMethod,
        preconditioner: PreconditionerChoice,
    },
}

impl SolverStrategy {
    pub fn is_direct(&self) -> bool {
        matches!(self, SolverStrategy::Direct(_))
    }
    pub fn label(&self) -> String {
        match self {
            SolverStrategy::Direct(kind) => format!("direct_{}", kind.as_str()),
            SolverStrategy::Iterative {
                method,
                preconditioner,
            } => {
                let method = match method {
                    KrylovMethod::Pcg => "pcg".to_string(),
                    KrylovMethod::Minres => "minres".to_string(),
                    KrylovMethod::Gmres { restart } => format!("gmres({restart})"),
                    KrylovMethod::Bicgstab => "bicgstab".to_string(),
                };
                format!("{method}+{}", preconditioner.as_str())
            }
        }
    }
}

/// Caller (study) override.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SolverOverride {
    #[default]
    Auto,
    /// Use the class's direct factorization.
    Direct,
    /// Use the class's iterative strategy.
    Iterative,
    Exact(SolverStrategy),
}

/// What the policy knows about a system.
#[derive(Debug, Clone, PartialEq)]
pub struct ProblemDescriptor {
    pub n: usize,
    pub nnz: usize,
    pub class: ProblemClass,
    /// Dofs per node (1 scalar, 2/3 elasticity).
    pub block_size: usize,
    pub complex: bool,
    /// Spatial dimension of the discretization (2 or 3) for fill estimates.
    pub spatial_dimension: u8,
    /// Rigid-body or other near-nullspace modes are available for multigrid.
    pub near_nullspace_available: bool,
    /// Expected number of solves sharing one factorization (time steps, eigen iterations, load
    /// cases, frequency points with one shift).
    pub solves_per_factorization: usize,
    /// Bytes the solve may use (`None`: unbounded).
    pub memory_budget_bytes: Option<u64>,
}

/// The decision with its quoted memory and reasoning.
#[derive(Debug, Clone, PartialEq)]
pub struct PolicyDecision {
    pub strategy: SolverStrategy,
    pub estimated_bytes: u64,
    pub reason: String,
}

/// Largest system the automatic policy factors directly for a single solve.
pub const DIRECT_SINGLE_SOLVE_LIMIT: usize = 400_000;
/// Largest system factored directly when the factorization is reused many times.
pub const DIRECT_REUSED_LIMIT: usize = 3_000_000;
/// Fraction of the budget a direct factorization may take.
const BUDGET_FRACTION: f64 = 0.7;

fn direct_kind(class: ProblemClass) -> FactorKind {
    match class {
        ProblemClass::SymmetricPositiveDefinite => FactorKind::Cholesky,
        ProblemClass::SymmetricIndefinite => FactorKind::BunchKaufman,
        ProblemClass::SaddlePoint => FactorKind::Ldlt,
        ProblemClass::ComplexSymmetric | ProblemClass::General => FactorKind::Lu,
    }
}

fn iterative_strategy(problem: &ProblemDescriptor) -> SolverStrategy {
    match problem.class {
        ProblemClass::SymmetricPositiveDefinite if !problem.complex => SolverStrategy::Iterative {
            method: KrylovMethod::Pcg,
            preconditioner: PreconditionerChoice::Amg,
        },
        ProblemClass::SymmetricPositiveDefinite => SolverStrategy::Iterative {
            method: KrylovMethod::Pcg,
            preconditioner: PreconditionerChoice::Ilu0,
        },
        ProblemClass::SymmetricIndefinite | ProblemClass::SaddlePoint => {
            SolverStrategy::Iterative {
                method: KrylovMethod::Gmres { restart: 100 },
                preconditioner: PreconditionerChoice::Ilut,
            }
        }
        ProblemClass::ComplexSymmetric | ProblemClass::General => SolverStrategy::Iterative {
            method: KrylovMethod::Gmres { restart: 100 },
            preconditioner: PreconditionerChoice::Ilut,
        },
    }
}

/// Bytes of a strategy for `problem`.
pub fn estimated_bytes(problem: &ProblemDescriptor, strategy: SolverStrategy) -> u64 {
    let scalar = if problem.complex { 16 } else { 8 };
    match strategy {
        SolverStrategy::Direct(kind) => {
            let (entries, _) =
                quick_direct_estimate(problem.n, problem.nnz, problem.spatial_dimension);
            let factor = if kind == FactorKind::Lu {
                2.0 * entries
            } else {
                entries
            };
            (factor * (scalar as f64 + 1.0) + (problem.n * 64) as f64) as u64
        }
        SolverStrategy::Iterative {
            method,
            preconditioner,
        } => {
            let workspace = krylov_workspace_bytes(method, problem.n, scalar);
            let preconditioner = match preconditioner {
                PreconditionerChoice::None => 0,
                PreconditionerChoice::Jacobi => (problem.n * scalar) as u64,
                PreconditionerChoice::BlockJacobi => {
                    (problem.n * problem.block_size.max(1) * scalar) as u64
                }
                PreconditionerChoice::Ilu0 => ilu0_bytes(problem.n, problem.nnz, scalar),
                PreconditionerChoice::Ilut => 3 * ilu0_bytes(problem.n, problem.nnz, scalar),
                PreconditionerChoice::Amg => amg_bytes(problem.n, problem.nnz, problem.block_size),
                PreconditionerChoice::AdditiveSchwarz => {
                    let (entries, _) =
                        quick_direct_estimate(problem.n, problem.nnz, problem.spatial_dimension);
                    (entries * 0.2 * (scalar as f64 + 8.0)) as u64
                }
            };
            workspace + preconditioner
        }
    }
}

/// Chooses a strategy.
pub fn choose(problem: &ProblemDescriptor, choice: SolverOverride) -> PolicyDecision {
    let direct = SolverStrategy::Direct(direct_kind(problem.class));
    let iterative = iterative_strategy(problem);
    let decide = |strategy: SolverStrategy, reason: String| PolicyDecision {
        strategy,
        estimated_bytes: estimated_bytes(problem, strategy),
        reason,
    };
    match choice {
        SolverOverride::Exact(strategy) => {
            return decide(strategy, "requested by the study".into())
        }
        SolverOverride::Direct => {
            return decide(direct, "direct solution requested by the study".into())
        }
        SolverOverride::Iterative => {
            return decide(
                iterative,
                "iterative solution requested by the study".into(),
            )
        }
        SolverOverride::Auto => {}
    }
    let direct_bytes = estimated_bytes(problem, direct);
    let fits = problem
        .memory_budget_bytes
        .is_none_or(|budget| (direct_bytes as f64) <= BUDGET_FRACTION * budget as f64);
    let reused = problem.solves_per_factorization >= 8;
    let limit = if reused {
        DIRECT_REUSED_LIMIT
    } else {
        DIRECT_SINGLE_SOLVE_LIMIT
    };
    // Classes without a scalable iterative preconditioner prefer direct whenever it fits.
    let iterative_is_strong =
        matches!(problem.class, ProblemClass::SymmetricPositiveDefinite) && !problem.complex;
    if fits && (problem.n <= limit || !iterative_is_strong) {
        return decide(
            direct,
            format!(
                "direct: n={} within the {} limit {} and estimated factor {} MiB fits the budget",
                problem.n,
                if reused {
                    "reused-factorization"
                } else {
                    "single-solve"
                },
                limit,
                direct_bytes >> 20
            ),
        );
    }
    decide(
        iterative,
        if !fits {
            format!(
                "iterative: estimated direct factor {} MiB exceeds the budget",
                direct_bytes >> 20
            )
        } else {
            format!(
                "iterative: n={} above the direct limit {limit}; multigrid scales linearly",
                problem.n
            )
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spd(n: usize) -> ProblemDescriptor {
        ProblemDescriptor {
            n,
            nnz: 27 * n,
            class: ProblemClass::SymmetricPositiveDefinite,
            block_size: 3,
            complex: false,
            spatial_dimension: 3,
            near_nullspace_available: true,
            solves_per_factorization: 1,
            memory_budget_bytes: Some(32 << 30),
        }
    }

    #[test]
    fn policy_prefers_direct_for_moderate_and_amg_for_huge_spd() {
        assert_eq!(
            choose(&spd(100_000), SolverOverride::Auto).strategy,
            SolverStrategy::Direct(FactorKind::Cholesky)
        );
        let huge = choose(&spd(5_000_000), SolverOverride::Auto);
        assert!(!huge.strategy.is_direct(), "{huge:?}");
        let forced = choose(&spd(5_000_000), SolverOverride::Direct);
        assert!(forced.strategy.is_direct());
        let general = ProblemDescriptor {
            class: ProblemClass::ComplexSymmetric,
            complex: true,
            ..spd(200_000)
        };
        assert_eq!(
            choose(&general, SolverOverride::Auto).strategy,
            SolverStrategy::Direct(FactorKind::Lu)
        );
    }
}
