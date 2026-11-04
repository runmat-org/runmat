//! Preconditioners.
//!
//! - [`Jacobi`] and [`BlockJacobi`] (point and node-block diagonal scaling);
//! - [`Ilu0`] (zero fill; symmetric input gives the IC(0)-equivalent factor with a Manteuffel
//!   shift retry) and [`Ilut`] (dual threshold);
//! - [`Amg`]: smoothed-aggregation algebraic multigrid with near-nullspace modes (constants for
//!   scalar problems, rigid-body modes for elasticity) and Chebyshev smoothing — the scalable
//!   preconditioner for SPD systems;
//! - [`AdditiveSchwarz`]: overlapping subdomains with pluggable local solvers (direct by default),
//!   the hook used by distributed domain decomposition;
//! - [`DirectPreconditioner`]: a (possibly inexact or lagged) direct factorization used as a
//!   preconditioner, e.g. a factor of a nearby matrix in Newton or frequency sweeps.

mod amg;
mod direct;
mod ilu;
mod jacobi;
mod schedule;
mod schwarz;
mod usage;

pub use amg::{
    rigid_body_modes, Amg, AmgCapacityError, AmgExport, AmgLevelExport, AmgOptions, AmgStatistics,
    NearNullspace,
};
pub use direct::DirectPreconditioner;
pub use ilu::{Ilu0, Ilu0CapacityError, Ilut, IlutOptions};
pub use jacobi::{BlockJacobi, Jacobi};
pub use schwarz::{
    graph_partition, AdditiveSchwarz, DirectSubdomainSolver, SchwarzOptions, SubdomainSolver,
};
