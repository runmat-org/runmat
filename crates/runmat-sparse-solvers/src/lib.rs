//! Shared, value-free sparse linear algebra for RunMat.
//!
//! One numerical layer serves the finite element solvers and the runtime's sparse builtins:
//!
//! - storage: [`CsrMatrix`] / [`CsrRef`] (compressed sparse rows, real or complex) and
//!   [`BsrMatrix`] for vector degrees of freedom;
//! - threaded kernels with fixed reduction orders ([`par`]), so results are bitwise identical for
//!   every thread count;
//! - direct factorizations ([`direct`]): supernodal Cholesky, LDLᵀ with dynamic regularization,
//!   Bunch–Kaufman LBLᵀ for symmetric indefinite systems, LU and QR on faer, with fill-reducing
//!   orderings ([`ordering`]: AMD and multilevel nested dissection), reusable symbolic analysis and
//!   memory estimates before numeric work;
//! - Krylov methods ([`krylov`]): PCG, MINRES, GMRES(m)/FGMRES and BiCGStab for real and complex
//!   systems. PCG and GMRES are the caller-owned, resumable recurrences shared with the runtime's
//!   `pcg`/`gmres` builtins (caller workspace, typed control, original callback errors); the FEA
//!   reaches them through thin synchronous drivers;
//! - preconditioners ([`precond`]): Jacobi and block Jacobi, ILU(0), ILUT, smoothed-aggregation
//!   algebraic multigrid with near-nullspace modes, auxiliary-space (curl-curl) and additive
//!   Schwarz with pluggable subdomain solvers;
//! - eigensolvers ([`eigen`]): shift-invert block Lanczos with thick restart and Sturm checks, and
//!   LOBPCG, for generalized symmetric problems (modal and buckling forms);
//! - an automatic direct/iterative [`policy`] with explicit overrides.
//!
//! Threading: work runs on the caller's installed Rayon pool (see `runmat-cpu-execution`). The
//! crate never creates a global pool; outside an admitted pool everything runs serially.

pub mod backward_error;
pub mod control;
pub mod csr;
pub mod direct;
pub mod distributed;
pub mod eigen;
pub mod error;
pub mod estimate;
pub mod krylov;
pub mod operator;
pub mod ordering;
pub mod par;
pub mod policy;
pub mod precond;
pub mod scalar;
pub mod bsr;
pub mod dense;
pub mod spgemm;

pub use bsr::BsrMatrix;
pub use control::{Progress, ProgressPhase, SolveControl};
pub use csr::{CsrMatrix, CsrRef, TripletBuilder};
pub use error::{SolverError, SolverResult};
pub use operator::{IdentityPreconditioner, LinearOperator, Preconditioner};
pub use scalar::{c32, c64, Scalar};
pub mod testutil;
