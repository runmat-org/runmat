//! `runmat-constraint`: the 2D sketch and 3D mate constraint solver for RunMat CAD.
//!
//! # Model
//!
//! A [`System`] holds scalar variables ([`VarId`]), entities that own them ([`EntityId`]: 2D
//! points, lines, circles, arcs, ellipses, B-splines; rigid bodies and mate connectors) and
//! constraints ([`ConstraintId`]). Constraints are validated when added. Every constraint is a
//! residual function written once over a generic scalar; forward-mode automatic
//! differentiation ([`ad`]) gives its exact Jacobian block.
//!
//! # Solve
//!
//! [`solve`] splits the system into connected components, splits each component by a
//! Dulmage–Mendelsohn decomposition into an over-determined block, square blocks in dependency
//! order and an under-determined block, and solves each with damped minimum-norm Newton steps
//! (sparse LDLᵀ of `J W Jᵀ + μI`), falling back to Levenberg–Marquardt. A component whose
//! blocks fail is solved whole. [`drag`] adds a soft target for chosen variables.
//!
//! # Diagnose
//!
//! At the solution, a Givens QR of the row-normalised Jacobian, processed in constraint
//! order, gives the rank (so the DOF), the constraints that are linearly dependent on earlier
//! ones (redundant if the system is consistent, conflicting if not, each with the constraints
//! it depends on) and a null-space basis for per-entity DOF. See [`SolveReport`].
//!
//! # Determinism
//!
//! No hash maps, no threads, fixed iteration orders: the same system and options give
//! bit-identical values, iteration counts and diagnostics.
//!
//! # Example
//!
//! ```
//! use runmat_constraint::*;
//! let mut s = System::new();
//! let a = s.point2(0.0, 0.0);
//! let b = s.point2(9.0, 1.0);
//! let l = s.line(a, b).unwrap();
//! s.constrain(Constraint::Fix(a)).unwrap();
//! s.constrain(Constraint::Horizontal(l)).unwrap();
//! s.constrain(Constraint::Length { line: l, value: 10.0 }).unwrap();
//! let report = solve(&s, &SolveOptions::default());
//! assert_eq!(report.status, Status::WellDefined);
//! assert!((report.values[s.entity_vars(b)[0].index()] - 10.0).abs() < 1e-9);
//! ```

pub mod ad;
mod decompose;
mod ids;
mod linalg;
mod report;
mod residual;
mod solver;
mod system;

pub use ids::{ConstraintId, EntityId, VarId};
pub use report::{
    ComponentReport, DependentConstraint, DragReport, EntityDof, Measurement, Method, SolveOptions, SolveReport,
    Status, WarmStart, WarmStartReport,
};
pub use solver::{drag, solve, DragRequest};
pub use system::{
    AngleMode, BuildError, Constraint, ConstraintKindRecord, ConstraintRecord, CurveEnd, DistanceMode, Entity,
    EntityKind, Frame, InternalKind, Mate, System, TangentKind, Target, VarFlag, VarRole, Variable,
    MAX_SPLINE_CONTROL_POINTS,
};
