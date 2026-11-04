//! The geometry-engine contract for RunMat CAD.
//!
//! A kernel takes one canonical operation ([`op::KernelOp`]: SI values, frames in part
//! coordinates, sketch curves by authored id) over operand bodies it made, and returns its
//! output bodies with a topology summary each plus one [`history::OpHistory`] obeying the
//! contract in [`history`]. Roles and tags come from the canonical operation, never from output
//! order; an operation that cannot report complete history fails with
//! [`op::KernelError::HistoryIncomplete`] instead of guessing.
//!
//! This crate is the one definition. The design layer (`runmat-cad`) re-exports it under its own
//! paths, its test kernel implements it on a grid, and the OCCT backend (`runmat-geometry-io`,
//! feature `occt-native`) implements it on exact B-rep. No kernel type crosses these traits.

pub mod analysis;
pub mod frame;
pub mod history;
pub mod ids;
pub mod names;
pub mod op;
pub mod queries;

pub use analysis::GeometryAnalyzer;
pub use frame::Frame;
pub use ids::SketchEntityId;
pub use op::{BodyHandle, Kernel, KernelError, KernelId, KernelOp, KernelOutput};
pub use queries::ExactQueries;
