//! RunMat CAD: the kernel-free design layer.
//!
//! - [`model`]: the part document (inputs, sketches, datums, features, part outputs, regions,
//!   publish entries, materials, links, configurations).
//! - [`refs`]: the reference and naming system and the history contract kernels emit.
//! - [`regen`]: dependency analysis, chained-hash keys, the `CadKernel` seam, regeneration and
//!   the fake test kernel.
//! - [`diagnostics`]: root cause vs cascade, codes, typed repairs, span placeholders.
//! - [`index`]: the deterministic JSON projection.
//! - [`units`], [`value`], [`sketch_solver`]: seams to the unit authority, the value phase and
//!   the constraint solver.

pub mod diagnostics;
pub mod ids;
pub mod index;
pub mod model;
pub mod refs;
pub mod regen;
mod serde_rows;
pub mod sketch_solver;
pub mod units;
pub mod value;
