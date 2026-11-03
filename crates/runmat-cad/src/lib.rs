//! RunMat CAD: the kernel-free design layer.
//!
//! - [`assembly`]: product structure, variants and effectivity, mates, patterns, fastener
//!   rules, representations and load levels, BOM, interference, the assembly index.
//! - [`config`]: columnar design tables, change impact and affected-only verify.
//! - [`scope`]: folder-scoped name resolution (no same-folder imports, `use`, `+pkg`).
//! - [`pmi`]: semantic PMI on parts (datums, GD&T, sizes, finishes), resolved by reference.
//! - [`props`]: property schemas, check packs over `runmat-rules` selectors, typed links,
//!   the release gate.
//! - [`drawing`]: sheets, views by reference, associative annotations, tables, HLR seam.
//! - [`model`]: the part document (inputs, sketches, datums, features, part outputs, regions,
//!   publish entries, materials, links, configurations).
//! - [`refs`]: the reference and naming system and the history contract kernels emit.
//! - [`regen`]: dependency analysis, chained-hash keys, the `CadKernel` seam, regeneration and
//!   the fake test kernel.
//! - [`diagnostics`]: root cause vs cascade, codes, typed repairs, span placeholders.
//! - [`index`]: the deterministic JSON projection.
//! - [`units`], [`value`], [`sketch_solver`]: seams to the unit authority, the value phase and
//!   the constraint solver.
//! - [`metering`]: the usage-metering seam (sketch solves, mate solves, regenerations).

pub mod assembly;
pub mod config;
pub mod diagnostics;
pub mod drawing;
pub mod ids;
pub mod index;
pub mod metering;
pub mod model;
pub mod pmi;
pub mod props;
pub mod refs;
pub mod regen;
pub mod scope;
mod serde_rows;
pub mod sketch_solver;
pub mod units;
pub mod value;
