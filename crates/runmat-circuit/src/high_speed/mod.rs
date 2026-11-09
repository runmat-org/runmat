//! High-speed routing: impedance-driven widths, matched lengths and delays.
//!
//! - [`controlled`]: a net class with a target impedance (`impedance` single-ended,
//!   `diff_impedance` differential, both in ohms) gets its track width, and for a pair its gap,
//!   computed per copper layer from the stackup. [`crate::layout::Board::rule_specs`] turns
//!   them into per-layer rules, so DRC, the autorouter, Route and Diff pair all use them.
//! - [`delay`]: each track's and via's propagation delay from the stackup's Dk.
//! - [`groups`]: matched-length groups (`match_group`), compared in length or in delay.
//! - [`cross_section`]: each controlled line's cross-section for the board study's field solve.
//! - [`signals`]: a signal's length and delay across two-pin series parts.
//! - [`pair_tune`]: coupled meanders that tune a differential pair as one.
//! - [`return_path`]: controlled tracks crossing a split or gap in their reference plane.
//! - [`stubs`]: via stubs past a net's connected layers, and back-drills that remove them.
//!
//! The data model leaves room for RF structures (coplanar waveguide with ground, via fences):
//! a layer's geometry is [`crate::impedance::TraceGeometry`], and a structure kind is one more
//! variant of it, with its own closed form and its own cross-section for the field solver.

mod connections;
pub mod controlled;
pub mod cross_section;
pub mod delay;
pub mod groups;
pub mod pair_tune;
pub mod return_path;
pub mod signals;
pub mod stubs;
pub mod tuning;

pub use controlled::{controlled_layers, ControlledLayer, ImpedanceTarget};
