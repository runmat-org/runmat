//! Assemblies: product structure, variants, mates, patterns, fasteners, representations and
//! load levels, BOM and where-used, interference, the index and summary cards.
//!
//! - [`model`]: the `.assembly` document (occurrences, overrides, connectors, mates, patterns,
//!   mirrors, envelopes, publishes, configurations).
//! - [`options`]: option and effectivity axes, `when` conditions and the exact region algebra.
//! - [`validate`]: proof that every legal configuration resolves uniquely.
//! - [`source`]: the lazy design-source seam and R0 summaries.
//! - [`resolve`]: composition of the 150% structure into a 100% occurrence tree.
//! - [`fasteners`]: fastener rules, columnar instance tables, the catalog and gauge seams.
//! - [`representation`]: representations, load levels R0–R3 and working sets.
//! - [`bom`]: configuration-aware BOM and where-used.
//! - [`explode`]: named exploded states (ordered and radial steps, explode lines with jogs).
//! - [`interference`]: interference and clearance requests through the kernel seam.
//! - [`index`]: the assembly JSON index and summary cards.
//! - [`synth`]: synthetic aircraft-scale fixtures for the scale gate.

pub mod bom;
pub mod explode;
pub mod fasteners;
pub mod index;
pub mod interference;
mod cam;
mod mates;
pub mod model;
pub mod options;
mod place;
pub mod polyline_frame;
pub mod representation;
pub mod resolve;
pub mod source;
pub mod synth;
pub mod validate;
pub mod views;

pub use model::*;
pub use options::{
    AxisDecl, AxisDomain, AxisValue, CivilDate, Condition, OptionSpace, OptionsBlock,
};
pub use resolve::{
    resolve, resolve_metered, NodeId, OccNode, ResolveContext, ResolveRequest, ResolvedAssembly,
    VariantSelection,
};
pub use source::{
    Aabb, ConfigKey, ConfiguredSummary, DefinitionSummary, DesignSource, MemorySource,
    PublishedGeometry,
};
