//! Canonical builtin entries organized by source-language domain.
//!
//! This private tree follows the public builtin taxonomy at the domain boundary.
//! A family owns its runtime-independent entry here: identity, signatures,
//! diagnostics, inference, placement, linking, capabilities, documentation,
//! examples, and evidence. The crate root re-exports those declarations so
//! consumers never depend on this physical layout.
//!
//! Executable behavior stays in `runmat-runtime`. Runtime modules implement and
//! bind these entries; they do not own another copy of the public contract. The
//! dependency boundary lets the parser, HIR, compiler, LSP, package tooling,
//! and browser-facing code consume builtin contracts without linking the
//! runtime or its native backends.
//!
//! Every coherent family owns its local `ENTRIES` slice. Domain modules compose
//! family slices, and the root registry composes domains. A family directory may
//! split contract assembly and substantial public documentation into focused
//! sibling modules, but it must not introduce a second identity list or a
//! name-selected metadata join. Runtime bindings retain only the implementation
//! variant and the spelling needed for their stable native symbol. Catalog
//! validation rejects missing or unexpected bindings, and catalog-backed
//! implementations must not redeclare contract metadata.

mod acceleration;
mod aggregate;
mod array;
mod introspection;
mod math;
mod parallel;

pub use acceleration::*;
pub use aggregate::*;
pub use array::*;
pub use introspection::*;
pub use math::*;
pub use parallel::*;

pub(super) const DOMAIN_ENTRY_GROUPS: &[&[&[&crate::BuiltinCatalogEntry]]] = &[
    acceleration::ENTRY_GROUPS,
    aggregate::ENTRY_GROUPS,
    array::ENTRY_GROUPS,
    introspection::ENTRY_GROUPS,
    math::ENTRY_GROUPS,
    parallel::ENTRY_GROUPS,
];
