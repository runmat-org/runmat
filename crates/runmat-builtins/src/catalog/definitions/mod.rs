//! Canonical builtin contracts organized by source-language domain.
//!
//! This tree mirrors `runmat-runtime::builtins` at the domain boundary while
//! allowing one file to own a coherent builtin family. Files here contain
//! declarative, runtime-independent contracts only: identities, signatures,
//! diagnostics, inference, placement, linking, and public capability metadata.
//! Executable behavior stays in `runmat-runtime`; it is an implementation of
//! these contracts, not a second metadata source. This separation lets the
//! parser, HIR, compiler, LSP, package tooling, and browser-facing code consume
//! builtin contracts without depending on the runtime or its native backends.
//!
//! Every coherent family owns its local `ENTRIES` slice. Domain modules compose
//! family slices, and the root registry composes domains. Adding a builtin to an
//! existing family therefore changes only its contract file and executable
//! binding. A binding declaration stores only its implementation variant; its
//! complete identity is derived from the owning catalog entry. Runtime
//! implementations declare only the linkage identity required to associate an
//! executable symbol with its catalog entry. Catalog validation rejects missing
//! or unexpected bindings, and implementations must not redeclare contract
//! metadata.

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
