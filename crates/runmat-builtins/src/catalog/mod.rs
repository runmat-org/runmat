//! Canonical builtin contracts organized by source-language domain.
//!
//! Each family owns its runtime-independent identity, signatures, diagnostics,
//! inference, placement, linking, capabilities, documentation, examples, and
//! evidence. This module re-exports those declarations so consumers do not
//! depend on the catalog's physical layout.
//!
//! Executable behavior stays in `runmat-runtime`. Runtime modules bind catalog
//! entries to implementations without creating another copy of the public
//! contract. This boundary lets the parser, HIR, compiler, LSP, package tools,
//! and browser code consume builtin contracts without linking the runtime or
//! its native backends.
//!
//! Families expose their local entry groups, and generated composition modules
//! assemble those groups into the root registry. Runtime bindings retain only
//! implementation authority and stable native symbols; catalog validation
//! rejects missing, unexpected, or duplicated bindings.

mod alias;
mod aliases;
mod callable;
mod constant;
mod contract;
mod descriptor;
mod documentation;
mod entries;
mod entry;
mod extension;
mod fingerprint;
mod inference;
mod integer;
mod link;
mod placement;
mod provenance;
mod registry;
mod validation;

#[cfg(test)]
mod tests;

pub use alias::*;
pub use callable::*;
pub use constant::*;
pub use contract::*;
pub use descriptor::*;
pub use documentation::*;
pub use entries::*;
pub use entry::*;
pub use extension::*;
pub use fingerprint::*;
pub use inference::*;
pub use integer::*;
pub use link::*;
pub use placement::*;
pub use provenance::*;
pub use registry::*;
pub use validation::*;
