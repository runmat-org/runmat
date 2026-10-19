//! Public deep-learning layer constructors.

use crate::*;

mod contract;
mod documentation;
mod entries;
mod inference;
mod schema;
#[cfg(test)]
mod tests;

pub use contract::*;
pub use entries::*;
pub use schema::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = entries::ENTRIES;
