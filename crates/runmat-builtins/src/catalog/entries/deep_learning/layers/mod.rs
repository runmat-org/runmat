//! Public deep-learning layer constructors.

use crate::*;

mod contract;
mod documentation;
mod entries;
mod inference;
#[cfg(test)]
mod tests;

pub use contract::*;
pub use entries::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = entries::ENTRIES;
