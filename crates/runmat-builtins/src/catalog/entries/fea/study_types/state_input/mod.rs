use crate::*;

mod descriptor;
mod inference;
mod initial_conditions;
mod upstream;
#[cfg(test)]
mod tests;

use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&initial_conditions::ENTRY, &upstream::ENTRY];
