use crate::*;

mod configure;
mod descriptor;
mod inference;
#[cfg(test)]
mod tests;

use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&configure::ENTRY];
