use crate::*;

mod descriptor;
mod inference;
mod new;
#[cfg(test)]
mod tests;

use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&new::ENTRY];
