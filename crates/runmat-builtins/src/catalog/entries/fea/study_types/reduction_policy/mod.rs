use crate::*;

mod descriptor;
mod inference;
mod new;
mod none;
#[cfg(test)]
mod tests;

use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&none::ENTRY, &new::ENTRY];
