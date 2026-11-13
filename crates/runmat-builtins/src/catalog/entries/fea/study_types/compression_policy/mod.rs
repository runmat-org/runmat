use crate::*;

mod descriptor;
mod error_bounded;
mod inference;
mod lossless;
mod none;
#[cfg(test)]
mod tests;

use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] =
    &[&lossless::ENTRY, &none::ENTRY, &error_bounded::ENTRY];
