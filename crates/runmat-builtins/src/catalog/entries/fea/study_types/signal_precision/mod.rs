use crate::*;

mod complex128;
mod complex64;
mod descriptor;
mod inference;
mod real32;
mod real64;
#[cfg(test)]
mod tests;

use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &real32::ENTRY,
    &real64::ENTRY,
    &complex64::ENTRY,
    &complex128::ENTRY,
];
