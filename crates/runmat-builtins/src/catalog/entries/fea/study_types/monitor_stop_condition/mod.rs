use crate::*;

mod absolute_change_below;
mod descriptor;
mod greater_than;
mod inference;
mod less_than;
mod none;
#[cfg(test)]
mod tests;

use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &none::ENTRY,
    &less_than::ENTRY,
    &greater_than::ENTRY,
    &absolute_change_below::ENTRY,
];
