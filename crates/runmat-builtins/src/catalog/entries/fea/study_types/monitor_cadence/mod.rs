use crate::*;

mod accepted_increments;
mod descriptor;
mod inference;
mod wall_seconds;
#[cfg(test)]
mod tests;

use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] =
    &[&accepted_increments::ENTRY, &wall_seconds::ENTRY];
