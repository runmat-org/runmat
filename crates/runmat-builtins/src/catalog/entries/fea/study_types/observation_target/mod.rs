use crate::*;

mod boundary;
mod connector;
mod constraint;
mod descriptor;
mod inference;
mod interface;
mod path;
mod point;
mod port;
mod volume;
mod whole_model;
#[cfg(test)]
mod tests;

use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &boundary::ENTRY,
    &volume::ENTRY,
    &point::ENTRY,
    &path::ENTRY,
    &interface::ENTRY,
    &port::ENTRY,
    &connector::ENTRY,
    &constraint::ENTRY,
    &whole_model::ENTRY,
];
