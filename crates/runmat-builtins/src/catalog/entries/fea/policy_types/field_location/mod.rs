//! Catalog identities of the `fea.FieldLocation` typed policy class.

use crate::*;

mod cell;
mod connector;
mod constraint;
mod edge;
mod element;
mod face;
mod global_location;
mod integration_point;
mod interface;
mod mode;
mod node;
mod point;
mod reference_point;
#[cfg(test)]
mod tests;

pub(super) const ROLE: &str = concat!(
    "A field location says where a value lives on the discretization. Each observation ",
    "names two: the source location where the solver natively computes the quantity and ",
    "the output location where it is reported. When they differ, the observation's ",
    "recovery policy states how values move between them.",
);
pub(super) const STUDY: &str = concat!(
    "Pass locations as the fourth (source) and fifth (output) arguments of an observation",
    " constructor. `fea.Observation.field`, `history`, `probe`, and `reduction` take ",
    "`(id, quantity, target, sourceLocation, outputLocation, recovery, component, frame, ",
    "sign, complexRepresentation, sampling, reduction, storage, purpose, cost)`.",
);
pub(super) const FAQ: BuiltinDocumentationFaq = BuiltinDocumentationFaq {
    question: "What if source and output locations are the same?",
    answer: concat!(
        "Then the recovery policy must be `fea.RecoveryPolicy.native()`; any other recovery ",
        "fails study validation.",
    ),
};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &node::FEA_FIELD_LOCATION_NODE_CATALOG_ENTRY,
    &point::FEA_FIELD_LOCATION_POINT_CATALOG_ENTRY,
    &cell::FEA_FIELD_LOCATION_CELL_CATALOG_ENTRY,
    &element::FEA_FIELD_LOCATION_ELEMENT_CATALOG_ENTRY,
    &integration_point::FEA_FIELD_LOCATION_INTEGRATION_POINT_CATALOG_ENTRY,
    &face::FEA_FIELD_LOCATION_FACE_CATALOG_ENTRY,
    &edge::FEA_FIELD_LOCATION_EDGE_CATALOG_ENTRY,
    &interface::FEA_FIELD_LOCATION_INTERFACE_CATALOG_ENTRY,
    &connector::FEA_FIELD_LOCATION_CONNECTOR_CATALOG_ENTRY,
    &constraint::FEA_FIELD_LOCATION_CONSTRAINT_CATALOG_ENTRY,
    &reference_point::FEA_FIELD_LOCATION_REFERENCE_POINT_CATALOG_ENTRY,
    &mode::FEA_FIELD_LOCATION_MODE_CATALOG_ENTRY,
    &global_location::FEA_FIELD_LOCATION_GLOBAL_LOCATION_CATALOG_ENTRY,
];
