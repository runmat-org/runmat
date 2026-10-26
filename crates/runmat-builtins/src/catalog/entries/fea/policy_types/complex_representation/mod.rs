//! Catalog identities of the `fea.ComplexRepresentation` typed policy class.

use crate::*;

mod complex_value;
mod imaginary;
mod magnitude;
mod phase_radians;
mod real;
#[cfg(test)]
mod tests;

pub(super) const ROLE: &str = concat!(
    "A complex representation states how an observation reports values that may be ",
    "complex. Choices other than `real` are valid only for harmonic, prestressed ",
    "harmonic, and random-response procedures, and the storage precision must be complex ",
    "exactly when the representation is `complex`.",
);
pub(super) const STUDY: &str = concat!(
    "Pass the representation as the tenth argument of an observation constructor. ",
    "`fea.Observation.field`, `history`, `probe`, and `reduction` take `(id, quantity, ",
    "target, sourceLocation, outputLocation, recovery, component, frame, sign, ",
    "complexRepresentation, sampling, reduction, storage, purpose, cost)`.",
);
pub(super) const FAQ: BuiltinDocumentationFaq = BuiltinDocumentationFaq {
    question: "How do I get amplitude and phase of a harmonic response?",
    answer: concat!(
        "Declare two observations, one with `magnitude()` and one with `phaseRadians()`, or ",
        "one with `complexValue()` and complex storage.",
    ),
};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &real::FEA_COMPLEX_REPRESENTATION_REAL_CATALOG_ENTRY,
    &imaginary::FEA_COMPLEX_REPRESENTATION_IMAGINARY_CATALOG_ENTRY,
    &magnitude::FEA_COMPLEX_REPRESENTATION_MAGNITUDE_CATALOG_ENTRY,
    &phase_radians::FEA_COMPLEX_REPRESENTATION_PHASE_RADIANS_CATALOG_ENTRY,
    &complex_value::FEA_COMPLEX_REPRESENTATION_COMPLEX_VALUE_CATALOG_ENTRY,
];
