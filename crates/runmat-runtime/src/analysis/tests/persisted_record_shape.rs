//! Published run records must not grow with mesh size. Response values live once in the
//! run's original fields and persisted states; these records carry scalar evidence only.
//! The exhaustive destructures fail to compile if a field is added, forcing a review of
//! whether the new field scales with the mesh.
use super::*;

const _: fn(StructuralResponseSpectrumResultsData) = |record| {
    let StructuralResponseSpectrumResultsData {
        numerical_receipt: _,
        accuracy: _,
        resource_usage: _,
        publication: _,
    } = record;
};

const _: fn(StructuralRandomResponseResultsData) = |record| {
    let StructuralRandomResponseResultsData {
        numerical_receipt: _,
        accuracy: _,
        resource_usage: _,
        publication: _,
    } = record;
};
