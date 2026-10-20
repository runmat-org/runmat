//! Published run records must not grow with mesh size. Response values live once in the
//! run's original fields and persisted states; these records carry scalar evidence only.
//! The exhaustive destructures fail to compile if a field is added, forcing a review of
//! whether the new field scales with the mesh.
use super::*;

#[allow(dead_code)]
fn response_spectrum_record_is_scalar_only(record: StructuralResponseSpectrumResultsData) {
    let StructuralResponseSpectrumResultsData {
        numerical_receipt: _,
        accuracy: _,
        resource_usage: _,
        publication: _,
    } = record;
}

#[allow(dead_code)]
fn random_response_record_is_scalar_only(record: StructuralRandomResponseResultsData) {
    let StructuralRandomResponseResultsData {
        numerical_receipt: _,
        accuracy: _,
        resource_usage: _,
        publication: _,
    } = record;
}
