//! Public field projection for statistical structural procedures.
//!
//! This is the only authority that maps typed response-spectrum and random-response results into
//! the generic analysis-field transport used by paging, plotting, and artifact persistence.

use runmat_analysis_core::AnalysisField;

use crate::{
    contracts::{RandomResponseFieldSelector, ResponseSpectrumFieldSelector},
    solve::dynamics::{ModalRandomResponseResult, ResponseSpectrumFieldRecovery},
};

pub fn response_spectrum_fields(recovery: &ResponseSpectrumFieldRecovery) -> Vec<AnalysisField> {
    let mut fields = Vec::new();
    let translation = recovery.nodal().translation();
    fields.push(AnalysisField::host_f64(
        ResponseSpectrumFieldSelector::Nodal(translation.quantity()).field_id(), translation.quantity().layout(),
        translation.shape().to_vec(),
        translation.values().to_vec(),
    ));
    if let Some(rotation) = recovery.nodal().rotation() {
        fields.push(AnalysisField::host_f64(
            ResponseSpectrumFieldSelector::Nodal(rotation.quantity()).field_id(), rotation.quantity().layout(),
            rotation.shape().to_vec(),
            rotation.values().to_vec(),
        ));
    }
    if let Some(deformation) = recovery.fields() {
        fields.extend(deformation.iter().map(|field| {
            AnalysisField::host_f64(
                ResponseSpectrumFieldSelector::Deformation(field.quantity()).field_id(), field.quantity().layout(),
                field.shape().to_vec(),
                field.values().to_vec(),
            )
        }));
    }
    fields
}

pub fn random_response_fields(result: &ModalRandomResponseResult) -> Vec<AnalysisField> {
    result
        .fields()
        .iter()
        .flat_map(|field| {
            let shape = field.shape().to_vec();
            [
                AnalysisField::host_f64(
                    RandomResponseFieldSelector::root_mean_square(field.quantity()).field_id(), field.quantity().layout(),
                    shape.clone(),
                    field.rms().to_vec(),
                ),
                AnalysisField::host_f64(
                    RandomResponseFieldSelector::variance(field.quantity()).field_id(), field.quantity().layout(),
                    shape,
                    field.variance().to_vec(),
                ),
            ]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        contracts::structural_linear_quantity_for_selector,
        post::fields::StructuralLinearFieldQuantity, StructuralFieldSelector,
    };

    #[test]
    fn statistical_field_ids_are_procedure_and_statistic_specific() {
        assert_eq!(
            ResponseSpectrumFieldSelector::Deformation(StructuralLinearFieldQuantity::Stress)
                .field_id(),
            "structural.response_spectrum.envelope.stress"
        );
        assert_eq!(
            RandomResponseFieldSelector::root_mean_square(
                StructuralLinearFieldQuantity::Displacement
            )
            .field_id(),
            "structural.random_response.rms.displacement"
        );
        assert_eq!(
            RandomResponseFieldSelector::variance(StructuralLinearFieldQuantity::Stress).field_id(),
            "structural.random_response.variance.stress"
        );
        assert_eq!(
            structural_linear_quantity_for_selector(StructuralFieldSelector::VonMises),
            None
        );
    }
}
