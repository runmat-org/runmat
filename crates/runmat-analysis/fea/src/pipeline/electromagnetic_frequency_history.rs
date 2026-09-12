use runmat_analysis_core::{AnalysisField, ObservationQuantity, ObservationValueShape};

use crate::{
    contracts::{FeaElectromagneticRunResult, FeaRunError},
    results::field::{RecoveredFieldEntityDomain, RecoveredFieldNumericRepresentation},
};

use super::{
    electromagnetic::project_edge_field_to_solid_elements, observation_history::HistoryFieldSeries,
};

const FIELD_MAGNETIC_FLUX_DENSITY: &str = "em.frequency.magnetic_flux_density";
const FIELD_MAGNETIC_FIELD_INTENSITY: &str = "em.frequency.magnetic_field_intensity";
const FIELD_ELECTRIC_CURRENT_DENSITY: &str = "em.frequency.electric_current_density";
const FIELD_ELECTRIC_FIELD: &str = "em.frequency.electric_field";
const FIELD_RESIDUAL: &str = "em.frequency.residual";

pub(super) struct OwnedHistorySeries {
    quantity: ObservationQuantity,
    source_field_id: &'static str,
    value_shape: ObservationValueShape,
    snapshots: Vec<AnalysisField>,
}

impl OwnedHistorySeries {
    pub(super) fn as_history_series(&self) -> HistoryFieldSeries<'_> {
        HistoryFieldSeries {
            quantity: self.quantity,
            source_field_id: self.source_field_id,
            location: runmat_analysis_core::FieldLocation::Element,
            entity_domain: RecoveredFieldEntityDomain::SolidElements,
            value_shape: self.value_shape.clone(),
            numeric_representation: RecoveredFieldNumericRepresentation::ComplexInterleaved,
            snapshots: &self.snapshots,
        }
    }
}

type FieldPair = fn(&FeaElectromagneticRunResult) -> (&AnalysisField, &AnalysisField);

pub(super) fn complex_history_series(
    mesh: &runmat_meshing_core::SolverMeshArtifact,
    runs: &[FeaElectromagneticRunResult],
) -> Result<Vec<OwnedHistorySeries>, FeaRunError> {
    [
        (
            ObservationQuantity::MagneticFluxDensity,
            FIELD_MAGNETIC_FLUX_DENSITY,
            ObservationValueShape::Vector3,
            magnetic_flux_density as FieldPair,
        ),
        (
            ObservationQuantity::MagneticFieldIntensity,
            FIELD_MAGNETIC_FIELD_INTENSITY,
            ObservationValueShape::Vector3,
            magnetic_field as FieldPair,
        ),
        (
            ObservationQuantity::ElectricCurrentDensity,
            FIELD_ELECTRIC_CURRENT_DENSITY,
            ObservationValueShape::Vector3,
            current_density as FieldPair,
        ),
        (
            ObservationQuantity::ElectricField,
            FIELD_ELECTRIC_FIELD,
            ObservationValueShape::Vector3,
            electric_field as FieldPair,
        ),
        (
            ObservationQuantity::Residual,
            FIELD_RESIDUAL,
            ObservationValueShape::Scalar,
            residual as FieldPair,
        ),
    ]
    .into_iter()
    .map(|(quantity, source_field_id, value_shape, fields)| {
        let snapshots = runs
            .iter()
            .map(|run| {
                let (real, imaginary) = fields(run);
                let real = project_edge_field_to_solid_elements(mesh, source_field_id, real)?;
                let imaginary =
                    project_edge_field_to_solid_elements(mesh, source_field_id, imaginary)?;
                interleave_complex(source_field_id, &real, &imaginary)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(OwnedHistorySeries {
            quantity,
            source_field_id,
            value_shape,
            snapshots,
        })
    })
    .collect()
}

fn interleave_complex(
    field_id: &'static str,
    real: &AnalysisField,
    imaginary: &AnalysisField,
) -> Result<AnalysisField, FeaRunError> {
    if real.shape != imaginary.shape {
        return Err(FeaRunError::Observation(format!(
            "complex field pair `{field_id}` has different real and imaginary shapes"
        )));
    }
    let real_values = real.as_host_f64().ok_or_else(|| {
        FeaRunError::Observation(format!(
            "complex field `{field_id}` real part is not host real64"
        ))
    })?;
    let imaginary_values = imaginary.as_host_f64().ok_or_else(|| {
        FeaRunError::Observation(format!(
            "complex field `{field_id}` imaginary part is not host real64"
        ))
    })?;
    if real_values.len() != imaginary_values.len() {
        return Err(FeaRunError::Observation(format!(
            "complex field pair `{field_id}` has different value counts"
        )));
    }
    let mut values = Vec::with_capacity(real_values.len().saturating_mul(2));
    for (&real, &imaginary) in real_values.iter().zip(imaginary_values) {
        values.push(real);
        values.push(imaginary);
    }
    let mut shape = real.shape.clone();
    shape.push(2);
    Ok(AnalysisField::host_f64(field_id, shape, values))
}

fn magnetic_flux_density(run: &FeaElectromagneticRunResult) -> (&AnalysisField, &AnalysisField) {
    (
        &run.magnetic_flux_density_real_field,
        &run.magnetic_flux_density_imag_field,
    )
}

fn magnetic_field(run: &FeaElectromagneticRunResult) -> (&AnalysisField, &AnalysisField) {
    (
        &run.magnetic_field_real_field,
        &run.magnetic_field_imag_field,
    )
}

fn current_density(run: &FeaElectromagneticRunResult) -> (&AnalysisField, &AnalysisField) {
    (
        &run.current_density_real_field,
        &run.current_density_imag_field,
    )
}

fn electric_field(run: &FeaElectromagneticRunResult) -> (&AnalysisField, &AnalysisField) {
    (
        &run.electric_field_real_field,
        &run.electric_field_imag_field,
    )
}

fn residual(run: &FeaElectromagneticRunResult) -> (&AnalysisField, &AnalysisField) {
    (&run.residual_real_field, &run.residual_imag_field)
}
