use std::collections::BTreeMap;
use std::f64::consts::TAU;

use runmat_analysis_core::{AnalysisFieldValues, RandomExcitationSpectrum};

use crate::{
    assembly::AssemblySummary,
    contracts::FeaRunError,
    pipeline::modal_basis::ModalBasisMode,
    post::{fields::recover_deformation_result_fields, StructuralLinearFieldQuantity},
};

use super::{
    complex::Complex,
    spectrum::{interpolate_spectral_matrix, response_density},
};

type ModalFieldInventory = BTreeMap<StructuralLinearFieldQuantity, (Vec<usize>, Vec<f64>)>;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct FieldLayout {
    pub quantity: StructuralLinearFieldQuantity,
    pub shape: Vec<usize>,
    pub offset: usize,
    pub length: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct ResponseLayout {
    pub dof_count: usize,
    pub fields: Vec<FieldLayout>,
    pub total_length: usize,
}

/// Evaluates spectral response after recovering each modal field exactly once.
///
/// Linear formulation recovery commutes with modal superposition. Retaining recovered signed mode
/// shapes avoids repeating element recovery at every quadrature point while preserving the same
/// complex transfer and cross-channel correlation calculation.
pub(super) struct FrequencyResponse<'a> {
    modes: &'a [ModalBasisMode],
    spectrum: &'a RandomExcitationSpectrum,
    participation_by_mode_channel: &'a [Vec<f64>],
    damping_ratio: f64,
    modal_fields: Vec<ModalFieldInventory>,
    layout: ResponseLayout,
}

impl<'a> FrequencyResponse<'a> {
    pub(super) fn new(
        summary: &AssemblySummary,
        modes: &'a [ModalBasisMode],
        spectrum: &'a RandomExcitationSpectrum,
        participation_by_mode_channel: &'a [Vec<f64>],
        damping_ratio: f64,
    ) -> Result<Self, FeaRunError> {
        let modal_fields = modes
            .iter()
            .map(|mode| recover_mode_fields(summary, mode))
            .collect::<Result<Vec<_>, _>>()?;
        let first = modal_fields.first().ok_or_else(|| {
            FeaRunError::Solve("random-response field recovery requires a selected mode".into())
        })?;
        let mut offset = summary.dof_count;
        let mut fields = Vec::with_capacity(first.len());
        for (quantity, (shape, values)) in first {
            fields.push(FieldLayout {
                quantity: *quantity,
                shape: shape.clone(),
                offset,
                length: values.len(),
            });
            offset += values.len();
        }
        for inventory in modal_fields.iter().skip(1) {
            if inventory.len() != first.len()
                || first.iter().any(|(quantity, (shape, values))| {
                    inventory
                        .get(quantity)
                        .is_none_or(|(candidate_shape, candidate_values)| {
                            candidate_shape != shape || candidate_values.len() != values.len()
                        })
                })
            {
                return Err(FeaRunError::Solve(
                    "linear field recovery layout differs between selected modes".into(),
                ));
            }
        }
        Ok(Self {
            modes,
            spectrum,
            participation_by_mode_channel,
            damping_ratio,
            modal_fields,
            layout: ResponseLayout {
                dof_count: summary.dof_count,
                fields,
                total_length: offset,
            },
        })
    }

    pub(super) const fn layout(&self) -> &ResponseLayout {
        &self.layout
    }

    pub(super) fn evaluate(&self, frequency_hz: f64) -> Result<Vec<f64>, FeaRunError> {
        let coordinates = self.modal_coordinate_transfers(frequency_hz)?;
        let channel_count = self.spectrum.channels.len();
        let spectral_matrix = interpolate_spectral_matrix(self.spectrum, frequency_hz)?;
        let mut densities = Vec::with_capacity(self.layout.total_length);

        for dof in 0..self.layout.dof_count {
            let mut transfer = vec![Complex::default(); channel_count];
            for (mode_index, mode) in self.modes.iter().enumerate() {
                for (channel_index, value) in transfer.iter_mut().enumerate() {
                    *value += coordinates[mode_index][channel_index].scale(mode.dof_shape()[dof]);
                }
            }
            densities.push(response_density(&transfer, &spectral_matrix)?);
        }

        for field in &self.layout.fields {
            for component in 0..field.length {
                let mut transfer = vec![Complex::default(); channel_count];
                for (mode_index, inventory) in self.modal_fields.iter().enumerate() {
                    let (_, values) = inventory.get(&field.quantity).ok_or_else(|| {
                        FeaRunError::Solve("random-response modal field inventory changed".into())
                    })?;
                    for (channel_index, value) in transfer.iter_mut().enumerate() {
                        *value += coordinates[mode_index][channel_index].scale(values[component]);
                    }
                }
                densities.push(response_density(&transfer, &spectral_matrix)?);
            }
        }
        if densities.len() != self.layout.total_length {
            return Err(FeaRunError::Solve(
                "random-response spectral projection differs from its recovery layout".into(),
            ));
        }
        Ok(densities)
    }

    fn modal_coordinate_transfers(
        &self,
        frequency_hz: f64,
    ) -> Result<Vec<Vec<Complex>>, FeaRunError> {
        let angular_frequency = TAU * frequency_hz;
        self.modes
            .iter()
            .enumerate()
            .map(|(mode_index, mode)| {
                let natural_angular_frequency = TAU * mode.frequency_hz();
                let denominator = Complex::new(
                    natural_angular_frequency * natural_angular_frequency
                        - angular_frequency * angular_frequency,
                    2.0 * self.damping_ratio * natural_angular_frequency * angular_frequency,
                );
                let reciprocal = denominator.reciprocal().ok_or_else(|| {
                    FeaRunError::Solve(format!(
                        "mode {} has a singular random-response transfer at {frequency_hz} Hz",
                        mode_index + 1
                    ))
                })?;
                Ok(self.participation_by_mode_channel[mode_index]
                    .iter()
                    .map(|participation| reciprocal.scale(-participation))
                    .collect())
            })
            .collect()
    }
}

fn recover_mode_fields(
    summary: &AssemblySummary,
    mode: &ModalBasisMode,
) -> Result<ModalFieldInventory, FeaRunError> {
    let fields = recover_deformation_result_fields(summary, mode.dof_shape())
        .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
    let mut recovered = BTreeMap::new();
    for quantity in StructuralLinearFieldQuantity::ALL {
        let Some(field) = fields
            .iter()
            .find(|field| field.field_id == quantity.source_field_id() && !field.is_empty())
        else {
            continue;
        };
        recovered.insert(
            quantity,
            (field.shape.clone(), host_values(field)?.to_vec()),
        );
    }
    Ok(recovered)
}

fn host_values(field: &runmat_analysis_core::AnalysisField) -> Result<&[f64], FeaRunError> {
    match &field.values {
        AnalysisFieldValues::HostF64(values)
            if values.len() == field.element_count()
                && values.iter().all(|value| value.is_finite()) =>
        {
            Ok(values)
        }
        AnalysisFieldValues::HostF64(_) => Err(FeaRunError::Solve(
            "random-response field recovery returned malformed host storage".into(),
        )),
        AnalysisFieldValues::DeviceRef(_) => Err(FeaRunError::Solve(
            "random-response field recovery requires materialized field values".into(),
        )),
    }
}
