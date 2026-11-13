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
    damping_ratio: runmat_types::physical::DampingRatio,
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
            damping_ratio: runmat_types::physical::DampingRatio::new(damping_ratio),
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

    pub(super) fn mode_count(&self) -> usize {
        self.modes.len()
    }

    /// Real part of the modal response covariance density `H S Hᴴ` (row-major, modes × modes).
    ///
    /// Recovered quantities are real linear combinations of modal coordinates, so their spectral
    /// density is `phiᵀ Re(H S Hᴴ) phi`; the imaginary part cancels exactly in that quadratic form.
    pub(super) fn modal_density(&self, frequency_hz: f64) -> Result<Vec<f64>, FeaRunError> {
        let coordinates = self.modal_coordinate_transfers(frequency_hz)?;
        let channel_count = self.spectrum.channels.len();
        let spectral_matrix = interpolate_spectral_matrix(self.spectrum, frequency_hz)?;
        if spectral_matrix.len() != channel_count * channel_count {
            return Err(FeaRunError::Solve(
                "random-response transfer and spectral matrix shapes differ".into(),
            ));
        }
        let mode_count = self.modes.len();
        // Row i of H S: (H S)_{i,b} = Σ_a H_{i,a} S_{a,b}.
        let weighted = coordinates
            .iter()
            .map(|row| {
                (0..channel_count)
                    .map(|column| {
                        row.iter()
                            .enumerate()
                            .fold(Complex::default(), |sum, (channel, value)| {
                                sum + *value * spectral_matrix[channel * channel_count + column]
                            })
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut density = vec![0.0; mode_count * mode_count];
        for row in 0..mode_count {
            for column in row..mode_count {
                let value = weighted[row]
                    .iter()
                    .zip(&coordinates[column])
                    .fold(Complex::default(), |sum, (left, right)| {
                        sum + *left * right.conjugate()
                    });
                density[row * mode_count + column] = value.real;
                density[column * mode_count + row] = value.real;
            }
            let diagonal = density[row * mode_count + row];
            if !diagonal.is_finite() {
                return Err(FeaRunError::Solve(
                    "random-response modal covariance density is not finite".into(),
                ));
            }
            if diagonal < 0.0 {
                let scale = response_density(&coordinates[row], &spectral_matrix)?;
                density[row * mode_count + row] = scale;
            }
        }
        Ok(density)
    }

    /// Projects an integrated modal covariance onto every DOF and recovered field component.
    pub(super) fn project(&self, covariance: &[f64]) -> Result<Vec<f64>, FeaRunError> {
        let mode_count = self.modes.len();
        if covariance.len() != mode_count * mode_count
            || covariance.iter().any(|value| !value.is_finite())
        {
            return Err(FeaRunError::Solve(
                "random-response modal covariance is malformed".into(),
            ));
        }
        let mut variances = Vec::with_capacity(self.layout.total_length);
        let mut shape = vec![0.0; mode_count];
        let quadratic = |shape: &[f64]| -> Result<f64, FeaRunError> {
            let mut value = 0.0;
            let mut scale = 0.0;
            for row in 0..mode_count {
                for column in 0..mode_count {
                    let term = shape[row] * covariance[row * mode_count + column] * shape[column];
                    value += term;
                    scale += term.abs();
                }
            }
            if value < -(64.0 * f64::EPSILON * mode_count as f64 * scale) {
                return Err(FeaRunError::Solve(
                    "random-response modal covariance projects to a negative variance".into(),
                ));
            }
            Ok(value.max(0.0))
        };
        for dof in 0..self.layout.dof_count {
            for (slot, mode) in shape.iter_mut().zip(self.modes) {
                *slot = mode.dof_shape()[dof];
            }
            variances.push(quadratic(&shape)?);
        }
        let field_values = self
            .layout
            .fields
            .iter()
            .map(|field| {
                self.modal_fields
                    .iter()
                    .map(|inventory| {
                        inventory
                            .get(&field.quantity)
                            .map(|(_, values)| values.as_slice())
                            .ok_or_else(|| {
                                FeaRunError::Solve(
                                    "random-response modal field inventory changed".into(),
                                )
                            })
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        for (field, modal_values) in self.layout.fields.iter().zip(&field_values) {
            for component in 0..field.length {
                for (slot, values) in shape.iter_mut().zip(modal_values) {
                    *slot = values[component];
                }
                variances.push(quadratic(&shape)?);
            }
        }
        if variances.len() != self.layout.total_length {
            return Err(FeaRunError::Solve(
                "random-response spectral projection differs from its recovery layout".into(),
            ));
        }
        Ok(variances)
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
                    2.0 * self.damping_ratio.canonical_value()
                        * natural_angular_frequency
                        * angular_frequency,
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
