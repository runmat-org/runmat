//! Signed modal and static-correction recovery for linear deformation envelopes.

use std::{collections::BTreeMap, f64::consts::TAU};

use runmat_analysis_core::{AnalysisField, AnalysisFieldValues, ModalTruncationCorrection};

use crate::{
    algebra::apply_m,
    assembly::AssemblySummary,
    contracts::FeaRunError,
    pipeline::{homogeneous_static::HomogeneousStaticSolver, modal_basis::ModalBasisArtifact},
    post::fields::{recover_deformation_result_fields, StructuralLinearFieldQuantity},
    solve::preconditioner::SpdPreconditionerKind,
};

use super::has_missing_mass_correction;
use super::{
    super::{
        combine::{
            combine_directional_responses, combine_modal_responses,
            combine_periodic_and_rigid_responses,
        },
        validation::vectors_close,
        ModalResponseSpectrumContribution, ModalResponseSpectrumModeContribution,
        ModalResponseSpectrumModeKind, ModalResponseSpectrumResult,
        ResponseSpectrumDisplacementConversion,
    },
    ResponseSpectrumLinearField,
};

pub(super) fn recover_deformation_fields(
    summary: &AssemblySummary,
    basis: &ModalBasisArtifact,
    result: &ModalResponseSpectrumResult,
    conversion: ResponseSpectrumDisplacementConversion,
) -> Result<
    (
        Vec<ResponseSpectrumLinearField>,
        Option<SpdPreconditionerKind>,
    ),
    FeaRunError,
> {
    let static_solver = has_missing_mass_correction(result)
        .then(|| HomogeneousStaticSolver::new(summary))
        .transpose()?;
    let directional = result
        .contributions()
        .iter()
        .map(|contribution| {
            recover_direction(
                summary,
                basis,
                contribution,
                conversion,
                static_solver.as_ref(),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let first = directional.first().ok_or_else(|| {
        FeaRunError::Solve("response-spectrum field recovery has no directions".into())
    })?;
    let mut fields = Vec::with_capacity(first.len());
    for quantity in StructuralLinearFieldQuantity::ALL {
        let Some(reference) = first.get(&quantity) else {
            continue;
        };
        let entries = directional
            .iter()
            .map(|direction| {
                direction.get(&quantity).ok_or_else(|| {
                    FeaRunError::Solve(
                        "response-spectrum directions produced different linear field inventories"
                            .into(),
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if entries.iter().any(|field| field.shape != reference.shape) {
            return Err(FeaRunError::Solve(
                "response-spectrum directions produced incompatible linear field shapes".into(),
            ));
        }
        let values = combine_directional_responses(
            &entries
                .iter()
                .map(|field| field.values.as_slice())
                .collect::<Vec<_>>(),
            result.directional_combination(),
        )?;
        fields.push(ResponseSpectrumLinearField {
            quantity,
            shape: reference.shape.clone(),
            values,
        });
    }
    if directional
        .iter()
        .any(|direction| direction.len() != fields.len())
    {
        return Err(FeaRunError::Solve(
            "response-spectrum directions produced different linear field inventories".into(),
        ));
    }
    Ok((
        fields,
        static_solver
            .as_ref()
            .map(HomogeneousStaticSolver::preconditioner_kind),
    ))
}

fn recover_direction(
    summary: &AssemblySummary,
    basis: &ModalBasisArtifact,
    contribution: &ModalResponseSpectrumContribution,
    conversion: ResponseSpectrumDisplacementConversion,
    static_solver: Option<&HomogeneousStaticSolver>,
) -> Result<BTreeMap<StructuralLinearFieldQuantity, ResponseSpectrumLinearField>, FeaRunError> {
    let mode_fields = contribution
        .modes
        .iter()
        .map(|mode| {
            let shape = mode_shape(basis, contribution, mode)?;
            let scale = displacement_scale(mode, conversion)?;
            let displacement = shape.iter().map(|value| scale * value).collect::<Vec<_>>();
            recover_linear_fields(summary, &displacement)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let correction_fields =
        recover_missing_mass_correction_fields(summary, contribution, conversion, static_solver)?;
    let first = mode_fields.first().ok_or_else(|| {
        FeaRunError::Solve("response-spectrum direction selected no modal contributions".into())
    })?;
    let frequencies = contribution
        .modes
        .iter()
        .map(|mode| mode.frequency_hz)
        .collect::<Vec<_>>();
    let mut combined = BTreeMap::new();
    for quantity in StructuralLinearFieldQuantity::ALL {
        let Some(reference) = first.get(&quantity) else {
            continue;
        };
        let entries = mode_fields
            .iter()
            .map(|fields| {
                fields.get(&quantity).ok_or_else(|| {
                    FeaRunError::Solve(
                        "modal contributions produced different linear field inventories".into(),
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if entries.iter().any(|field| field.shape != reference.shape) {
            return Err(FeaRunError::Solve(
                "modal contributions produced incompatible linear field shapes".into(),
            ));
        }
        let correction = correction_fields
            .as_ref()
            .map(|fields| {
                fields.get(&quantity).ok_or_else(|| {
                    FeaRunError::Solve(
                        "missing-mass correction produced a different linear field inventory"
                            .into(),
                    )
                })
            })
            .transpose()?;
        if correction.is_some_and(|field| field.shape != reference.shape) {
            return Err(FeaRunError::Solve(
                "missing-mass correction produced an incompatible linear field shape".into(),
            ));
        }
        let vectors = entries
            .iter()
            .map(|field| field.values.clone())
            .collect::<Vec<_>>();
        let (periodic, rigid) = if contribution.rigid_response_policy
            == runmat_analysis_core::RigidResponsePolicy::Disabled
        {
            (
                combine_modal_responses(
                    &vectors,
                    &frequencies,
                    contribution.damping_ratio.canonical_value(),
                    contribution.modal_combination,
                )?,
                vec![0.0; reference.values.len()],
            )
        } else {
            combine_periodic_and_rigid_responses(
                &vectors,
                &frequencies,
                &contribution.rigid_coefficients,
                contribution.damping_ratio.canonical_value(),
                contribution.modal_combination,
            )?
        };
        let correction = correction
            .map(|field| field.values.as_slice())
            .unwrap_or(&[]);
        let values = periodic
            .iter()
            .zip(rigid)
            .enumerate()
            .map(|(index, (periodic, rigid))| {
                let correction = correction.get(index).copied().unwrap_or(0.0);
                let in_phase = rigid + correction;
                (periodic * periodic + in_phase * in_phase).sqrt()
            })
            .collect();
        combined.insert(
            quantity,
            ResponseSpectrumLinearField {
                quantity,
                shape: reference.shape.clone(),
                values,
            },
        );
    }
    if mode_fields
        .iter()
        .any(|fields| fields.len() != combined.len())
        || correction_fields
            .as_ref()
            .is_some_and(|fields| fields.len() != combined.len())
    {
        return Err(FeaRunError::Solve(
            "modal or missing-mass contributions produced different linear field inventories"
                .into(),
        ));
    }
    Ok(combined)
}

fn recover_missing_mass_correction_fields(
    summary: &AssemblySummary,
    contribution: &ModalResponseSpectrumContribution,
    conversion: ResponseSpectrumDisplacementConversion,
    static_solver: Option<&HomogeneousStaticSolver>,
) -> Result<Option<BTreeMap<StructuralLinearFieldQuantity, ResponseSpectrumLinearField>>, FeaRunError>
{
    if !matches!(
        contribution.truncation.correction,
        ModalTruncationCorrection::MissingMass { .. }
    ) {
        return Ok(None);
    }
    if conversion != ResponseSpectrumDisplacementConversion::DivideByAngularFrequencySquared {
        return Err(FeaRunError::Solve(
            "missing-mass deformation recovery requires pseudo-acceleration ordinates".into(),
        ));
    }
    let static_solver = static_solver.ok_or_else(|| {
        FeaRunError::Solve("missing-mass deformation recovery has no static solver".into())
    })?;
    let correction = &contribution.truncation.correction_response;
    if correction.len() != summary.dof_count {
        return Err(FeaRunError::Solve(
            "missing-mass correction has the wrong physical DOF count".into(),
        ));
    }
    let displacement = if correction.iter().all(|value| *value == 0.0) {
        vec![0.0; summary.dof_count]
    } else {
        let inertial_load = apply_m(&summary.operator, correction);
        static_solver.solve(&inertial_load)?.displacement
    };
    recover_linear_fields(summary, &displacement).map(Some)
}

fn displacement_scale(
    mode: &ModalResponseSpectrumModeContribution,
    conversion: ResponseSpectrumDisplacementConversion,
) -> Result<f64, FeaRunError> {
    let angular_frequency = TAU * mode.frequency_hz;
    let divisor = match conversion {
        ResponseSpectrumDisplacementConversion::Direct => 1.0,
        ResponseSpectrumDisplacementConversion::DivideByAngularFrequency => angular_frequency,
        ResponseSpectrumDisplacementConversion::DivideByAngularFrequencySquared => {
            angular_frequency * angular_frequency
        }
    };
    let scale = mode.generalized_peak / divisor;
    if !scale.is_finite() {
        return Err(FeaRunError::Solve(
            "response-spectrum modal displacement conversion is nonfinite".into(),
        ));
    }
    Ok(scale)
}

fn mode_shape<'a>(
    basis: &'a ModalBasisArtifact,
    contribution: &ModalResponseSpectrumContribution,
    mode: &ModalResponseSpectrumModeContribution,
) -> Result<&'a [f64], FeaRunError> {
    match mode.kind {
        ModalResponseSpectrumModeKind::Eigenmode => basis
            .modes()
            .get(mode.mode_index.checked_sub(1).ok_or_else(|| {
                FeaRunError::Solve("response-spectrum eigenmode index is zero".into())
            })?)
            .map(|basis_mode| basis_mode.dof_shape())
            .ok_or_else(|| FeaRunError::Solve("response-spectrum eigenmode is absent".into())),
        ModalResponseSpectrumModeKind::ResidualVector => basis
            .residual_vectors()
            .iter()
            .find(|residual| vectors_close(&residual.direction(), &contribution.direction))
            .map(|residual| residual.dof_shape())
            .ok_or_else(|| {
                FeaRunError::Solve(
                    "response-spectrum residual contribution has no matching basis vector".into(),
                )
            }),
    }
}

fn recover_linear_fields(
    summary: &AssemblySummary,
    displacement: &[f64],
) -> Result<BTreeMap<StructuralLinearFieldQuantity, ResponseSpectrumLinearField>, FeaRunError> {
    let recovered = recover_deformation_result_fields(summary, displacement)
        .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
    let mut fields = BTreeMap::new();
    for quantity in StructuralLinearFieldQuantity::ALL {
        let Some(field) = recovered
            .iter()
            .find(|field| field.field_id == quantity.source_field_id() && !field.is_empty())
        else {
            continue;
        };
        let values = host_values(field)?;
        fields.insert(
            quantity,
            ResponseSpectrumLinearField {
                quantity,
                shape: field.shape.clone(),
                values,
            },
        );
    }
    Ok(fields)
}

fn host_values(field: &AnalysisField) -> Result<Vec<f64>, FeaRunError> {
    match &field.values {
        AnalysisFieldValues::HostF64(values) => {
            if values.len() != field.element_count()
                || values.iter().any(|value| !value.is_finite())
            {
                return Err(FeaRunError::Solve(
                    "linear field recovery returned malformed host storage".into(),
                ));
            }
            Ok(values.clone())
        }
        AnalysisFieldValues::DeviceRef(_) => Err(FeaRunError::Solve(
            "response-spectrum linear recovery requires materialized field values".into(),
        )),
    }
}
