use runmat_analysis_core::RandomExcitationSpectrum;

use crate::{
    assembly::AssemblySummary, contracts::FeaRunError, pipeline::modal_basis::ModalBasisArtifact,
};

use super::{
    super::{
        quadrature::compensated_add, ModalRandomResponseResult, ModalRandomResponseResultParts,
        RandomResponseFieldStatistics, RandomResponseIntegrationEvidence,
        RandomResponseIntegrationPolicy,
    },
    describe_modal_random_response, ModalRandomResponseIntervalResult,
    RandomResponsePlanDescription,
};

pub fn assemble_modal_random_response_intervals(
    summary: &AssemblySummary,
    basis: &ModalBasisArtifact,
    excitation: &RandomExcitationSpectrum,
    policy: RandomResponseIntegrationPolicy,
    intervals: Vec<ModalRandomResponseIntervalResult>,
) -> Result<ModalRandomResponseResult, FeaRunError> {
    let description = describe_modal_random_response(summary, basis, excitation, policy)?;
    assemble_described_random_response_intervals(&description, intervals)
}

/// Deterministically assembles canonical interval contributions without requiring model residency.
///
/// This is the placement-neutral fan-in boundary used after local or distributed interval work.
/// The description binds the exact modal basis, excitation, integration policy, breakpoint set,
/// and result layout needed to reject missing, duplicated, substituted, or reordered work.
pub fn assemble_described_random_response_intervals(
    description: &RandomResponsePlanDescription,
    mut intervals: Vec<ModalRandomResponseIntervalResult>,
) -> Result<ModalRandomResponseResult, FeaRunError> {
    description.validate()?;
    intervals.sort_by_key(ModalRandomResponseIntervalResult::ordinal);
    let expected_count = description.interval_count();
    if intervals.len() != expected_count {
        return Err(FeaRunError::Solve(
            "random-response interval assembly is incomplete".into(),
        ));
    }
    for (ordinal, interval) in intervals.iter().enumerate() {
        interval.validate()?;
        let expected_bounds = [
            description.breakpoints_hz()[ordinal],
            description.breakpoints_hz()[ordinal + 1],
        ];
        if interval.ordinal != ordinal
            || interval.basis != description.basis()
            || interval.excitation != *description.excitation()
            || interval.excitation_digest != description.excitation_digest()
            || interval.policy != description.policy()
            || interval.frequency_bounds_hz != expected_bounds
            || interval.selected_mode_count != description.selected_mode_count()
            || interval.captured_mass_ratio_by_channel
                != description.captured_mass_ratio_by_channel()
            || interval.dof_variance.len() != description.physical_dof_count()
            || interval.fields.len() != description.fields().len()
            || interval
                .fields
                .iter()
                .zip(description.fields())
                .any(|(field, expected)| {
                    field.quantity != expected.quantity()
                        || field.shape != expected.shape()
                        || field.variance.len() != expected.component_count()
                })
        {
            return Err(FeaRunError::Solve(
                "random-response interval differs from its canonical assembly plan".into(),
            ));
        }
    }

    let flat_length = description.value_count();
    let mut total = vec![0.0; flat_length];
    let mut compensation = vec![0.0; flat_length];
    for interval in &intervals {
        let mut flat = Vec::with_capacity(flat_length);
        flat.extend_from_slice(&interval.dof_variance);
        for field in &interval.fields {
            flat.extend_from_slice(&field.variance);
        }
        compensated_add(&mut total, &mut compensation, &flat);
    }
    let dof_variance = total[..description.physical_dof_count()].to_vec();
    let mut offset = description.physical_dof_count();
    let fields = description
        .fields()
        .iter()
        .map(|field| {
            let end = offset + field.component_count();
            let variance = total[offset..end].to_vec();
            offset = end;
            let rms = variance.iter().map(|value| value.sqrt()).collect();
            RandomResponseFieldStatistics {
                quantity: field.quantity(),
                shape: field.shape().to_vec(),
                variance,
                rms,
            }
        })
        .collect();
    ModalRandomResponseResult::from_parts(ModalRandomResponseResultParts {
        basis: description.basis(),
        excitation: description.excitation().clone(),
        excitation_digest: description.excitation_digest(),
        policy: description.policy(),
        selected_mode_count: description.selected_mode_count(),
        captured_mass_ratio_by_channel: description.captured_mass_ratio_by_channel().to_vec(),
        dof_variance,
        fields,
        evidence: RandomResponseIntegrationEvidence {
            authored_interval_count: description.authored_interval_count(),
            resonance_split_count: expected_count
                .saturating_sub(description.authored_interval_count()),
            evaluated_frequency_count: intervals
                .iter()
                .map(|interval| interval.evidence.evaluated_frequency_count)
                .sum(),
            maximum_refinement_level_used: intervals
                .iter()
                .map(|interval| interval.evidence.refinement_level)
                .max()
                .unwrap_or(0),
            maximum_estimated_relative_variance_error: intervals
                .iter()
                .map(|interval| interval.evidence.estimated_relative_error)
                .fold(0.0_f64, f64::max),
        },
    })
}
