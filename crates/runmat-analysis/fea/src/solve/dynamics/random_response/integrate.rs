use runmat_analysis_core::RandomExcitationSpectrum;

use crate::{
    assembly::AssemblySummary, contracts::FeaRunError, pipeline::modal_basis::ModalBasisArtifact,
};

use super::{
    plan::RandomResponsePlan, quadrature::integrate_breakpoints, random_excitation_spectrum_digest,
    ModalRandomResponseResult, ModalRandomResponseResultParts, RandomResponseFieldStatistics,
    RandomResponseIntegrationPolicy,
};

/// Solves a zero-mean stationary base-acceleration response in an exact structural modal basis.
///
/// The excitation owns the complete cross-power spectral-density matrix, so channel correlation is
/// never inferred by this solver. Variances are formed from complex linear transfers and integrated
/// over ordinary frequency in hertz; RMS fields are derived only after integration.
pub fn solve_modal_random_response(
    summary: &AssemblySummary,
    basis: &ModalBasisArtifact,
    excitation: &RandomExcitationSpectrum,
    policy: RandomResponseIntegrationPolicy,
) -> Result<ModalRandomResponseResult, FeaRunError> {
    let plan = RandomResponsePlan::new(summary, basis, excitation, policy)?;
    let frequency_response = plan.frequency_response(summary, basis, excitation)?;
    let integrated = integrate_breakpoints(
        plan.breakpoints_hz(),
        plan.authored_interval_count(),
        policy,
        |frequency| frequency_response.evaluate(frequency),
    )?;
    let layout = frequency_response.layout();
    if integrated.values.len() != layout.total_length {
        return Err(FeaRunError::Solve(
            "random-response integrated values do not match their recovery layout".into(),
        ));
    }
    let dof_variance = integrated.values[..layout.dof_count].to_vec();
    let fields = layout
        .fields
        .iter()
        .map(|field| {
            let variance = integrated.values[field.offset..field.offset + field.length].to_vec();
            let rms = variance.iter().map(|value| value.sqrt()).collect();
            RandomResponseFieldStatistics {
                quantity: field.quantity,
                shape: field.shape.clone(),
                variance,
                rms,
            }
        })
        .collect();
    ModalRandomResponseResult::from_parts(ModalRandomResponseResultParts {
        basis: basis.identity(),
        excitation: excitation.excitation_id.clone(),
        excitation_digest: random_excitation_spectrum_digest(excitation)?,
        policy: plan.policy(),
        selected_mode_count: plan.selected_mode_count(),
        captured_mass_ratio_by_channel: plan.captured_mass_ratio_by_channel().to_vec(),
        dof_variance,
        fields,
        evidence: integrated.evidence,
    })
}
