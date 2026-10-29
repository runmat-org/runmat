use runmat_analysis_core::RandomExcitationSpectrum;

use crate::{
    assembly::AssemblySummary, contracts::FeaRunError, pipeline::modal_basis::ModalBasisArtifact,
};

use super::{
    interval::{
        assemble_described_random_response_intervals, describe_planned, solve_planned_interval,
    },
    plan::RandomResponsePlan,
    ModalRandomResponseResult, RandomResponseIntegrationPolicy,
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
    let description = describe_planned(&plan, &frequency_response, basis, excitation)?;
    // A whole solve is the deterministic fan-in of its canonical interval contributions.
    let intervals = (0..description.interval_count())
        .map(|ordinal| {
            solve_planned_interval(&plan, &frequency_response, basis, excitation, ordinal)
        })
        .collect::<Result<Vec<_>, _>>()?;
    assemble_described_random_response_intervals(&description, intervals)
}
