use runmat_analysis_core::{
    CoordinateFrame, RandomExcitationQuantity, RandomExcitationSpectrum, SpectralDensityConvention,
};

use crate::{
    assembly::AssemblySummary, contracts::FeaRunError, pipeline::modal_basis::ModalBasisArtifact,
};

use super::{
    quadrature::integrate_breakpoints, random_excitation_digest, recovery::FrequencyResponse,
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
    basis.validate_for(summary)?;
    excitation
        .validate_intrinsic()
        .map_err(FeaRunError::Solve)?;
    policy.validate()?;
    if excitation.convention != SpectralDensityConvention::OneSidedPerHertz {
        return Err(FeaRunError::Solve(
            "random response requires a one-sided density per hertz".into(),
        ));
    }
    if excitation.channels.iter().any(|channel| {
        channel.direction.frame != CoordinateFrame::Global
            || channel.quantity != RandomExcitationQuantity::BaseAcceleration
    }) {
        return Err(FeaRunError::Solve(
            "random response currently requires global base-acceleration channels".into(),
        ));
    }

    let directions = excitation
        .channels
        .iter()
        .map(|channel| normalized_direction(channel.direction.vector))
        .collect::<Result<Vec<_>, _>>()?;
    let total_mass = directions
        .iter()
        .map(|direction| basis.total_free_mass_in_direction(*direction))
        .collect::<Vec<_>>();
    if total_mass
        .iter()
        .any(|mass| !mass.is_finite() || *mass <= 0.0)
    {
        return Err(FeaRunError::Solve(
            "a random-excitation direction has no positive free structural mass".into(),
        ));
    }

    let mut captured_mass = vec![0.0; directions.len()];
    let mut participation_by_mode_channel = Vec::new();
    for mode in basis.modes() {
        if !mode.frequency_hz().is_finite() || mode.frequency_hz() <= 0.0 {
            return Err(FeaRunError::Solve(
                "random response requires strictly positive selected modal frequencies".into(),
            ));
        }
        let participation = directions
            .iter()
            .map(|direction| {
                direction
                    .iter()
                    .zip(mode.directional())
                    .map(|(component, metric)| component * metric.participation_factor())
                    .sum::<f64>()
            })
            .collect::<Vec<_>>();
        for ((captured, factor), _) in captured_mass
            .iter_mut()
            .zip(&participation)
            .zip(&directions)
        {
            *captured += factor * factor * mode.modal_mass();
        }
        participation_by_mode_channel.push(participation);
        if captured_mass
            .iter()
            .zip(&total_mass)
            .all(|(captured, total)| {
                captured / total + 64.0 * f64::EPSILON >= policy.minimum_cumulative_mass_ratio
            })
        {
            break;
        }
    }
    if participation_by_mode_channel.is_empty() {
        return Err(FeaRunError::Solve(
            "random response requires at least one modal basis vector".into(),
        ));
    }
    let captured_mass_ratio_by_channel = captured_mass
        .iter()
        .zip(&total_mass)
        .map(|(captured, total)| captured / total)
        .collect::<Vec<_>>();
    if captured_mass_ratio_by_channel
        .iter()
        .any(|ratio| *ratio + 64.0 * f64::EPSILON < policy.minimum_cumulative_mass_ratio)
    {
        return Err(FeaRunError::Solve(format!(
            "modal basis does not capture the required {:.6} directional mass ratio for every random-excitation channel",
            policy.minimum_cumulative_mass_ratio
        )));
    }

    let selected_mode_count = participation_by_mode_channel.len();
    let modes = &basis.modes()[..selected_mode_count];
    let mut breakpoints = excitation.frequencies_hz.clone();
    let lower = breakpoints[0];
    let upper = *breakpoints.last().expect("validated random spectrum");
    breakpoints.extend(
        modes
            .iter()
            .map(|mode| mode.frequency_hz())
            .filter(|frequency| *frequency > lower && *frequency < upper),
    );
    breakpoints.sort_by(f64::total_cmp);
    breakpoints.dedup_by(|left, right| left.to_bits() == right.to_bits());

    let frequency_response = FrequencyResponse::new(
        summary,
        modes,
        excitation,
        &participation_by_mode_channel,
        policy.modal_damping_ratio,
    )?;
    let integrated = integrate_breakpoints(
        &breakpoints,
        excitation.frequencies_hz.len() - 1,
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
        excitation_digest: random_excitation_digest(excitation)?,
        policy,
        selected_mode_count,
        captured_mass_ratio_by_channel,
        dof_variance,
        fields,
        evidence: integrated.evidence,
    })
}

fn normalized_direction(direction: [f64; 3]) -> Result<[f64; 3], FeaRunError> {
    let norm = direction
        .iter()
        .map(|component| component * component)
        .sum::<f64>()
        .sqrt();
    if !norm.is_finite() || norm <= f64::EPSILON {
        return Err(FeaRunError::Solve(
            "random-excitation direction is not finite and nonzero".into(),
        ));
    }
    Ok(direction.map(|component| component / norm))
}
