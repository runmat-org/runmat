use runmat_analysis_core::{
    CoordinateFrame, RandomExcitationQuantity, RandomExcitationSpectrum, SpectralDensityConvention,
};

use crate::{
    assembly::AssemblySummary, contracts::FeaRunError, pipeline::modal_basis::ModalBasisArtifact,
};

use super::{recovery::FrequencyResponse, RandomResponseIntegrationPolicy};

pub(super) struct RandomResponsePlan {
    policy: RandomResponseIntegrationPolicy,
    selected_mode_count: usize,
    captured_mass_ratio_by_channel: Vec<f64>,
    participation_by_mode_channel: Vec<Vec<f64>>,
    breakpoints_hz: Vec<f64>,
    authored_interval_count: usize,
}

impl RandomResponsePlan {
    pub(super) fn new(
        summary: &AssemblySummary,
        basis: &ModalBasisArtifact,
        excitation: &RandomExcitationSpectrum,
        policy: RandomResponseIntegrationPolicy,
    ) -> Result<Self, FeaRunError> {
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
        // Truncation never splits a repeated-frequency cluster: any orthonormal basis of a
        // degenerate eigenspace must give the same response.
        let mut closing_cluster: Option<f64> = None;
        for mode in basis.modes() {
            if let Some(frequency) = closing_cluster {
                if (mode.frequency_hz() - frequency).abs() > 1.0e-8 * frequency {
                    break;
                }
            }
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
            for (captured, factor) in captured_mass.iter_mut().zip(&participation) {
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
                closing_cluster.get_or_insert(mode.frequency_hz());
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
        let mut breakpoints_hz = excitation.frequencies_hz.canonical_values().clone();
        let lower = breakpoints_hz[0];
        let upper = *breakpoints_hz.last().expect("validated random spectrum");
        breakpoints_hz.extend(
            basis.modes()[..selected_mode_count]
                .iter()
                .map(|mode| mode.frequency_hz())
                .filter(|frequency| *frequency > lower && *frequency < upper),
        );
        breakpoints_hz.sort_by(f64::total_cmp);
        breakpoints_hz.dedup_by(|left, right| left.to_bits() == right.to_bits());
        Ok(Self {
            policy,
            selected_mode_count,
            captured_mass_ratio_by_channel,
            participation_by_mode_channel,
            breakpoints_hz,
            authored_interval_count: excitation.frequencies_hz.canonical_values().len() - 1,
        })
    }

    pub(super) const fn policy(&self) -> RandomResponseIntegrationPolicy {
        self.policy
    }

    pub(super) const fn selected_mode_count(&self) -> usize {
        self.selected_mode_count
    }

    pub(super) fn captured_mass_ratio_by_channel(&self) -> &[f64] {
        &self.captured_mass_ratio_by_channel
    }

    pub(super) fn breakpoints_hz(&self) -> &[f64] {
        &self.breakpoints_hz
    }

    pub(super) const fn authored_interval_count(&self) -> usize {
        self.authored_interval_count
    }

    pub(super) fn frequency_response<'a>(
        &'a self,
        summary: &AssemblySummary,
        basis: &'a ModalBasisArtifact,
        excitation: &'a RandomExcitationSpectrum,
    ) -> Result<FrequencyResponse<'a>, FeaRunError> {
        FrequencyResponse::new(
            summary,
            &basis.modes()[..self.selected_mode_count],
            excitation,
            &self.participation_by_mode_channel,
            self.policy.modal_damping_ratio,
        )
    }
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
