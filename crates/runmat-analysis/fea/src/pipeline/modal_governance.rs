use serde::{Deserialize, Serialize};

use crate::contracts::FeaRunError;

pub const DEFAULT_EIGENSPACE_RELATIVE_TOLERANCE: f64 = 1.0e-6;

mod convergence;
pub use convergence::ModalConvergencePolicy;

/// Governs the retained modal subspace.
///
/// `requested_mode_count` is the desired cut. Fewer modes are retained only when the constrained
/// operator has fewer modes available; more are retained when the cut lands inside a repeated or
/// numerically clustered eigenspace.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModalTruncationPolicy {
    requested_mode_count: usize,
    eigenspace_relative_tolerance: f64,
}

impl ModalTruncationPolicy {
    pub fn requested(requested_mode_count: usize) -> Self {
        Self {
            requested_mode_count,
            eigenspace_relative_tolerance: DEFAULT_EIGENSPACE_RELATIVE_TOLERANCE,
        }
    }

    pub fn with_eigenspace_relative_tolerance(mut self, tolerance: f64) -> Self {
        self.eigenspace_relative_tolerance = tolerance;
        self
    }

    pub const fn requested_mode_count(self) -> usize {
        self.requested_mode_count
    }

    pub const fn eigenspace_relative_tolerance(self) -> f64 {
        self.eigenspace_relative_tolerance
    }

    pub(crate) fn validate(self) -> Result<(), FeaRunError> {
        if self.requested_mode_count == 0 {
            return Err(FeaRunError::InvalidModel(
                "modal truncation requires at least one mode".into(),
            ));
        }
        if !self.eigenspace_relative_tolerance.is_finite()
            || self.eigenspace_relative_tolerance <= 0.0
            || self.eigenspace_relative_tolerance >= 1.0
        {
            return Err(FeaRunError::InvalidModel(
                "modal eigenspace relative tolerance must be finite and lie strictly between zero and one"
                    .into(),
            ));
        }
        Ok(())
    }
}

impl Default for ModalTruncationPolicy {
    fn default() -> Self {
        Self::requested(3)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// Quality evidence for one invariant modal subspace.
///
/// Individual vectors remain a deterministic computational basis. For a repeated eigenspace this
/// record governs the complete subspace and deliberately does not claim that any one rotated basis
/// inside it is physically unique.
pub struct ModalEigenspaceEvidence {
    first_mode_index: usize,
    mode_count: usize,
    minimum_frequency_hz: f64,
    maximum_frequency_hz: f64,
    maximum_residual_norm: f64,
    maximum_mass_orthogonality_error: f64,
}

impl ModalEigenspaceEvidence {
    pub const fn first_mode_index(&self) -> usize {
        self.first_mode_index
    }

    pub const fn mode_count(&self) -> usize {
        self.mode_count
    }

    pub const fn minimum_frequency_hz(&self) -> f64 {
        self.minimum_frequency_hz
    }

    pub const fn maximum_frequency_hz(&self) -> f64 {
        self.maximum_frequency_hz
    }

    pub const fn maximum_residual_norm(&self) -> f64 {
        self.maximum_residual_norm
    }

    pub const fn maximum_mass_orthogonality_error(&self) -> f64 {
        self.maximum_mass_orthogonality_error
    }

    pub const fn is_repeated(&self) -> bool {
        self.mode_count > 1
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModalTruncationEvidence {
    policy: ModalTruncationPolicy,
    available_mode_count: usize,
    retained_mode_count: usize,
    boundary_probe_count: usize,
    boundary_probe_frequency_hz: Option<f64>,
    expanded_for_eigenspace: bool,
    complete_boundary_eigenspace: bool,
}

impl ModalTruncationEvidence {
    pub const fn policy(&self) -> ModalTruncationPolicy {
        self.policy
    }

    pub const fn available_mode_count(&self) -> usize {
        self.available_mode_count
    }

    pub const fn retained_mode_count(&self) -> usize {
        self.retained_mode_count
    }

    pub const fn boundary_probe_count(&self) -> usize {
        self.boundary_probe_count
    }

    pub const fn boundary_probe_frequency_hz(&self) -> Option<f64> {
        self.boundary_probe_frequency_hz
    }

    pub const fn expanded_for_eigenspace(&self) -> bool {
        self.expanded_for_eigenspace
    }

    pub const fn complete_boundary_eigenspace(&self) -> bool {
        self.complete_boundary_eigenspace
    }
}

pub(crate) fn build_modal_governance(
    policy: ModalTruncationPolicy,
    frequencies_hz: &[f64],
    residual_norms: &[f64],
    mass_orthogonality: &[f64],
    available_mode_count: usize,
    boundary_probe_count: usize,
    boundary_probe_frequency_hz: Option<f64>,
) -> Result<(ModalTruncationEvidence, Vec<ModalEigenspaceEvidence>), FeaRunError> {
    policy.validate()?;
    let count = frequencies_hz.len();
    if count < policy.requested_mode_count.min(available_mode_count)
        || count > available_mode_count
        || residual_norms.len() != count
        || mass_orthogonality.len() != count.saturating_mul(count)
    {
        return Err(FeaRunError::Assembly(
            "modal solve did not establish a complete truncation-boundary eigenspace".into(),
        ));
    }
    let complete_boundary_eigenspace = if count == available_mode_count {
        boundary_probe_count == 0 && boundary_probe_frequency_hz.is_none()
    } else {
        boundary_probe_count > 0
            && boundary_probe_count <= available_mode_count - count
            && boundary_probe_frequency_hz.is_some_and(|probe| {
                probe.is_finite()
                    && probe >= frequencies_hz[count - 1]
                    && relative_frequency_separation(frequencies_hz[count - 1], probe)
                        > policy.eigenspace_relative_tolerance
            })
    };
    if !complete_boundary_eigenspace {
        return Err(FeaRunError::Assembly(
            format!(
                "modal solve did not establish a complete truncation-boundary eigenspace: retained_mode_count={count} available_mode_count={available_mode_count} boundary_probe_frequency_hz={boundary_probe_frequency_hz:?}"
            ),
        ));
    }

    let ranges = eigenspace_ranges(frequencies_hz, policy.eigenspace_relative_tolerance);
    let eigenspaces = ranges
        .into_iter()
        .map(|(start, end)| {
            let maximum_mass_orthogonality_error = (start..end)
                .flat_map(|row| (start..end).map(move |column| (row, column)))
                .filter(|(row, column)| row != column)
                .map(|(row, column)| mass_orthogonality[row * count + column].abs())
                .fold(0.0_f64, f64::max);
            ModalEigenspaceEvidence {
                first_mode_index: start + 1,
                mode_count: end - start,
                minimum_frequency_hz: frequencies_hz[start],
                maximum_frequency_hz: frequencies_hz[end - 1],
                maximum_residual_norm: residual_norms[start..end]
                    .iter()
                    .copied()
                    .fold(0.0_f64, f64::max),
                maximum_mass_orthogonality_error,
            }
        })
        .collect();
    Ok((
        ModalTruncationEvidence {
            policy,
            available_mode_count,
            retained_mode_count: count,
            boundary_probe_count,
            boundary_probe_frequency_hz,
            expanded_for_eigenspace: count > policy.requested_mode_count,
            complete_boundary_eigenspace,
        },
        eigenspaces,
    ))
}

pub(crate) fn relative_frequency_separation(left: f64, right: f64) -> f64 {
    (right - left).abs() / left.abs().max(right.abs()).max(1.0e-12)
}

fn eigenspace_ranges(frequencies_hz: &[f64], tolerance: f64) -> Vec<(usize, usize)> {
    if frequencies_hz.is_empty() {
        return Vec::new();
    }
    let mut ranges = Vec::new();
    let mut start = 0;
    for index in 1..frequencies_hz.len() {
        if relative_frequency_separation(frequencies_hz[index - 1], frequencies_hz[index])
            > tolerance
        {
            ranges.push((start, index));
            start = index;
        }
    }
    ranges.push((start, frequencies_hz.len()));
    ranges
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convergence_policy_rejects_unbounded_or_empty_controls() {
        assert!(ModalConvergencePolicy::production()
            .with_algebraic_tolerance(f64::NAN)
            .validate()
            .is_err());
        assert!(ModalConvergencePolicy::production()
            .with_algebraic_tolerance(1.0)
            .validate()
            .is_err());
        assert!(ModalConvergencePolicy::production()
            .with_maximum_subspace_iterations(0)
            .validate()
            .is_err());
    }

    #[test]
    fn rejects_an_unproven_truncation_boundary() {
        let error = build_modal_governance(
            ModalTruncationPolicy::requested(2),
            &[1.0, 2.0, 2.0],
            &[0.0; 3],
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            4,
            1,
            None,
        )
        .unwrap_err();
        assert!(error.to_string().contains("complete truncation-boundary"));
    }

    #[test]
    fn records_a_complete_repeated_eigenspace() {
        let (truncation, eigenspaces) = build_modal_governance(
            ModalTruncationPolicy::requested(2),
            &[1.0, 2.0, 2.0],
            &[1.0e-8, 2.0e-8, 3.0e-8],
            &[1.0, 0.0, 0.0, 0.0, 1.0, 2.0e-9, 0.0, 2.0e-9, 1.0],
            4,
            1,
            Some(3.0),
        )
        .unwrap();
        assert_eq!(truncation.retained_mode_count(), 3);
        assert!(truncation.expanded_for_eigenspace());
        assert_eq!(eigenspaces.len(), 2);
        assert!(eigenspaces[1].is_repeated());
        assert_eq!(eigenspaces[1].mode_count(), 2);
    }

    #[test]
    fn rejects_a_probe_that_does_not_clear_the_boundary_eigenspace() {
        let error = build_modal_governance(
            ModalTruncationPolicy::requested(2),
            &[1.0, 2.0],
            &[0.0; 2],
            &[1.0, 0.0, 0.0, 1.0],
            4,
            1,
            Some(2.0 + 1.0e-8),
        )
        .unwrap_err();
        assert!(error.to_string().contains("complete truncation-boundary"));
    }

    #[test]
    fn rejects_nonfinite_or_unbounded_cluster_tolerances() {
        for tolerance in [f64::NAN, 0.0, 1.0] {
            let error = ModalTruncationPolicy::requested(2)
                .with_eigenspace_relative_tolerance(tolerance)
                .validate()
                .unwrap_err();
            assert!(error.to_string().contains("relative tolerance"));
        }
    }
}
