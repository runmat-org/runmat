use runmat_meshing_core::{MeshingCancellationSignal, StableDigest};
use serde::{Deserialize, Serialize};

use super::{
    validate_delaunay_volume_provenance, validate_delaunay_volume_quality,
    DelaunayInsertionErrorKind, DelaunayInsertionOptions, DelaunayVolumeNode,
    DelaunayVolumeProvenanceErrorKind, DelaunayVolumeQuality, DelaunayVolumeQualityError,
    DelaunayVolumeQualityErrorKind, DelaunayVolumeRefinementInput, DelaunayVolumeTopology,
};

mod relocation;
mod validation;

use relocation::{quality_spectrum, relocation_candidates};
pub use validation::validate_delaunay_volume_sliver_treatment;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DelaunayVolumeSliverOptions {
    pub maximum_passes: u64,
    pub maximum_candidate_evaluations_per_pass: u64,
    pub cancellation_check_interval: u64,
    pub insertion: DelaunayInsertionOptions,
}

impl Default for DelaunayVolumeSliverOptions {
    fn default() -> Self {
        Self {
            maximum_passes: 1_000_000,
            maximum_candidate_evaluations_per_pass: 32,
            cancellation_check_interval: 32,
            insertion: DelaunayInsertionOptions::default(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DelaunayVolumeSliverRelocation {
    pub source_node_identity: StableDigest,
    pub replacement_node: DelaunayVolumeNode,
    pub source_tetrahedron_node_identities: [StableDigest; 4],
}

#[derive(Clone, Debug, PartialEq)]
pub struct DelaunayVolumeSliverTreatment {
    pub topology: DelaunayVolumeTopology,
    pub quality: DelaunayVolumeQuality,
    /// Deterministic mutation order; each replacement becomes the source of any later move.
    pub relocations: Vec<DelaunayVolumeSliverRelocation>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DelaunayVolumeSliverErrorKind {
    InvalidOptions,
    InvalidInput,
    InvalidTopology,
    InvalidProvenance,
    InvalidQuality,
    NoAdmissibleRelocation,
    ResourceLimit,
    Cancelled,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DelaunayVolumeSliverError {
    pub kind: DelaunayVolumeSliverErrorKind,
    pub reason: String,
}

impl std::fmt::Display for DelaunayVolumeSliverError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "3D Delaunay volume sliver treatment {:?}: {}",
            self.kind, self.reason
        )
    }
}

impl std::error::Error for DelaunayVolumeSliverError {}

pub fn treat_delaunay_volume_slivers(
    input: DelaunayVolumeRefinementInput<'_>,
    options: DelaunayVolumeSliverOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<DelaunayVolumeSliverTreatment, DelaunayVolumeSliverError> {
    validate_options(options)?;
    validate_input(input, options, cancellation)?;
    let treatment = run_treatment(input, options, cancellation)?;
    validate_delaunay_volume_sliver_treatment(input, &treatment, options, cancellation)?;
    Ok(treatment)
}

pub(super) fn run_treatment(
    input: DelaunayVolumeRefinementInput<'_>,
    options: DelaunayVolumeSliverOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<DelaunayVolumeSliverTreatment, DelaunayVolumeSliverError> {
    let mut topology = input.topology.clone();
    let mut quality = input.quality.clone();
    let mut relocations = Vec::new();
    for pass in 0..options.maximum_passes {
        checkpoint(pass, options, cancellation)?;
        let Some(source) = worst_sliver(
            &quality,
            input.quality_options.minimum_metric_scaled_jacobian,
        ) else {
            return Ok(DelaunayVolumeSliverTreatment {
                topology,
                quality,
                relocations,
            });
        };
        let current_spectrum = quality_spectrum(&quality);
        let current = DelaunayVolumeRefinementInput {
            topology: &topology,
            metric_request: input.metric_request,
            provenance: input.provenance,
            quality: &quality,
            quality_options: input.quality_options,
        };
        let candidates =
            relocation_candidates(current, source, &current_spectrum, options, cancellation)?;
        let Some(candidate) = candidates.into_iter().next() else {
            return Err(error(
                DelaunayVolumeSliverErrorKind::NoAdmissibleRelocation,
                format!(
                    "no legal interior relocation improves metric sliver {} with scaled Jacobian {} below {}",
                    stable_simplex(source.node_identities),
                    source.metric_scaled_jacobian,
                    input.quality_options.minimum_metric_scaled_jacobian
                ),
            ));
        };
        topology = candidate.topology;
        quality = candidate.quality;
        relocations.push(candidate.relocation);
    }
    let remaining = worst_sliver(
        &quality,
        input.quality_options.minimum_metric_scaled_jacobian,
    )
    .map(|tetrahedron| tetrahedron.metric_scaled_jacobian)
    .unwrap_or(input.quality_options.minimum_metric_scaled_jacobian);
    Err(error(
        DelaunayVolumeSliverErrorKind::ResourceLimit,
        format!(
            "sliver treatment exhausted {} passes with minimum metric scaled Jacobian {remaining}",
            options.maximum_passes
        ),
    ))
}

fn worst_sliver(
    quality: &DelaunayVolumeQuality,
    minimum: f64,
) -> Option<&super::DelaunayTetrahedronQuality> {
    quality
        .tetrahedra
        .iter()
        .filter(|tetrahedron| tetrahedron.metric_scaled_jacobian < minimum)
        .min_by(|left, right| {
            left.metric_scaled_jacobian
                .total_cmp(&right.metric_scaled_jacobian)
                .then_with(|| left.node_identities.cmp(&right.node_identities))
        })
}

pub(super) fn validate_options(
    options: DelaunayVolumeSliverOptions,
) -> Result<(), DelaunayVolumeSliverError> {
    if options.maximum_passes == 0
        || options.maximum_candidate_evaluations_per_pass == 0
        || options.cancellation_check_interval == 0
    {
        return Err(error(
            DelaunayVolumeSliverErrorKind::InvalidOptions,
            "pass, candidate-evaluation, and cancellation limits must be nonzero",
        ));
    }
    super::insertion::validate_options(options.insertion).map_err(|failure| {
        error(
            DelaunayVolumeSliverErrorKind::InvalidOptions,
            failure.to_string(),
        )
    })?;
    Ok(())
}

pub(super) fn validate_input(
    input: DelaunayVolumeRefinementInput<'_>,
    options: DelaunayVolumeSliverOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayVolumeSliverError> {
    let protected_faces = input
        .provenance
        .facets
        .iter()
        .map(|facet| facet.node_identities)
        .collect::<Vec<_>>();
    super::insertion::validate_constrained_delaunay_volume_topology(
        input.topology,
        &protected_faces,
        options.insertion,
        cancellation,
    )
    .map_err(|failure| {
        let kind = match failure.kind {
            DelaunayInsertionErrorKind::InvalidOptions => {
                DelaunayVolumeSliverErrorKind::InvalidOptions
            }
            DelaunayInsertionErrorKind::ResourceLimit => {
                DelaunayVolumeSliverErrorKind::ResourceLimit
            }
            DelaunayInsertionErrorKind::Cancelled => DelaunayVolumeSliverErrorKind::Cancelled,
            _ => DelaunayVolumeSliverErrorKind::InvalidTopology,
        };
        error(kind, failure.to_string())
    })?;
    validate_delaunay_volume_provenance(
        input.topology,
        input.provenance,
        input.quality_options.provenance,
        cancellation,
    )
    .map_err(|failure| {
        let kind = match failure.kind {
            DelaunayVolumeProvenanceErrorKind::InvalidOptions => {
                DelaunayVolumeSliverErrorKind::InvalidOptions
            }
            DelaunayVolumeProvenanceErrorKind::InvalidTopology => {
                DelaunayVolumeSliverErrorKind::InvalidTopology
            }
            DelaunayVolumeProvenanceErrorKind::InvalidProvenance => {
                DelaunayVolumeSliverErrorKind::InvalidProvenance
            }
            DelaunayVolumeProvenanceErrorKind::ResourceLimit => {
                DelaunayVolumeSliverErrorKind::ResourceLimit
            }
            DelaunayVolumeProvenanceErrorKind::Cancelled => {
                DelaunayVolumeSliverErrorKind::Cancelled
            }
        };
        error(kind, failure.to_string())
    })?;
    validate_delaunay_volume_quality(
        input.topology,
        input.metric_request,
        input.provenance,
        input.quality,
        input.quality_options,
        cancellation,
    )
    .map_err(quality_error)
}

pub(super) fn checkpoint(
    step: u64,
    options: DelaunayVolumeSliverOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayVolumeSliverError> {
    if step.is_multiple_of(options.cancellation_check_interval) && cancellation.is_cancelled() {
        return Err(error(DelaunayVolumeSliverErrorKind::Cancelled, "cancelled"));
    }
    Ok(())
}

pub(super) fn quality_error(failure: DelaunayVolumeQualityError) -> DelaunayVolumeSliverError {
    let kind = match failure.kind {
        DelaunayVolumeQualityErrorKind::InvalidOptions => {
            DelaunayVolumeSliverErrorKind::InvalidOptions
        }
        DelaunayVolumeQualityErrorKind::InvalidTopology => {
            DelaunayVolumeSliverErrorKind::InvalidTopology
        }
        DelaunayVolumeQualityErrorKind::InvalidMetric
        | DelaunayVolumeQualityErrorKind::InvalidMetricContext => {
            DelaunayVolumeSliverErrorKind::InvalidProvenance
        }
        DelaunayVolumeQualityErrorKind::InvalidQuality
        | DelaunayVolumeQualityErrorKind::NumericalFailure => {
            DelaunayVolumeSliverErrorKind::InvalidQuality
        }
        DelaunayVolumeQualityErrorKind::ResourceLimit => {
            DelaunayVolumeSliverErrorKind::ResourceLimit
        }
        DelaunayVolumeQualityErrorKind::Cancelled => DelaunayVolumeSliverErrorKind::Cancelled,
    };
    error(kind, failure.to_string())
}

pub(super) fn relocation_identity_is_valid(relocation: &DelaunayVolumeSliverRelocation) -> bool {
    relocation::relocation_identity_is_valid(relocation)
}

pub(super) fn error(
    kind: DelaunayVolumeSliverErrorKind,
    reason: impl Into<String>,
) -> DelaunayVolumeSliverError {
    DelaunayVolumeSliverError {
        kind,
        reason: reason.into(),
    }
}

fn stable_simplex(mut identities: [StableDigest; 4]) -> String {
    identities.sort_unstable();
    identities
        .into_iter()
        .map(|identity| format!("{:02x?}", &identity.bytes()[..4]))
        .collect::<Vec<_>>()
        .join(":")
}

#[cfg(test)]
#[path = "volume_sliver/tests.rs"]
mod tests;
