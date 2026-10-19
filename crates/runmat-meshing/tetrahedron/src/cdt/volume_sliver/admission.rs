use std::{cmp::Ordering, collections::BTreeMap};

use runmat_meshing_core::{MeshingCancellationSignal, StableDigest};

use super::{
    error, quality_error, DelaunayVolumeSliverError, DelaunayVolumeSliverErrorKind,
    DelaunayVolumeSliverOptions,
};
use crate::cdt::{
    insertion::{
        validate_conforming_volume_topology, validate_constrained_delaunay_volume_topology,
    },
    DelaunayInsertionErrorKind, DelaunayVolumeQuality, DelaunayVolumeRefinementInput,
    DelaunayVolumeTopology,
};

mod embedded_nodes;
use embedded_nodes::contains_embedded_node;

#[derive(Clone, Copy)]
pub(super) enum CandidateTopologyContract {
    Delaunay,
    CertifiedConformingCavity,
}

#[derive(Clone, Copy)]
pub(super) enum CandidateRefinementPolicy {
    Preserve,
    AllowForCoupledConvergence,
}

pub(super) use crate::cdt::volume_quality::{compare_spectra, QualityObjective};

#[derive(Clone, Copy)]
pub(super) struct CandidateAdmissionPolicy {
    pub(super) topology_contract: CandidateTopologyContract,
    pub(super) refinement: CandidateRefinementPolicy,
    pub(super) objective: QualityObjective,
    pub(super) options: DelaunayVolumeSliverOptions,
}

pub(super) fn quality_spectrum(
    quality: &DelaunayVolumeQuality,
    objective: QualityObjective,
) -> Vec<f64> {
    quality_spectrum_from_entries(quality.tetrahedra.iter(), objective)
}

pub(super) fn quality_spectrum_from_entries<'a>(
    entries: impl Iterator<Item = &'a crate::cdt::DelaunayTetrahedronQuality>,
    objective: QualityObjective,
) -> Vec<f64> {
    candidate_spectrum_from_entries(entries, CandidateRefinementPolicy::Preserve, objective)
}

pub(super) fn candidate_spectrum(
    quality: &DelaunayVolumeQuality,
    policy: CandidateRefinementPolicy,
    objective: QualityObjective,
) -> Vec<f64> {
    candidate_spectrum_from_entries(quality.tetrahedra.iter(), policy, objective)
}

pub(super) fn candidate_spectrum_from_entries<'a>(
    entries: impl Iterator<Item = &'a crate::cdt::DelaunayTetrahedronQuality>,
    policy: CandidateRefinementPolicy,
    objective: QualityObjective,
) -> Vec<f64> {
    let mut spectrum = entries
        .flat_map(|cell| {
            if matches!(
                policy,
                CandidateRefinementPolicy::AllowForCoupledConvergence
            ) {
                [
                    cell.optimization_violation_ratio,
                    cell.refinement_violation_ratio,
                ]
            } else {
                [
                    match objective {
                        QualityObjective::Shape => cell.optimization_violation_ratio,
                        QualityObjective::Refinement => cell.refinement_violation_ratio,
                    },
                    0.,
                ]
            }
        })
        .filter(|ratio| *ratio > 1.)
        .collect::<Vec<_>>();
    spectrum.sort_by(|a, b| b.total_cmp(a));
    spectrum
}

pub(super) fn worst_quality_source<'a>(
    quality: &'a DelaunayVolumeQuality,
    topology: &DelaunayVolumeTopology,
    objective: QualityObjective,
) -> Option<&'a crate::cdt::DelaunayTetrahedronQuality> {
    quality
        .tetrahedra
        .iter()
        .filter(|tetrahedron| match objective {
            QualityObjective::Shape => tetrahedron.requires_optimization(),
            QualityObjective::Refinement => tetrahedron.requires_refinement(),
        })
        .max_by(|left, right| {
            let ratio = |tetrahedron: &crate::cdt::DelaunayTetrahedronQuality| match objective {
                QualityObjective::Shape => tetrahedron.optimization_violation_ratio,
                QualityObjective::Refinement => tetrahedron.refinement_violation_ratio,
            };
            let left_priority = super::super::volume_quality::violation_priority_key(ratio(left));
            let right_priority = super::super::volume_quality::violation_priority_key(ratio(right));
            left_priority.cmp(&right_priority).then_with(|| {
                super::super::volume_quality::compare_violation_priority(
                    ratio(left),
                    super::super::volume_quality::tetrahedron_geometry_key(
                        topology,
                        left.node_identities,
                    ),
                    left.node_identities,
                    ratio(right),
                    super::super::volume_quality::tetrahedron_geometry_key(
                        topology,
                        right.node_identities,
                    ),
                    right.node_identities,
                )
            })
        })
}

#[cfg(test)]
pub(super) fn source_for_policy<'a>(
    quality: &'a DelaunayVolumeQuality,
    topology: &DelaunayVolumeTopology,
    policy: super::SliverSearchPolicy,
) -> Option<&'a crate::cdt::DelaunayTetrahedronQuality> {
    let shape = worst_quality_source(quality, topology, QualityObjective::Shape);
    let refinement = worst_quality_source(quality, topology, QualityObjective::Refinement);
    match policy.current_objective(shape, refinement) {
        QualityObjective::Shape => shape,
        QualityObjective::Refinement => refinement,
    }
}

pub(super) fn evaluate_candidate(
    topology: &DelaunayVolumeTopology,
    input: DelaunayVolumeRefinementInput<'_>,
    protected_faces: &[[StableDigest; 3]],
    current_optimization_spectrum: &[f64],
    policy: CandidateAdmissionPolicy,
    cancellation: &dyn MeshingCancellationSignal,
    rejection_counts: &mut BTreeMap<&'static str, u64>,
) -> Result<Option<DelaunayVolumeQuality>, DelaunayVolumeSliverError> {
    let CandidateAdmissionPolicy {
        topology_contract,
        refinement: refinement_policy,
        objective,
        options,
    } = policy;
    let Some(quality) = evaluate_candidate_unranked(
        topology,
        input,
        protected_faces,
        topology_contract,
        options,
        cancellation,
        rejection_counts,
    )?
    else {
        return Ok(None);
    };
    if !candidate_quality_is_admissible(
        &quality,
        input,
        current_optimization_spectrum,
        refinement_policy,
        objective,
        rejection_counts,
    ) {
        return Ok(None);
    }
    Ok(Some(quality))
}

pub(super) fn candidate_quality_is_admissible(
    quality: &DelaunayVolumeQuality,
    input: DelaunayVolumeRefinementInput<'_>,
    current_optimization_spectrum: &[f64],
    refinement_policy: CandidateRefinementPolicy,
    objective: QualityObjective,
    rejection_counts: &mut BTreeMap<&'static str, u64>,
) -> bool {
    let spectrum = candidate_spectrum(quality, refinement_policy, objective);
    let progress = if matches!(
        refinement_policy,
        CandidateRefinementPolicy::AllowForCoupledConvergence
    ) {
        let floor = current_optimization_spectrum
            .first()
            .map(|value| super::super::volume_quality::violation_priority_key(*value))
            .unwrap_or(0);
        let prefix = |values: &[f64]| {
            values.partition_point(|value| {
                super::super::volume_quality::violation_priority_key(*value) >= floor
            })
        };
        compare_spectra(
            &spectrum[..prefix(&spectrum)],
            &current_optimization_spectrum[..prefix(current_optimization_spectrum)],
        )
    } else {
        compare_spectra(&spectrum, current_optimization_spectrum)
    };
    if progress != Ordering::Less {
        reject(rejection_counts, "optimization_quality_not_improved");
        return false;
    }
    if matches!(refinement_policy, CandidateRefinementPolicy::Preserve)
        && compare_spectra(
            &refinement_spectrum(quality),
            &refinement_spectrum(input.quality),
        ) == Ordering::Greater
    {
        reject(rejection_counts, "refinement_quality_regressed");
        return false;
    }
    true
}

pub(super) fn working_quality_is_admissible(
    quality: &crate::cdt::volume_quality::working::WorkingVolumeQuality,
    refinement_baseline: &crate::cdt::volume_quality::working::SpectrumDelta,
    refinement_policy: CandidateRefinementPolicy,
    objective: QualityObjective,
    rejection_counts: &mut BTreeMap<&'static str, u64>,
) -> bool {
    let coupled = matches!(
        refinement_policy,
        CandidateRefinementPolicy::AllowForCoupledConvergence
    );
    let progress = if coupled {
        quality.compare_joint_repair_baseline()
    } else {
        quality.compare_baseline(objective)
    };
    if progress != Ordering::Less {
        reject(
            rejection_counts,
            if coupled {
                "coupled_quality_not_improved"
            } else {
                "optimization_quality_not_improved"
            },
        );
        return false;
    }
    if matches!(refinement_policy, CandidateRefinementPolicy::Preserve)
        && quality.compare_target(refinement_baseline, QualityObjective::Refinement)
            == Ordering::Greater
    {
        reject(rejection_counts, "refinement_quality_regressed");
        return false;
    }
    true
}

pub(super) fn evaluate_candidate_unranked(
    topology: &DelaunayVolumeTopology,
    input: DelaunayVolumeRefinementInput<'_>,
    protected_faces: &[[StableDigest; 3]],
    topology_contract: CandidateTopologyContract,
    options: DelaunayVolumeSliverOptions,
    cancellation: &dyn MeshingCancellationSignal,
    rejection_counts: &mut BTreeMap<&'static str, u64>,
) -> Result<Option<DelaunayVolumeQuality>, DelaunayVolumeSliverError> {
    if !matches!(
        topology_contract,
        CandidateTopologyContract::CertifiedConformingCavity
    ) && contains_embedded_node(topology, options, cancellation)?
    {
        reject(rejection_counts, "embedded_node");
        return Ok(None);
    }
    let validation = match topology_contract {
        CandidateTopologyContract::Delaunay => validate_constrained_delaunay_volume_topology,
        CandidateTopologyContract::CertifiedConformingCavity => validate_conforming_volume_topology,
    };
    let validation = validation(topology, protected_faces, options.insertion, cancellation);
    finish_candidate_evaluation(topology, input, validation, cancellation, rejection_counts)
}

#[cfg(test)]
pub(super) fn evaluate_canonical_candidate_unranked(
    topology: &crate::cdt::topology::CanonicalVolumeTopology,
    input: DelaunayVolumeRefinementInput<'_>,
    protected_faces: &[[StableDigest; 3]],
    options: DelaunayVolumeSliverOptions,
    cancellation: &dyn MeshingCancellationSignal,
    rejection_counts: &mut BTreeMap<&'static str, u64>,
) -> Result<Option<DelaunayVolumeQuality>, DelaunayVolumeSliverError> {
    let validation = crate::cdt::insertion::validate_canonical_conforming_volume_topology(
        topology,
        protected_faces,
        options.insertion,
        cancellation,
    );
    finish_candidate_evaluation(topology, input, validation, cancellation, rejection_counts)
}

fn finish_candidate_evaluation(
    topology: &DelaunayVolumeTopology,
    input: DelaunayVolumeRefinementInput<'_>,
    validation: Result<(), crate::cdt::DelaunayInsertionError>,
    cancellation: &dyn MeshingCancellationSignal,
    rejection_counts: &mut BTreeMap<&'static str, u64>,
) -> Result<Option<DelaunayVolumeQuality>, DelaunayVolumeSliverError> {
    if let Err(failure) = validation {
        return match failure.kind {
            DelaunayInsertionErrorKind::ResourceLimit => Err(error(
                DelaunayVolumeSliverErrorKind::ResourceLimit,
                failure.to_string(),
            )),
            DelaunayInsertionErrorKind::Cancelled => Err(error(
                DelaunayVolumeSliverErrorKind::Cancelled,
                failure.to_string(),
            )),
            _ => {
                reject(rejection_counts, "invalid_conforming_topology");
                Ok(None)
            }
        };
    }
    let quality = super::super::volume_quality::evaluate_quality_after_mutation(
        topology,
        (input.topology, input.quality),
        input.metric_request,
        input.provenance,
        input.quality_options,
        cancellation,
    )
    .map_err(quality_error)?;
    Ok(Some(quality))
}

fn refinement_spectrum(quality: &DelaunayVolumeQuality) -> Vec<f64> {
    quality_spectrum(quality, QualityObjective::Refinement)
}

fn reject(counts: &mut BTreeMap<&'static str, u64>, reason: &'static str) {
    *counts.entry(reason).or_default() += 1;
}

#[cfg(test)]
mod comparison_tests {
    use super::compare_spectra;

    #[test]
    fn spectrum_order_ignores_only_low_order_quality_noise() {
        let reference: [f64; 2] = [1.25, 1.125];
        let low_order_noise = [f64::from_bits(reference[0].to_bits() + 32), reference[1]];
        assert_eq!(
            compare_spectra(&reference, &low_order_noise),
            std::cmp::Ordering::Equal
        );
        assert_eq!(
            compare_spectra(&[1.2, 1.1], &reference),
            std::cmp::Ordering::Less
        );
    }
}

#[cfg(test)]
mod tests;
