use std::{cmp::Ordering, collections::BTreeMap};

use runmat_meshing_core::{
    quality::predicate::{orient3d, PredicateSign},
    MeshingCancellationSignal, StableDigest,
};

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum QualityObjective {
    Shape,
    Refinement,
}

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
    let mut spectrum = quality
        .tetrahedra
        .iter()
        .filter_map(|tetrahedron| {
            let ratio = match objective {
                QualityObjective::Shape => tetrahedron.optimization_violation_ratio,
                QualityObjective::Refinement => tetrahedron.refinement_violation_ratio,
            };
            (ratio > 1.0).then_some(ratio)
        })
        .collect::<Vec<_>>();
    spectrum.sort_by(|left, right| right.total_cmp(left));
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
    if compare_spectra(
        &quality_spectrum(quality, objective),
        current_optimization_spectrum,
    ) != Ordering::Less
    {
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
    ) && contains_embedded_node(topology)?
    {
        reject(rejection_counts, "embedded_node");
        return Ok(None);
    }
    let validation = match topology_contract {
        CandidateTopologyContract::Delaunay => validate_constrained_delaunay_volume_topology,
        CandidateTopologyContract::CertifiedConformingCavity => validate_conforming_volume_topology,
    };
    if let Err(failure) = validation(topology, protected_faces, options.insertion, cancellation) {
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
    let quality =
        super::super::volume_quality::evaluate_delaunay_volume_quality_from_admitted_input(
            topology,
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

pub(super) fn compare_spectra(left: &[f64], right: &[f64]) -> Ordering {
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            super::super::volume_quality::violation_priority_key(*left).cmp(
                &super::super::volume_quality::violation_priority_key(*right),
            )
        })
        .find(|ordering| *ordering != Ordering::Equal)
        .unwrap_or_else(|| left.len().cmp(&right.len()))
}

fn reject(counts: &mut BTreeMap<&'static str, u64>, reason: &'static str) {
    *counts.entry(reason).or_default() += 1;
}

fn contains_embedded_node(
    topology: &DelaunayVolumeTopology,
) -> Result<bool, DelaunayVolumeSliverError> {
    let mut nodes_by_x = topology
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.coordinates_m[0], index))
        .collect::<Vec<_>>();
    nodes_by_x.sort_by(|left, right| {
        left.0
            .total_cmp(&right.0)
            .then_with(|| left.1.cmp(&right.1))
    });
    for tetrahedron in &topology.tetrahedra {
        let points = tetrahedron
            .vertex_indices
            .map(|vertex| topology.nodes[vertex as usize].coordinates_m);
        let minimum: [f64; 3] = std::array::from_fn(|axis| {
            points
                .iter()
                .map(|point| point[axis])
                .min_by(f64::total_cmp)
                .expect("tetrahedron has vertices")
        });
        let maximum: [f64; 3] = std::array::from_fn(|axis| {
            points
                .iter()
                .map(|point| point[axis])
                .max_by(f64::total_cmp)
                .expect("tetrahedron has vertices")
        });
        let first_candidate =
            nodes_by_x.partition_point(|(coordinate, _)| *coordinate < minimum[0]);
        let end_candidate = nodes_by_x.partition_point(|(coordinate, _)| *coordinate <= maximum[0]);
        for (_, node_index) in &nodes_by_x[first_candidate..end_candidate] {
            let node = &topology.nodes[*node_index];
            if tetrahedron.vertex_indices.contains(&(*node_index as u32))
                || (1..3).any(|axis| {
                    node.coordinates_m[axis] < minimum[axis]
                        || node.coordinates_m[axis] > maximum[axis]
                })
            {
                continue;
            }
            let mut interior = true;
            for replace in 0..4 {
                let mut simplex = points;
                simplex[replace] = node.coordinates_m;
                let sign = orient3d(simplex).map_err(|failure| {
                    error(
                        DelaunayVolumeSliverErrorKind::InvalidTopology,
                        format!("optimized topology containment predicate failed: {failure:?}"),
                    )
                })?;
                if sign != PredicateSign::Positive {
                    interior = false;
                    break;
                }
            }
            if interior {
                return Ok(true);
            }
        }
    }
    Ok(false)
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
