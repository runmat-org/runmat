use super::*;
use crate::cdt::volume_quality::tests::{metric_contexts, metric_request, topology};
use crate::cdt::{mutation::TopologyMutation, DelaunayTopologyOptions, DelaunayVolumeNode};
use runmat_meshing_core::NeverCancelled;

fn spectrum(quality: &DelaunayVolumeQuality, objective: QualityObjective) -> Vec<f64> {
    let mut values = quality
        .tetrahedra
        .iter()
        .map(|cell| match objective {
            QualityObjective::Shape => cell.optimization_violation_ratio,
            QualityObjective::Refinement => cell.refinement_violation_ratio,
        })
        .filter(|value| *value > 1.0)
        .collect::<Vec<_>>();
    values.sort_by(|left, right| right.total_cmp(left));
    values
}

fn materialize(
    owner: &WorkingVolumeQuality,
    topology: &DelaunayVolumeTopology,
) -> DelaunayVolumeQuality {
    summary::summarize(
        topology,
        owner
            .cells
            .values()
            .map(|entry| entry.quality.clone())
            .collect(),
        DelaunayVolumeQualityOptions::default(),
        &NeverCancelled,
    )
    .unwrap()
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Uniform and scoped insertion trials compare every quality field and both ranking
//   spectra with full canonical evaluation through new identity orderings. Rollback restores
//   original graph-slot ownership and quality. This finite family does not cover every possible
//   metric or original CAD model.
#[test]
fn local_quality_matches_full_reference_through_inserted_identity_reordering_and_rollback() {
    check_insert_and_rollback(false);
    check_insert_and_rollback(true);
}

fn check_insert_and_rollback(scoped: bool) {
    let base = topology();
    let provenance = metric_contexts(&base);
    let mut request = metric_request();
    if !scoped {
        request.contributions.clear();
    }
    let options = DelaunayVolumeQualityOptions::default();
    let reference =
        evaluate_delaunay_volume_quality(&base, &request, &provenance, options, &NeverCancelled)
            .unwrap();
    let prepared = PreparedLocalQuality::prepare(&request, &provenance, options).unwrap();
    for bytes in [[0x01; 32], [0x03; 32], [0x06; 32]] {
        let mut identity = bytes;
        identity[31] = 7;
        let mut topology = TopologyMutation::from_canonical(base.clone());
        let mut owner =
            WorkingVolumeQuality::new(reference.clone(), &topology, options, &NeverCancelled)
                .unwrap();
        prepared
            .initialize_scoped(&mut owner, &topology, &NeverCancelled)
            .unwrap();
        let original_dependencies = owner
            .cells
            .iter()
            .map(|(key, entry)| (*key, entry.grading.clone()))
            .collect::<Vec<_>>();
        {
            let mut cells = topology.cavity_trial();
            let mut quality = owner.trial();
            let source = cells.cell(0);
            let vertices = source.vertex_indices;
            let region = source.region_id.clone();
            let removed = [cells.cell_key(0)];
            let point = std::array::from_fn(|axis| {
                vertices
                    .iter()
                    .map(|vertex| cells.node(*vertex).coordinates_m.canonical_value()[axis])
                    .sum::<f64>()
                    / 4.0
            });
            let inserted = cells
                .append_node(DelaunayVolumeNode::explicit(
                    StableDigest::from_bytes(identity),
                    point,
                ))
                .unwrap();
            let created = (0..4)
                .map(|opposite| {
                    let mut corners = vertices;
                    corners[opposite] = inserted;
                    (corners, region.clone())
                })
                .collect();
            let created = cells.replace_cavity(&BTreeSet::from([0]), created).unwrap();
            prepared
                .replace_quality(&mut quality, &removed, &created, &*cells, &NeverCancelled)
                .unwrap();
            let frozen = cells
                .validated_snapshot(DelaunayTopologyOptions::default(), &NeverCancelled)
                .unwrap();
            let full = evaluate_delaunay_volume_quality(
                &frozen,
                &request,
                &provenance,
                options,
                &NeverCancelled,
            )
            .unwrap();
            assert_eq!(materialize(&quality, &frozen), full);
            for cell in &full.tetrahedra {
                let mut reordered = cell.node_identities;
                reordered.reverse();
                assert_eq!(
                    quality
                        .quality_for_cell(reordered)
                        .map(|cell| cell.refinement_violation_ratio),
                    Some(cell.refinement_violation_ratio)
                );
            }
            for objective in [QualityObjective::Shape, QualityObjective::Refinement] {
                assert_eq!(
                    quality.compare_baseline(objective),
                    compare_spectra(
                        &spectrum(&full, objective),
                        &spectrum(&reference, objective)
                    )
                );
                let expected = match objective {
                    QualityObjective::Shape => full.worst_optimization_tetrahedron,
                    QualityObjective::Refinement => full.worst_refinement_tetrahedron,
                };
                assert_eq!(
                    quality.worst(objective).map(|cell| cell.node_identities),
                    expected
                );
            }
            if !scoped {
                assert_eq!(quality.journals.last().unwrap().len(), 5);
            }
            assert_eq!(
                quality
                    .cells
                    .values()
                    .filter(|entry| entry.grading.is_some())
                    .count(),
                if scoped { full.tetrahedra.len() } else { 0 }
            );
        }
        let frozen = topology
            .freeze(DelaunayTopologyOptions::default(), &NeverCancelled)
            .unwrap();
        assert_eq!(frozen, base);
        assert!(owner
            .cells
            .iter()
            .map(|(key, entry)| (*key, entry.grading.clone()))
            .eq(original_dependencies));
        assert_eq!(
            owner.compare_baseline(QualityObjective::Shape),
            Ordering::Equal
        );
        assert_eq!(
            owner.compare_baseline(QualityObjective::Refinement),
            Ordering::Equal
        );
        assert!(owner.baseline_originals.is_empty());
        assert_eq!(
            owner
                .into_canonical(&frozen, options, &NeverCancelled)
                .unwrap(),
            reference
        );
    }
}

#[test]
fn spectrum_bin_differences_preserve_original_lexicographic_comparison() {
    let values = [
        1.0 + f64::EPSILON,
        1.5,
        2.0,
        f64::from_bits(2.0_f64.to_bits() + 8),
        4.0,
        f64::MAX,
    ];
    let mut spectra = vec![Vec::new()];
    for first in values {
        for second in values {
            let mut sequence = vec![first, second];
            sequence.sort_by(|left, right| right.total_cmp(left));
            spectra.push(sequence);
            spectra.push(vec![first]);
        }
    }
    for left in &spectra {
        for right in &spectra {
            let mut ranking = Ranking::default();
            for (values, direction) in [(left, 1), (right, -1)] {
                for value in values {
                    *ranking
                        .delta
                        .entry(violation_priority_key(*value))
                        .or_default() += direction;
                }
            }
            ranking.delta.retain(|_, count| *count != 0);
            assert_eq!(
                ranking.compare_baseline(),
                compare_spectra(left, right),
                "{left:?} versus {right:?}"
            );
        }
    }
}

mod retention;

#[test]
fn maintained_refinement_windows_match_the_full_priority_order() {
    let topology = topology();
    let provenance = metric_contexts(&topology);
    let request = metric_request();
    let options = DelaunayVolumeQualityOptions {
        maximum_metric_edge_length: 0.1,
        ..DelaunayVolumeQualityOptions::default()
    };
    let reference = evaluate_delaunay_volume_quality(
        &topology,
        &request,
        &provenance,
        options,
        &NeverCancelled,
    )
    .unwrap();
    let owner =
        WorkingVolumeQuality::new(reference.clone(), &topology, options, &NeverCancelled).unwrap();
    let mut expected = reference
        .tetrahedra
        .iter()
        .filter(|cell| cell.requires_refinement())
        .collect::<Vec<_>>();
    expected.sort_by(|a, b| {
        compare_violation_priority(
            b.refinement_violation_ratio,
            tetrahedron_geometry_key(&topology, b.node_identities),
            b.node_identities,
            a.refinement_violation_ratio,
            tetrahedron_geometry_key(&topology, a.node_identities),
            a.node_identities,
        )
    });
    assert!(!expected.is_empty());
    for width in [0, 1, 2, 128] {
        let mut after = None;
        let mut actual = Vec::new();
        loop {
            let window = owner.refinement_window(after, width);
            if window.is_empty() {
                break;
            }
            for (cursor, source) in window {
                after = Some(cursor);
                actual.push(source.node_identities);
            }
        }
        let expected = if width == 0 {
            Vec::new()
        } else {
            expected
                .iter()
                .map(|source| source.node_identities)
                .collect::<Vec<_>>()
        };
        assert_eq!(actual, expected);
    }
}
