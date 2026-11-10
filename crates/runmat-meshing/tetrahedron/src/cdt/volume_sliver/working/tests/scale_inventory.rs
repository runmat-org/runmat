//! Representative retained-owner stress gate, separate from native CAD acceptance.
use super::*;
use crate::cdt::test_fixtures::structured_volume::{grid, identity, region};
use crate::cdt::DelaunayVolumeQualityOptions;
use crate::cdt::{
    volume_quality::evaluate_delaunay_volume_quality, DelaunayFacetProvenance,
    DelaunayVolumeProvenance,
};
use runmat_geometry_core::{PersistentEntityId, PersistentEntityKind};
use runmat_meshing_size::metric::{MetricCombinationRule, MetricFieldRequest, MetricTensor3};
use sha2::{Digest, Sha256};
use std::io::{self, Write};

pub(super) fn digest(value: &impl serde::Serialize) -> [u8; 32] {
    struct Sink(Sha256);
    impl Write for Sink {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0.update(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let mut sink = Sink(Sha256::new());
    serde_json::to_writer(&mut sink, value).unwrap();
    sink.0.finalize().into()
}

#[test]
#[ignore = "explicit large working-owner memory gate; RUNMAT_WORKING_SCALE_CELLS selects grid side"]
fn representative_grid_preserves_complete_topology_and_quality_through_working_trials() {
    let side = std::env::var("RUNMAT_WORKING_SCALE_CELLS")
        .unwrap_or_else(|_| "16".into())
        .parse::<u32>()
        .unwrap();
    let thin = std::env::var_os("RUNMAT_WORKING_SCALE_THIN").is_some();
    assert!((2..=if thin { 408 } else { 64 }).contains(&side));
    eprintln!("working-scale phase=canonical side={side}");
    let topology = grid(side, thin);
    let nodes = topology.nodes.len();
    let cells = topology.tetrahedra.len();
    let original_topology = digest(&topology);
    let mut provenance = DelaunayVolumeProvenance {
        nodes: Vec::new(),
        segments: Vec::new(),
        facets: Vec::new(),
    };
    let scoped = std::env::var_os("RUNMAT_WORKING_SCALE_SCOPED").is_some();
    if thin || scoped {
        provenance.facets = topology
            .incidence
            .boundary_facets
            .iter()
            .map(|face| {
                let mut identities = face
                    .vertex_indices
                    .map(|i| topology.nodes[i as usize].identity);
                identities.sort_unstable();
                DelaunayFacetProvenance {
                    node_identities: identities,
                    chart_id: identity(u64::MAX - 1000),
                    entity_ids: vec![PersistentEntityId::new(
                        PersistentEntityKind::Face,
                        "representative-outer-boundary".into(),
                        Vec::new(),
                    )],
                    region_ids: vec![region("representative-working-region")],
                }
            })
            .collect();
        provenance.facets.sort_by_key(|face| face.node_identities);
    }
    eprintln!(
        "working-scale boundary_facets={} thin={thin}",
        provenance.facets.len()
    );
    let mut request = MetricFieldRequest {
        entity_supports: Vec::new(),
        located_sources: Vec::new(),
        combination: MetricCombinationRule::MostRestrictiveIntersection,
        global_metric: MetricTensor3::isotropic_length_m(2.0).unwrap(),
        maximum_grading_ratio: 1.5,
        contributions: Vec::new(),
        cells_across_gap: None,
    };
    if scoped {
        request
            .contributions
            .push(runmat_meshing_size::metric::MetricContribution {
                source: runmat_meshing_size::metric::MetricSourceKind::Face,
                scope: runmat_meshing_size::metric::MetricContributionScope::Entity {
                    entity_id: provenance.facets[0].entity_ids[0].clone(),
                },
                metric: MetricTensor3::isotropic_length_m(0.5).unwrap(),
            });
    }
    // Exercise dense violation rankings without changing geometry or any native
    // study acceptance. Ordinary regular grids leave these inventories empty.
    let dense_ranking = std::env::var_os("RUNMAT_WORKING_SCALE_DENSE_RANKING").is_some();
    let options = DelaunayVolumeQualityOptions {
        minimum_metric_scaled_jacobian: if dense_ranking {
            0.99
        } else {
            DelaunayVolumeQualityOptions::default().minimum_metric_scaled_jacobian
        },
        ..DelaunayVolumeQualityOptions::default()
    };
    eprintln!("working-scale phase=quality nodes={nodes} cells={cells}");
    let quality = evaluate_delaunay_volume_quality(
        &topology,
        &request,
        &provenance,
        options,
        &NeverCancelled,
    )
    .unwrap();
    if dense_ranking {
        assert!(quality
            .tetrahedra
            .iter()
            .all(|cell| cell.optimization_violation_ratio > 1.0));
    }
    let original_quality = digest(&quality);
    let prepared = PreparedLocalQuality::prepare(&request, &provenance, options).unwrap();
    eprintln!("working-scale phase=working nodes={nodes} cells={cells}");
    let mut state =
        WorkingState::new(topology, quality, options, &prepared, &[], &NeverCancelled).unwrap();
    let trials_started = std::time::Instant::now();
    let mut maximum_dependency_reads = 0;
    for position in 0..64_u32 {
        let mut trial = state.trial();
        let vertices = trial.topology().cell(position).vertex_indices;
        let region = trial.topology().cell(position).region_id.clone();
        let point = std::array::from_fn(|axis| {
            vertices
                .iter()
                .map(|v| trial.topology().node(*v).coordinates_m.canonical_value()[axis])
                .sum::<f64>()
                / 4.
        });
        let inserted = trial
            .append_node(
                DelaunayVolumeNode::explicit(identity(u64::MAX - u64::from(position)), point),
                DelaunayVolumeSliverOptions::default(),
            )
            .unwrap();
        let created = (0..4)
            .map(|opposite| {
                let mut corners = vertices;
                corners[opposite] = inserted;
                (corners, region.clone())
            })
            .collect();
        trial
            .replace(
                &BTreeSet::from([position]),
                created,
                DelaunayVolumeSliverOptions::default(),
                &NeverCancelled,
            )
            .unwrap();
        if scoped {
            let (invalidated, reads) = trial.quality().last_scoped_update_work().unwrap();
            assert!(invalidated <= 2048 && reads <= 20000,
                "local scope exceeded representative dependency work: {invalidated} vertices, {reads} reads");
            maximum_dependency_reads = maximum_dependency_reads.max(reads);
        }
        // A query consumes deferred changes; dropping the trial must restore both owners.
        trial
            .topology()
            .prepare_cell_spatial_index(&mut || Ok::<_, ()>(()))
            .unwrap();
    }
    eprintln!("working-scale trials=64 scoped={scoped} maximum_dependency_reads={maximum_dependency_reads} elapsed_seconds={}", trials_started.elapsed().as_secs_f64());
    eprintln!("working-scale phase=independent-handoff nodes={nodes} cells={cells}");
    let (topology, quality) = state
        .finish(DelaunayVolumeSliverOptions::default(), &NeverCancelled)
        .unwrap();
    assert_eq!(topology.nodes.len(), nodes);
    assert_eq!(topology.tetrahedra.len(), cells);
    assert_eq!(digest(&topology), original_topology);
    assert_eq!(digest(&quality), original_quality);
    let node_storage = topology.nodes.as_ptr();
    let cell_storage = topology.tetrahedra.as_ptr();
    let quality_storage = quality.tetrahedra.as_ptr();
    let mesh = crate::cdt::volume_mesh::assemble_completed_components(
        vec![crate::cdt::DelaunayVolumeMesh {
            topology,
            quality,
            provenance,
            facet_recovery_insertions: Vec::new(),
            quality_mutations: Vec::new(),
        }],
        &request,
        crate::cdt::DelaunayVolumeMeshOptions {
            quality: options,
            ..Default::default()
        },
        &NeverCancelled,
    )
    .unwrap();
    assert_eq!(mesh.topology.nodes.as_ptr(), node_storage);
    assert_eq!(mesh.topology.tetrahedra.as_ptr(), cell_storage);
    assert_eq!(mesh.quality.tetrahedra.as_ptr(), quality_storage);
    assert_eq!(digest(&mesh.topology), original_topology);
    assert_eq!(digest(&mesh.quality), original_quality);
    if std::env::var_os("RUNMAT_WORKING_SCALE_CODEC").is_some() {
        let original_mesh = digest(&mesh);
        super::scale_codec::round_trip(mesh, original_mesh);
    }
    eprintln!("working-scale phase=complete nodes={nodes} cells={cells}");
}
