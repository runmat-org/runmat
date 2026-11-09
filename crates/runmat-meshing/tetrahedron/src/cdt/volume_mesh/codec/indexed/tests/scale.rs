use super::*;
use crate::cdt::test_fixtures::structured_volume::{grid, identity, region};
use crate::cdt::{DelaunayFacetProvenance, DelaunayVolumeProvenance, DelaunayVolumeQualityOptions};
use runmat_meshing_size::metric::{MetricCombinationRule, MetricFieldRequest, MetricTensor3};
#[test]
#[ignore = "explicit representative publication-budget gate without CAD or meshing"]
fn representative_volume_publication_fits_original_default_memory_admission() {
    publication_case(32, false);
    publication_case(192, true);
}
fn publication_case(side: u32, thin: bool) {
    // Bulk196608cells/35937nodes and thin221184cells/74498nodes exercise
    // representation admission, not original CAD or whole-task acceptance.
    let topology = grid(side, thin);
    let mut provenance = DelaunayVolumeProvenance {
        nodes: Vec::new(),
        segments: Vec::new(),
        facets: topology
            .incidence
            .boundary_facets
            .iter()
            .map(|face| {
                let mut node_identities = face
                    .vertex_indices
                    .map(|index| topology.nodes[index as usize].identity);
                node_identities.sort_unstable();
                DelaunayFacetProvenance {
                    node_identities,
                    chart_id: identity(u64::MAX - 1000),
                    entity_ids: vec![runmat_geometry_core::PersistentEntityId::new(
                        runmat_geometry_core::PersistentEntityKind::Face,
                        "representative-outer-boundary".into(),
                        Vec::new(),
                    )],
                    region_ids: vec![region("representative-working-region")],
                }
            })
            .collect(),
    };
    provenance.facets.sort_by_key(|face| face.node_identities);
    let metric = MetricFieldRequest {
        entity_supports: Vec::new(),
        located_sources: Vec::new(),
        combination: MetricCombinationRule::MostRestrictiveIntersection,
        global_metric: MetricTensor3::isotropic_length_m(2.).unwrap(),
        maximum_grading_ratio: 1.5,
        contributions: Vec::new(),
        cells_across_gap: None,
    };
    let quality = crate::cdt::evaluate_delaunay_volume_quality(
        &topology,
        &metric,
        &provenance,
        DelaunayVolumeQualityOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let mesh = DelaunayVolumeMesh {
        topology,
        provenance,
        quality,
        facet_recovery_insertions: Vec::new(),
        quality_mutations: Vec::new(),
    };
    let preparation = VolumeMeshEncoding::preparation_bytes(&mesh).unwrap();
    let prepared = VolumeMeshEncoding::new(&mesh, &Control).unwrap();
    let profile = prepared.encoding_profile(&Control).unwrap();
    assert!(
        preparation
            .checked_add(profile.resident_workspace_bytes_estimate)
            .unwrap()
            <= 512 << 20
    );
    let bytes = prepared.encode(&Control).unwrap();
    assert!(bytes.len() as u64 <= profile.encoded_bytes_upper_bound);
    let mut typed_bytes = 0;
    let decoded: DecodedEnvelope = runmat_canonical_codec::decode_contract_with_admission(
        CODEC_PREFIX,
        CONTRACT_DOMAIN,
        &bytes,
        CONTRACT_LIMITS,
        &Control,
        |value| {
            typed_bytes = crate::cdt::volume_mesh::codec::admission::indexed_materialization_bytes(
                value, &Control,
            )?;
            assert!(typed_bytes + bytes.len() as u64 <= 512 << 20);
            Ok(())
        },
    )
    .unwrap();
    eprintln!(
        "typed-decode-scale thin={thin} wire={} typed_wire_and_row_scratch={typed_bytes}",
        bytes.len()
    );
    drop(decoded);
    eprintln!(
        "publication-scale thin={thin} nodes={} cells={} preparation={} encoding_workspace={} actual_wire={}",
        mesh.topology.nodes.len(),
        mesh.topology.tetrahedra.len(),
        preparation,
        profile.resident_workspace_bytes_estimate,
        bytes.len()
    );
    drop(bytes);
    drop(prepared);
    let snapshot = crate::cdt::VolumeQualitySnapshot {
        topology: &mesh.topology,
        provenance: &mesh.provenance,
        metric: &metric,
        fixed_nodes: &[],
        options: DelaunayVolumeMeshOptions::default(),
    };
    let diagnostic_preparation =
        crate::cdt::volume_quality_snapshot_preparation_storage_bytes(&snapshot).unwrap();
    let diagnostic = crate::cdt::prepare_volume_quality_snapshot(
        &snapshot,
        crate::cdt::VolumeQualityBinding {
            request: identity(1),
            geometry: identity(2),
            implementation: identity(3),
        },
        &Control,
    )
    .unwrap();
    let (encoding, materialization) = diagnostic.profile(CONTRACT_LIMITS, &Control).unwrap();
    assert!(diagnostic_preparation + encoding.resident_workspace_bytes_estimate <= 512 << 20);
    eprintln!("diagnostic-scale thin={thin} preparation={diagnostic_preparation} encoding_workspace={} materialization={materialization}",encoding.resident_workspace_bytes_estimate);
}
