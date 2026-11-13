#![cfg(feature = "occt-native")]
use runmat_geometry_core::{LengthUnit, PersistentEntityKind};
use runmat_geometry_io::{
    close_cad_preview_session, import_exact_cad, import_geometry_with_context,
    read_cad_preview_session_chunk, start_cad_preview_session, ExactCadImportOptions,
    GeometryFormat, GeometryImportContext, GeometryImportOptions,
};
use std::collections::BTreeSet;

#[test]
fn full_and_progressive_cad_preview_publish_exact_occurrence_ids() {
    let bytes = include_bytes!("fixtures/two_box_assembly.step");
    let context = GeometryImportContext::new();
    let options = GeometryImportOptions::default();
    let exact = import_exact_cad(
        "assembly.step",
        bytes,
        GeometryFormat::Step,
        &ExactCadImportOptions::default(),
        &context,
    )
    .unwrap();
    let expected: BTreeSet<_> = exact
        .topology
        .faces
        .iter()
        .map(|f| &f.id)
        .chain(exact.topology.solids.iter().map(|s| &s.id))
        .map(|id| id.analysis_selection_id().unwrap())
        .collect();
    let full =
        import_geometry_with_context("assembly.step", bytes, options.clone(), &context).unwrap();
    let actual: BTreeSet<_> = full
        .asset
        .regions
        .iter()
        .map(|r| r.region_id.clone())
        .collect();
    assert_eq!(actual, expected);
    // Author exactly the IDs exposed by inspection, then resolve through the production
    // execution selector. No ordinal/label resolver participates in this round trip.
    let definition = runmat_geometry_core::GeometryDefinition {
        module_path: "project/main.fea".into(),
        geometry_id: "part".into(),
        parameters: Vec::new(),
        cad_parameters: Vec::new(),
        source: runmat_geometry_core::GeometryDefinitionSource::Imported(Box::new(
            runmat_geometry_core::ImportedGeometryDefinition {
                preview_extent_m: None,
                logical_path: "assembly.step".into(),
                format: runmat_geometry_core::GeometrySourceFormat::Step,
                authored_units: None,
                unit_evidence: exact.source_unit_evidence.clone(),
                content_digest: exact.source_digest,
                topology_digest: exact.build_closure().unwrap().manifest.topology.digest,
                capabilities: runmat_geometry_core::GeometryAuthoringCapabilities {
                    topology: true,
                    persistent_ids: true,
                    ..Default::default()
                },
                exchange: None,
                shared_interfaces: false,
            },
        )),
        regions: full
            .asset
            .regions
            .iter()
            .enumerate()
            .map(
                |(index, region)| runmat_geometry_core::GeometryRegionDefinition {
                    region_id: format!("selection_{index}"),
                    role: if region.tag.as_deref() == Some("cad_body") {
                        runmat_geometry_core::GeometryRegionRole::Solid
                    } else {
                        runmat_geometry_core::GeometryRegionRole::Face
                    },
                    selector: runmat_geometry_core::GeometryRegionSelector::PersistentId(
                        region.region_id.clone(),
                    ),
                    cardinality: runmat_geometry_core::GeometryRegionCardinality::ExactlyOne,
                },
            )
            .collect(),
    };
    let resolved = runmat_geometry_ops::resolve_geometry_regions(
        &definition,
        &exact.topology,
        &Default::default(),
        &NoNamedEvidence,
    )
    .unwrap();
    assert_eq!(resolved.len(), expected.len());
    assert!(resolved.values().all(|region| region.entity_ids.len() == 1));
    assert_eq!(exact.topology.solids.len(), 2);
    assert!(exact
        .topology
        .solids
        .iter()
        .all(|s| s.id.kind == PersistentEntityKind::Solid));
    let started =
        start_cad_preview_session("assembly.step", bytes, options.clone(), &context).unwrap();
    let mut streamed = BTreeSet::new();
    for _ in 0..100 {
        let chunk =
            read_cad_preview_session_chunk(&started, 1, 1, options.clone(), &context).unwrap();
        streamed.extend(
            chunk
                .result
                .asset
                .regions
                .iter()
                .map(|r| r.region_id.clone()),
        );
        if chunk.done {
            break;
        }
    }
    close_cad_preview_session(started.session_id);
    assert_eq!(streamed, expected);
}

#[test]
fn reordered_compound_children_keep_exact_selection_identity() {
    let context = GeometryImportContext::new();
    let options = ExactCadImportOptions {
        source_units: Some(LengthUnit::Meter),
        ..Default::default()
    };
    let first = import_exact_cad(
        "part.brep",
        include_bytes!("fixtures/disconnected_solids.brep"),
        GeometryFormat::Brep,
        &options,
        &context,
    )
    .unwrap();
    let second = import_exact_cad(
        "part.brep",
        include_bytes!("fixtures/disconnected_solids_reversed.brep"),
        GeometryFormat::Brep,
        &options,
        &context,
    )
    .unwrap();
    let ids = |imported: &runmat_geometry_io::ImportedExactCad| -> BTreeSet<String> {
        imported
            .topology
            .faces
            .iter()
            .map(|f| &f.id)
            .chain(imported.topology.solids.iter().map(|s| &s.id))
            .map(|id| id.analysis_selection_id().unwrap())
            .collect()
    };
    assert_eq!(ids(&first), ids(&second));
}

struct NoNamedEvidence;
impl runmat_geometry_ops::GeometryRegionEvidence for NoNamedEvidence {
    type Error = String;
    fn by_name(
        &self,
        _: runmat_geometry_core::GeometryRegionRole,
        _: &str,
    ) -> Result<Vec<runmat_geometry_core::PersistentEntityId>, String> {
        Err("name lookup must not be used".into())
    }
    fn by_tag(
        &self,
        _: runmat_geometry_core::GeometryRegionRole,
        _: &str,
    ) -> Result<Vec<runmat_geometry_core::PersistentEntityId>, String> {
        Err("tag lookup must not be used".into())
    }
    fn generated_output(
        &self,
        _: runmat_geometry_core::GeometryRegionRole,
        _: &str,
    ) -> Result<Vec<runmat_geometry_core::PersistentEntityId>, String> {
        Err("generated lookup must not be used".into())
    }
}
