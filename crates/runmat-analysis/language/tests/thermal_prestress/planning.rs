use super::support::{source_module, TestHost};
use runmat_analysis_core::{FormulationRequirement, StateKind, StateSource, StateUse};
use runmat_analysis_fea::{study::*, ComputeBackend};
use runmat_analysis_language::*;
use runmat_geometry_core::{GeometryRegionRole, ResolvedGeometryRegion};
use runmat_meshing_core::{fixtures, CanonicalMeshingContract, ElementOrder, StableDigest};
use std::collections::BTreeMap;

#[test]
fn runnable_thermal_prestress_source_compiles_and_plans_without_model_patching() {
    // This language/planning test supplies the canonical STEP tetrahedron through
    // the inspection host, so only its asset reference differs from the example.
    // The native driver-replacement test uses the original box.brep and OCCT meshing.
    let source =
        include_str!("../../../../../docs/fea/examples/thermal-prestress-experimental.fea")
            .replace("./geometry/box.brep", "./geometry/tetrahedron.step");
    let (document, topology, _) = runmat_geometry_fixtures::exact_tetrahedron();
    let host = TestHost::default().with_geometry_inspection(
        "project/geometry/tetrahedron.step",
        GeometryInspection {
            content_digest: SourceDigest(*document.source.content_digest.bytes()),
            topology_digest: SourceDigest([19; 32]),
            capabilities: vec![
                GeometryInspectionCapability::Topology,
                GeometryInspectionCapability::PersistentIds,
            ],
            exchange: None,
        },
    );
    let checked = FeaCompiler::new(&host).check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", &source, 1),
        },
    });
    assert!(checked.is_valid(), "{:#?}", checked.diagnostics);
    let resolved = checked
        .resolve_study("project/main.fea", "heated_modes")
        .unwrap();
    let realized = resolved
        .study
        .realize_with_models(&resolved.models, &resolved.geometry, &[])
        .unwrap();
    let regions = BTreeMap::from([
        (
            "body".into(),
            ResolvedGeometryRegion {
                role: GeometryRegionRole::Solid,
                entity_ids: vec![topology.solids[0].id.clone()],
            },
        ),
        (
            "support".into(),
            ResolvedGeometryRegion {
                role: GeometryRegionRole::Face,
                entity_ids: vec![topology.faces[0].id.clone()],
            },
        ),
        (
            "hot".into(),
            ResolvedGeometryRegion {
                role: GeometryRegionRole::Face,
                entity_ids: vec![topology.faces[1].id.clone()],
            },
        ),
    ]);
    let mut mesh = fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    mesh.geometry.source_digest = StableDigest::from_bytes(*document.source.content_digest.bytes());
    mesh.geometry.geometry_revision = document.revision.revision;
    mesh.geometry.persistent_mapping_version = document.revision.persistent_mapping_version;
    for node in &mut mesh.topology.nodes {
        node.provenance = vec![topology.solids[0].id.clone()];
    }
    mesh.topology.volume_elements[0].region_id = topology.regions[0].id.clone();
    mesh.topology.volume_elements[0].provenance = vec![topology.solids[0].id.clone()];
    mesh.topology.regions[0].region_id = topology.regions[0].id.clone();
    for (face, source) in mesh.topology.boundary_faces.iter_mut().zip(&topology.faces) {
        face.provenance = vec![source.id.clone()];
    }
    for (edge, source) in mesh.topology.boundary_edges.iter_mut().zip(&topology.edges) {
        edge.provenance = vec![source.id.clone()];
    }
    mesh.seal_canonical_digest().unwrap();
    mesh.validate_canonical().unwrap();
    let product = build_structural_study_product(StructuralStudyProductInput {
        study: &realized,
        geometry: &document,
        regions: &regions,
        materials: &resolved.material_catalog,
        mesh: &mesh,
        planning: StructuralStudyPlanningPolicy {
            backend: ComputeBackend::Cpu,
            formulation_requirement: FormulationRequirement::Experimental,
            numerics: StructuralNumericalPlanningPolicy::production(),
        },
    })
    .unwrap();
    product.compiled.validate_against(&mesh).unwrap();
    let prefix = product.workloads.prestress_prefix().unwrap();
    assert!(prefix.thermal.is_some());
    // Planning retains an unbound template: the solved temperature state does not
    // exist until execution, so only its authored source is known at this point.
    assert!(prefix.preload.state_input.is_none());
    prefix.preload.validate_sequence_template().unwrap();
    let inputs = &prefix.preload.procedure_step().unwrap().state_inputs;
    assert_eq!(inputs.len(), 1);
    assert_eq!(inputs[0].required_kind, StateKind::Thermal);
    assert_eq!(inputs[0].use_as, StateUse::Prestress);
    assert!(matches!(
        &inputs[0].source,
        StateSource::Upstream { producer_step, state_id }
            if producer_step.as_str() == "heat" && state_id.0 == "heated"
    ));
    assert_eq!(
        prefix
            .consumer_step_ids
            .iter()
            .map(|id| id.as_str())
            .collect::<Vec<_>>(),
        vec!["modes"]
    );
}
