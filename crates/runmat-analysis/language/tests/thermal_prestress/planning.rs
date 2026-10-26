use super::support::{source_module, TestHost};
use runmat_analysis_core::{AnalysisProcedure, SolutionSpace, StateKind, StateSource, StateUse};
use runmat_analysis_fea::{study::*, ComputeBackend};
use runmat_analysis_language::*;
use runmat_geometry_core::{GeometryRegionRole, ResolvedGeometryRegion};
use runmat_meshing_core::{fixtures, CanonicalMeshingContract, ElementOrder, StableDigest};
use std::collections::BTreeMap;

macro_rules! planning_case {
    ($name:ident, $order:ident, $consumer:ident) => {
        #[test]
        fn $name() {
            check_planning(ElementOrder::$order, Consumer::$consumer);
        }
    };
}

planning_case!(authored_tet4_modal, Tet4, Modal);
planning_case!(authored_tet10_modal, Tet10, Modal);
planning_case!(authored_tet4_buckling, Tet4, Buckling);
planning_case!(authored_tet10_buckling, Tet10, Buckling);
planning_case!(authored_tet4_direct_harmonic, Tet4, Direct);
planning_case!(authored_tet10_direct_harmonic, Tet10, Direct);
planning_case!(authored_tet4_reduced_harmonic, Tet4, Reduced);
planning_case!(authored_tet10_reduced_harmonic, Tet10, Reduced);

#[derive(Clone, Copy)]
enum Consumer {
    Modal,
    Buckling,
    Direct,
    Reduced,
}

fn check_planning(order: ElementOrder, consumer: Consumer) {
    // This language/planning test supplies the canonical STEP tetrahedron through
    // the inspection host, so only its asset reference differs from the example.
    // The native driver-replacement test uses the original box.brep and OCCT meshing.
    let mut source = include_str!("../../../../../docs/simulation/examples/thermal-prestress.fea")
        .replace("./geometry/box.brep", "./geometry/tetrahedron.step");
    if order == ElementOrder::Tet10 {
        source = source.replace("order = linear", "order = quadratic");
    }
    let (procedure, space) = match consumer {
        Consumer::Modal => (
            AnalysisProcedure::PrestressedModalExtraction,
            SolutionSpace::Direct,
        ),
        Consumer::Buckling => {
            source = source.replace("structural prestressed_modal", "structural buckling");
            (AnalysisProcedure::EigenvalueBuckling, SolutionSpace::Direct)
        }
        Consumer::Direct => {
            source = source
                .replace(
                    "structural prestressed_modal",
                    "structural prestressed_harmonic",
                )
                .replace("modes = 1", "frequencies = [60 Hz]");
            (
                AnalysisProcedure::PrestressedHarmonicResponse,
                SolutionSpace::Direct,
            )
        }
        Consumer::Reduced => {
            source = format!("{}\nanalysis response after analysis.modes {{\nstructural prestressed_harmonic {{\nprestress = analysis.preload.state.geometric_stiffness\nbasis = analysis.modes.state.modal_basis\nfrequencies = [60 Hz]\n}}\n}}\n}}", source.trim_end().strip_suffix('}').unwrap());
            (
                AnalysisProcedure::PrestressedHarmonicResponse,
                SolutionSpace::ModalReduced,
            )
        }
    };
    let (document, topology, _) = runmat_geometry_fixtures::exact_tetrahedron();
    let host = TestHost::default().with_geometry_inspection(
        "project/geometry/tetrahedron.step",
        GeometryInspection {preview_extent_m: None,
            named_selections: Some({
                let mut metadata: runmat_geometry_core::GeometryNamedSelections = serde_json::from_str(r#"{
  "schema_version": 1,
  "source_digest": [
    121,
    173,
    34,
    177,
    204,
    101,
    47,
    35,
    7,
    161,
    216,
    249,
    195,
    236,
    86,
    68,
    250,
    47,
    134,
    222,
    29,
    151,
    0,
    130,
    13,
    35,
    78,
    225,
    25,
    158,
    109,
    52
  ],
  "regions": [
    {
      "region_id": "body",
      "role": "solid",
      "selector": {
        "kind": "persistent_id",
        "value": "persistent-entity:sha256:eb1929ea502ba35824eceb7a1aa23719ab7fa023c89662b8de06cc458560a3f0"
      },
      "cardinality": "exactly_one"
    },
    {
      "region_id": "support",
      "role": "face",
      "selector": {
        "kind": "persistent_id",
        "value": "persistent-entity:sha256:8af3f3e08304762c1332b213d0ecd818feb256a2d233fd3e99e8978e41ae7a23"
      },
      "cardinality": "exactly_one"
    },
    {
      "region_id": "load_face",
      "role": "face",
      "selector": {
        "kind": "persistent_id",
        "value": "persistent-entity:sha256:db5e0a2817b1cfb20a910953eda8e66c0ed46aa95c61e200de19fec0ab781129"
      },
      "cardinality": "exactly_one"
    },
    {
      "region_id": "hot",
      "role": "face",
      "selector": {
        "kind": "persistent_id",
        "value": "persistent-entity:sha256:db5e0a2817b1cfb20a910953eda8e66c0ed46aa95c61e200de19fec0ab781129"
      },
      "cardinality": "exactly_one"
    },
    {
      "region_id": "bulk",
      "role": "solid",
      "selector": {
        "kind": "persistent_id",
        "value": "persistent-entity:sha256:eb1929ea502ba35824eceb7a1aa23719ab7fa023c89662b8de06cc458560a3f0"
      },
      "cardinality": "exactly_one"
    },
    {
      "region_id": "left",
      "role": "face",
      "selector": {
        "kind": "persistent_id",
        "value": "persistent-entity:sha256:8af3f3e08304762c1332b213d0ecd818feb256a2d233fd3e99e8978e41ae7a23"
      },
      "cardinality": "exactly_one"
    },
    {
      "region_id": "right",
      "role": "face",
      "selector": {
        "kind": "persistent_id",
        "value": "persistent-entity:sha256:db5e0a2817b1cfb20a910953eda8e66c0ed46aa95c61e200de19fec0ab781129"
      },
      "cardinality": "exactly_one"
    },
    {
      "region_id": "bottom_y",
      "role": "face",
      "selector": {
        "kind": "persistent_id",
        "value": "persistent-entity:sha256:ae08f150d6636feb8607ede706d4202f4481fb1d7b9399842aae13d235837e33"
      },
      "cardinality": "exactly_one"
    },
    {
      "region_id": "bottom_z",
      "role": "face",
      "selector": {
        "kind": "persistent_id",
        "value": "persistent-entity:sha256:1f2056eadc83682feeabae84eae920614441a06141b296350bb4d4548218a75e"
      },
      "cardinality": "exactly_one"
    }
  ]
}"#).unwrap();
                metadata.source_digest = document.source.content_digest;
                metadata
            }),
            unit_evidence: Default::default(),
            content_digest: SourceDigest(*document.source.content_digest.bytes()),
            topology_digest: SourceDigest([19; 32]),
            capabilities: vec![
                GeometryInspectionCapability::Topology,
                GeometryInspectionCapability::PersistentIds,
            ],
            exchange: None,
            resolved_units: None,
            authored_names: Default::default(),
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
    assert_eq!(realized.sequence.steps.last().unwrap().procedure, procedure);
    assert_eq!(
        realized.sequence.steps.last().unwrap().solution_space,
        space
    );
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
    let mut mesh = fixtures::canonical_tetrahedron_solver_mesh(order);
    mesh.geometry.source_digest = StableDigest::from_bytes(*document.source.content_digest.bytes());
    mesh.geometry.geometry_revision = document.revision.revision;
    mesh.geometry.persistent_mapping_version = document.revision.persistent_mapping_version;
    let face_ids = mesh
        .topology
        .boundary_faces
        .iter()
        .zip(&topology.faces)
        .map(|(face, source)| (face.provenance[0].clone(), source.id.clone()))
        .collect::<BTreeMap<_, _>>();
    for node in &mut mesh.topology.nodes {
        for id in &mut node.provenance {
            *id = face_ids
                .get(id)
                .cloned()
                .unwrap_or_else(|| topology.solids[0].id.clone());
        }
        node.provenance.sort();
        // Retain the synthetic fixture's parameter/provenance consistency for Tet10.
        // This planning test does not attest exact-CAD parameter evaluation.
        for parameter in &mut node.exact_parameters {
            if let runmat_meshing_core::SolverNodeExactParameter::Surface {
                source_face_id, ..
            } = parameter
            {
                *source_face_id = face_ids[source_face_id].clone();
            }
        }
        runmat_meshing_core::sort_solver_node_exact_parameters(&mut node.exact_parameters);
    }
    mesh.topology.volume_elements[0].region_id = topology.regions[0].id.clone();
    mesh.topology.volume_elements[0].provenance = vec![topology.solids[0].id.clone()];
    mesh.topology.regions[0].region_id = topology.regions[0].id.clone();
    for (face, source) in mesh.topology.boundary_faces.iter_mut().zip(&topology.faces) {
        face.provenance = vec![source.id.clone()];
    }
    // The synthetic fixture carries solid-owned edges, not exact curve samples.
    // Retain that contract; no authored edge region is used by this planning test.
    // Exact CAD edge/parameter conformance belongs to the native meshing matrix.
    for edge in &mut mesh.topology.boundary_edges {
        edge.provenance = vec![topology.solids[0].id.clone()];
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
        if matches!(consumer, Consumer::Reduced) {
            vec!["modes", "response"]
        } else {
            vec!["modes"]
        }
    );
}
