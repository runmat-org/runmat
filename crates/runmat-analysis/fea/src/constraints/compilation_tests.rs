//! Canonical byte goldens captured from the pre-shared-pipeline compiler. These exercise the
//! compiler only, independently of numerical solves and elapsed-time/memory heuristics.
//! Baseline source SHA-1: compile.rs bf5c6204bc3b46004808d5645dd2b43007117278;
//! mixed.rs fd8c4e97b68f173fdb5abc16987eb2c5a8555d21. The goldens pin SHA-256 of
//! canonical plan bytes, including topology, groups, row identities, and rank evidence.
use super::*;
use crate::fixtures::{
    fixture_model, mixed_solid_beam_model, mixed_solid_beam_shell_model, FixtureId,
};
use runmat_analysis_core::{
    AnalysisConstraint, AnalysisModel, ConformalSolidAttachment, CoordinateFrame, Displacement,
    PrescribedStructuralDofs, PrescribedValue, RegionTarget, StructuralConstraint,
    StructuralConstraintKind, StructuralTranslationDof, TranslationDofValue,
};
use runmat_meshing_core::{
    fixtures::canonical_tetrahedron_solver_mesh, ElementOrder, PersistentEntityId,
    PersistentEntityKind, SolverMeshArtifact, StableDigest,
};

fn provenance(kind: PersistentEntityKind, name: &str) -> PersistentEntityId {
    PersistentEntityId {
        kind,
        source_topology_id: name.into(),
        assembly_path: vec!["root".into()],
    }
}

fn mesh(order: ElementOrder) -> SolverMeshArtifact {
    let mut mesh = canonical_tetrahedron_solver_mesh(order);
    mesh.topology.nodes[0]
        .provenance
        .push(provenance(PersistentEntityKind::Vertex, "beam_root"));
    mesh.topology.nodes[0].provenance.sort();
    for face in &mut mesh.topology.boundary_faces {
        face.provenance
            .push(provenance(PersistentEntityKind::Face, "root"));
        face.provenance.sort();
    }
    mesh.canonical_digest = StableDigest::ZERO;
    mesh.seal_canonical_digest().unwrap();
    mesh
}

fn prescribed(id: &str, target: RegionTarget, value: f64) -> AnalysisConstraint {
    AnalysisConstraint::Structural(StructuralConstraint {
        constraint_id: id.into(),
        frame: CoordinateFrame::Global,
        kind: StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
            target,
            translations: vec![TranslationDofValue {
                dof: StructuralTranslationDof::Ux,
                value: PrescribedValue::Constant(Displacement::new(value)),
            }],
            rotations: Vec::new(),
        }),
    })
}

fn solid(order: ElementOrder) -> (AnalysisModel, SolverMeshArtifact) {
    let mesh = mesh(order);
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.geometry_revision = mesh.geometry.geometry_revision as u32;
    (model, mesh)
}

fn shared_shell_boundary() -> (AnalysisModel, SolverMeshArtifact) {
    let mut mesh = mesh(ElementOrder::Tet4);
    let mut model = mixed_solid_beam_shell_model(mesh.topology.nodes[0].coordinates_m);
    let face_nodes = mesh.topology.boundary_faces[0].node_ids.clone();
    mesh.topology.boundary_faces[0]
        .provenance
        .push(provenance(PersistentEntityKind::Face, "skin"));
    mesh.topology.boundary_faces[0].provenance.sort();
    for (ordinal, explicit_id) in [101, 103, 104].into_iter().enumerate() {
        let node = mesh
            .topology
            .nodes
            .iter_mut()
            .find(|node| node.node_id == face_nodes[ordinal])
            .unwrap();
        let entity = provenance(
            PersistentEntityKind::Vertex,
            &format!("skin_vertex_{ordinal}"),
        );
        node.provenance.push(entity.clone());
        node.provenance.sort();
        let explicit = model
            .structural
            .as_mut()
            .unwrap()
            .nodes
            .iter_mut()
            .find(|node| node.node_id == explicit_id)
            .unwrap();
        explicit.coordinates_m = node.coordinates_m;
        explicit.solid_attachment = Some(ConformalSolidAttachment::GeometryVertex { entity });
    }
    mesh.canonical_digest = StableDigest::ZERO;
    mesh.seal_canonical_digest().unwrap();
    model.constraints = vec![AnalysisConstraint::Structural(StructuralConstraint {
        constraint_id: "shared_shell_boundary".into(),
        frame: CoordinateFrame::Global,
        kind: StructuralConstraintKind::Fixed {
            target: RegionTarget::boundary("skin"),
        },
    })];
    (model, mesh)
}

fn cases() -> Vec<(&'static str, AnalysisModel, Option<SolverMeshArtifact>)> {
    let (tet4, mesh4) = solid(ElementOrder::Tet4);
    let (tet10, mesh10) = solid(ElementOrder::Tet10);
    let mixed_mesh = mesh(ElementOrder::Tet4);
    let mixed = mixed_solid_beam_model(mixed_mesh.topology.nodes[0].coordinates_m);
    let mut generic = mixed.clone();
    generic.constraints.retain(|constraint| {
        constraint.structural().unwrap().constraint_id == "beam_root_rotation"
    });
    let mut mixed_shell = mixed_solid_beam_shell_model(mixed_mesh.topology.nodes[0].coordinates_m);
    mixed_shell
        .constraints
        .push(AnalysisConstraint::Structural(StructuralConstraint {
            constraint_id: "shell_symmetry".into(),
            frame: CoordinateFrame::BoundaryNormalTangent,
            kind: StructuralConstraintKind::Symmetry {
                target: RegionTarget::boundary("skin"),
            },
        }));
    let mut redundant_solid = tet4.clone();
    redundant_solid.constraints = vec![
        prescribed("first", RegionTarget::boundary("root"), 0.0),
        prescribed("second", RegionTarget::boundary("root"), 0.0),
    ];
    let mut redundant_mixed = mixed.clone();
    redundant_mixed.constraints = vec![
        prescribed("first", RegionTarget::point("node:102"), 0.0),
        prescribed("second", RegionTarget::point("node:102"), 0.0),
    ];
    let (shared, shared_mesh) = shared_shell_boundary();
    vec![
        ("solid_tet4", tet4, Some(mesh4.clone())),
        ("solid_tet10", tet10, Some(mesh10)),
        (
            "explicit_beam",
            fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference),
            None,
        ),
        (
            "explicit_shell",
            fixture_model(FixtureId::StructuralShellPlateMomentReference),
            None,
        ),
        ("mixed_solid_beam", mixed, Some(mixed_mesh.clone())),
        ("mixed_only_generic", generic, Some(mixed_mesh.clone())),
        (
            "mixed_shell_symmetry",
            mixed_shell,
            Some(mixed_mesh.clone()),
        ),
        ("mixed_shared_shell_boundary", shared, Some(shared_mesh)),
        ("solid_redundant", redundant_solid, Some(mesh4)),
        ("mixed_generic_redundant", redundant_mixed, Some(mixed_mesh)),
    ]
}

#[test]
fn compiler_canonical_bytes_match_pre_shared_pipeline_goldens() {
    for (name, mut model, mesh) in cases() {
        let baseline = CanonicalStructuralConstraintPlan::compile_for_model(&model, mesh.as_ref())
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let bytes = baseline.canonical_bytes().unwrap();
        let digest: String = runmat_execution::Digest::sha256(&bytes)
            .bytes()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        let (expected_length, expected_digest) = golden(name);
        assert_eq!(
            bytes.len(),
            expected_length,
            "{name}: canonical byte length"
        );
        assert_eq!(digest, expected_digest, "{name}: canonical byte digest");
        model.constraints.reverse();
        let reordered =
            CanonicalStructuralConstraintPlan::compile_for_model(&model, mesh.as_ref()).unwrap();
        assert_eq!(
            bytes,
            reordered.canonical_bytes().unwrap(),
            "{name}: authored order changed bytes"
        );
        let decoded = CanonicalStructuralConstraintPlan::from_canonical_bytes(&bytes).unwrap();
        assert_eq!(baseline, decoded, "{name}: canonical roundtrip");
        if name.ends_with("redundant") {
            assert!(baseline.evidence.redundant_row_count > 0, "{name}");
        }
        if name == "mixed_only_generic" {
            assert_eq!(baseline.groups.len(), 1);
            assert_eq!(baseline.groups[0].constraint_id, "beam_root_rotation");
        }
        if name == "mixed_shared_shell_boundary" {
            assert_eq!(baseline.rows.len(), 18);
        }
    }
}

fn golden(name: &str) -> (usize, &'static str) {
    match name {
        "solid_tet4" => (
            4961,
            "50411bd494749ef115b45614b13b1d3613e018dddbbaf4bda727bd3b751f311c",
        ),
        "solid_tet10" => (
            11638,
            "802e2a86a34abe88c4511f908d5b970b1cd4e742c4d04761d47c72ae75d4a88b",
        ),
        "explicit_beam" => (
            3111,
            "e3af736ee0eda2178c212c6b6c95fb8a5042aaa975b9c225716b413cf077ebd4",
        ),
        "explicit_shell" => (
            3132,
            "1d9a60e1adda238b83631cfc04cfbaea52f65fc526c38e6cab7615292ecbe6d6",
        ),
        "mixed_solid_beam" => (
            6593,
            "e6140918f682ed230c4ebda93d45326b217d13fa8d426e2fa04bd846361b25fa",
        ),
        "mixed_only_generic" => (
            2049,
            "96eaa3289d16ac769b7f0b6a4794319cfc607669047ff68d16f161f17e95a516",
        ),
        "mixed_shell_symmetry" => (
            8248,
            "c7a5b9636d9bb72bcda228e6d9350d858e5e9d7799159cdaf3e414c98d989b6e",
        ),
        "mixed_shared_shell_boundary" => (
            7171,
            "113793dfcda25551c2bc275a1e04f460e8265dbe36ad5cf48d7260bbedb4f88a",
        ),
        "solid_redundant" => (
            4247,
            "3fa136852a5eb41437142f7f69f3546a7761e01831e9662ebf8ba40d95c2d3ca",
        ),
        "mixed_generic_redundant" => (
            2298,
            "729ca1c67cd579c76ca3210b83f99d016747fe6ad32b481c396ee390aff1c7c0",
        ),
        _ => panic!("missing canonical compilation golden for {name}"),
    }
}

#[test]
fn conflicting_solid_and_generic_prescriptions_retain_typed_provenance() {
    for mixed in [false, true] {
        let (mut model, mesh) = solid(ElementOrder::Tet4);
        if mixed {
            model = mixed_solid_beam_model(mesh.topology.nodes[0].coordinates_m);
        }
        let target = if mixed {
            RegionTarget::point("node:102")
        } else {
            RegionTarget::boundary("root")
        };
        model.constraints = vec![
            prescribed("first", target.clone(), 0.0),
            prescribed("second", target, 0.1),
        ];
        for reverse in [false, true] {
            if reverse {
                model.constraints.reverse();
            }
            let error = CanonicalStructuralConstraintPlan::compile_for_model(&model, Some(&mesh))
                .unwrap_err();
            assert!(matches!(error, ConstraintPlanError::ConflictingRow {
                first_constraint_id, second_constraint_id,
                conflicting_dof: Some(CanonicalStructuralDofAddress {
                    owner: CanonicalStructuralDofOwner::StructuralNode(identity),
                    dof: CanonicalStructuralDof::Translation(StructuralTranslationDof::Ux),
                }),
            } if first_constraint_id == "first" && second_constraint_id == "second" && identity != StableDigest::ZERO));
        }
    }
}

#[test]
fn duplicate_ids_are_rejected_across_solid_and_generic_lowering() {
    let mesh = mesh(ElementOrder::Tet4);
    let mut model = mixed_solid_beam_model(mesh.topology.nodes[0].coordinates_m);
    for constraint in &mut model.constraints {
        if let AnalysisConstraint::Structural(constraint) = constraint {
            constraint.constraint_id = "duplicate".into();
        }
    }
    for reverse in [false, true] {
        if reverse {
            model.constraints.reverse();
        }
        assert!(
            matches!(CanonicalStructuralConstraintPlan::compile(&model, &mesh),
            Err(ConstraintPlanError::DuplicateConstraintId { constraint_id }) if constraint_id == "duplicate")
        );
    }
}

#[test]
fn every_topology_can_finalize_an_empty_constraint_plan() {
    for (_, mut model, mesh) in cases() {
        model.constraints.clear();
        let plan =
            CanonicalStructuralConstraintPlan::compile_for_model(&model, mesh.as_ref()).unwrap();
        assert!(plan.rows.is_empty());
        assert!(plan.groups.is_empty());
        assert!(plan.reference_points.is_empty());
        assert_eq!(plan.enforcement, ConstraintEnforcement::DirectElimination);
        assert_eq!(plan.evidence.rank, 0);
        plan.validate().unwrap();
    }
}

#[test]
fn explicit_compilation_checks_malformed_frame_before_equation_lowering() {
    let mut model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
    let AnalysisConstraint::Structural(constraint) = &mut model.constraints[0] else {
        panic!("structural fixture")
    };
    constraint.frame = CoordinateFrame::Cartesian(runmat_analysis_core::CartesianFrame {
        frame_id: "degenerate".into(),
        origin: runmat_analysis_core::Position::new([0.0; 3]),
        x_axis: [0.0; 3],
        y_axis: [0.0; 3],
    });
    assert!(matches!(
        CanonicalStructuralConstraintPlan::compile_for_model(&model, None),
        Err(ConstraintPlanError::DegenerateFrame { .. })
    ));
}
