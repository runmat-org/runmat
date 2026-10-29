//! Canonical compiler fixtures for typed physical quantities and canonical entity selectors.
//! The wire contract includes quantity field names and selector-derived identities, so the
//! pre-quantity/pre-selector byte hashes are not compatible. These goldens pin the combined
//! contract; independent checks below retain row counts, redundancy, authored-order invariance
//! and cold round trips. They do not establish numerical solver accuracy.
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
    PersistentEntityId::new(kind, name.into(), vec!["root".into()])
}

pub(super) fn mesh(order: ElementOrder) -> SolverMeshArtifact {
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
    let mut model =
        mixed_solid_beam_shell_model(mesh.topology.nodes[0].coordinates_m.canonical_value());
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
        explicit.coordinates_m = runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Position,
            _,
        >::new(node.coordinates_m.canonical_value());
        explicit.solid_attachment = Some(ConformalSolidAttachment::GeometryVertex { entity });
    }
    mesh.canonical_digest = StableDigest::ZERO;
    mesh.seal_canonical_digest().unwrap();
    model.constraints = vec![AnalysisConstraint::Structural(StructuralConstraint {
        constraint_id: "shared_shell_boundary".into(),
        frame: CoordinateFrame::Global,
        kind: StructuralConstraintKind::Fixed {
            target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("skin")),
        },
    })];
    (model, mesh)
}

pub(super) fn cases() -> Vec<(&'static str, AnalysisModel, Option<SolverMeshArtifact>)> {
    let (tet4, mesh4) = solid(ElementOrder::Tet4);
    let (tet10, mesh10) = solid(ElementOrder::Tet10);
    let mixed_mesh = mesh(ElementOrder::Tet4);
    let mixed =
        mixed_solid_beam_model(mixed_mesh.topology.nodes[0].coordinates_m.canonical_value());
    let mut generic = mixed.clone();
    generic.constraints.retain(|constraint| {
        constraint.structural().unwrap().constraint_id == "beam_root_rotation"
    });
    let mut mixed_shell =
        mixed_solid_beam_shell_model(mixed_mesh.topology.nodes[0].coordinates_m.canonical_value());
    mixed_shell
        .constraints
        .push(AnalysisConstraint::Structural(StructuralConstraint {
            constraint_id: "shell_symmetry".into(),
            frame: CoordinateFrame::BoundaryNormalTangent,
            kind: StructuralConstraintKind::Symmetry {
                target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("skin")),
            },
        }));
    let mut redundant_solid = tet4.clone();
    redundant_solid.constraints = vec![
        prescribed(
            "first",
            RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("root")),
            0.0,
        ),
        prescribed(
            "second",
            RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("root")),
            0.0,
        ),
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
fn compiler_canonical_bytes_match_typed_quantity_and_selector_goldens() {
    let mut differences = Vec::new();
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
        if bytes.len() != expected_length || digest != expected_digest {
            differences.push(format!("{name}: length={} digest={digest}", bytes.len()));
        }
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
    assert!(
        differences.is_empty(),
        "canonical fixture differences:\n{}",
        differences.join("\n")
    );
}

fn golden(name: &str) -> (usize, &'static str) {
    match name {
        "solid_tet4" => (
            5051,
            "a3d23b5a4b7e222aa18f0cbd9d698c36c62e15ddd1f327d9df7b769a7db77c5c",
        ),
        "solid_tet10" => (
            11728,
            "36c02b66837b42ead37aef9b946b8bd0f178280b7c7a93a0257f45c28e58c414",
        ),
        "explicit_beam" => (
            3111,
            "a075db220289c68cbe47ee7d693b84223a9468fc99cce1f13ccdf964edfb7ce5",
        ),
        "explicit_shell" => (
            3125,
            "9e6131f35b15b7964c8de2da343ae437f167e0e0ab1deeab087cf26613349e70",
        ),
        "mixed_solid_beam" => (
            7582,
            "9ff207c4f266517a63a95e4fe63e39f4730241763db32c15847a01e408d44729",
        ),
        "mixed_only_generic" => (
            2053,
            "496c932e693058be71845840e5be89cb63a5fb0ee6fb17ecfe4be3a039f77cf3",
        ),
        "mixed_shell_symmetry" => (
            9321,
            "fae9fac2654191619cef23d23f9b1e9c73c1f3f74556cc3642e838236d042287",
        ),
        "mixed_shared_shell_boundary" => (
            7254,
            "9e0d4448ac7d6ca69eba0295aba5cda600f03076ead99dad1d39107ef296cc70",
        ),
        "solid_redundant" => (
            4423,
            "f614ab766f8a1d43b79c3c7935f2560d2f340fc647c836e2878487d48b060961",
        ),
        "mixed_generic_redundant" => (
            2302,
            "cbb84289e408e4c9aa38b6f97a5243d71a8bf4a1ed6d12ab03151da595eccbd5",
        ),
        _ => panic!("missing canonical compilation golden for {name}"),
    }
}

#[test]
fn conflicting_solid_and_generic_prescriptions_retain_typed_provenance() {
    for mixed in [false, true] {
        let (mut model, mesh) = solid(ElementOrder::Tet4);
        if mixed {
            model = mixed_solid_beam_model(mesh.topology.nodes[0].coordinates_m.canonical_value());
        }
        let target = if mixed {
            RegionTarget::point("node:102")
        } else {
            RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("root"))
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
    let mut model = mixed_solid_beam_model(mesh.topology.nodes[0].coordinates_m.canonical_value());
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

#[test]
fn counted_unique_ids_preserve_first_authored_duplicate_diagnostic() {
    let mesh = mesh(ElementOrder::Tet4);
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let mut constraints = Vec::new();
    for id in ["z", "z", "a", "a"] {
        constraints.push(prescribed(
            id,
            RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("root")),
            0.0,
        ));
    }
    model.constraints = constraints;
    assert_eq!(
        CanonicalStructuralConstraintPlan::compile(&model, &mesh).unwrap_err(),
        ConstraintPlanError::DuplicateConstraintId {
            constraint_id: "z".into()
        }
    );
}
