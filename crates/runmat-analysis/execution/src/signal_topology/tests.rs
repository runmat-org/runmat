use super::*;
use runmat_analysis_core::{
    BodyForceDensity, Signal, SignalApproximationEvidence, SignalId, SignalPrecision,
    SignalProvenance, SignalRepresentation, SignalValue, Temperature,
};
use runmat_analysis_fea::topology::AnalysisProjection;
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

fn signal(mesh: &SolverMeshArtifact, thermal: bool) -> PhysicalSignal {
    let rows = mesh.topology.nodes.len();
    let shape = SignalShape::TopologyField {
        topology_id: Digest::from_bytes(*mesh.canonical_digest.bytes()).to_string(),
        location: FieldLocation::Node,
        component_count: if thermal { 1 } else { 3 },
    };
    let evidence = SignalApproximationEvidence {
        max_absolute_error: Some(0.0),
        max_relative_error: Some(0.0),
        derivative_max_absolute_error: None,
        sample_count: 1,
    };
    if thermal {
        PhysicalSignal::Temperature(Signal {
            signal_id: SignalId("temperature".into()),
            shape,
            frame: None,
            representation: SignalRepresentation::Constant(SignalValue::Tensor {
                shape: vec![rows, 1],
                values: vec![Temperature::new(293.15); rows],
            }),
            precision: SignalPrecision::Real64,
            evidence,
            provenance: SignalProvenance::Authored,
        })
    } else {
        PhysicalSignal::BodyForceDensity(Signal {
            signal_id: SignalId("body".into()),
            shape,
            frame: None,
            representation: SignalRepresentation::Constant(SignalValue::Tensor {
                shape: vec![rows, 3],
                values: vec![BodyForceDensity::new(2.0); rows * 3],
            }),
            precision: SignalPrecision::Real64,
            evidence,
            provenance: SignalProvenance::Authored,
        })
    }
}

fn shape_mut(signal: &mut PhysicalSignal) -> &mut SignalShape {
    match signal {
        PhysicalSignal::BodyForceDensity(signal) => &mut signal.shape,
        PhysicalSignal::Temperature(signal) => &mut signal.shape,
        _ => unreachable!("test signal quantity"),
    }
}

#[test]
fn thermal_and_body_rows_bind_exact_tet4_tet10_and_survive_transport() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let reopened =
            SolverMeshArtifact::canonical_decode(&mesh.canonical_encode().unwrap()).unwrap();
        let projection = AnalysisProjection::derive(&mesh).unwrap();
        assert_eq!(projection.nodes.len(), mesh.topology.nodes.len());
        for (projected, authoritative) in projection.nodes.iter().zip(&mesh.topology.nodes) {
            assert_eq!(projected.node_id, authoritative.node_id);
            assert_eq!(projected.stable_identity, authoritative.stable_identity);
        }
        for thermal in [false, true] {
            let signal = signal(&mesh, thermal);
            let binding = SignalTopologyBinding::derive(&signal, &mesh)
                .unwrap()
                .unwrap();
            assert_eq!(
                binding.entity_count(),
                if order == ElementOrder::Tet4 { 4 } else { 10 }
            );
            assert_eq!(binding.component_count(), if thermal { 1 } else { 3 });
            assert_eq!(
                binding.scalar_component_count().unwrap(),
                binding.entity_count() * binding.component_count()
            );
            assert_eq!(binding.location(), FieldLocation::Node);
            assert_eq!(binding.mesh_digest().bytes(), mesh.canonical_digest.bytes());
            binding.validate_shape(&signal).unwrap();
            binding.validate_for(&signal, &reopened).unwrap();
            let bytes = serde_json::to_vec(&binding).unwrap();
            let transported: SignalTopologyBinding = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(binding, transported);
            transported.validate_for(&signal, &reopened).unwrap();
            let limits = runmat_canonical_codec::CanonicalLimits::new(1024, 16, 128, 8);
            let bytes = runmat_canonical_codec::encode_contract(
                b"runmat-test-signal-topology/v1\0",
                "signal-topology",
                &binding,
                limits,
            )
            .unwrap();
            let canonical: SignalTopologyBinding = runmat_canonical_codec::decode_contract(
                b"runmat-test-signal-topology/v1\0",
                "signal-topology",
                &bytes,
                limits,
            )
            .unwrap();
            assert_eq!(canonical, binding);
            canonical.validate_for(&signal, &reopened).unwrap();
        }
    }
}

#[test]
fn mesh_domains_have_explicit_independent_order_and_counts() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let mut signal = signal(&mesh, true);
        let mut digests = std::collections::BTreeSet::new();
        for (location, count) in [
            (FieldLocation::Node, mesh.topology.nodes.len()),
            (FieldLocation::Cell, mesh.topology.volume_elements.len()),
            (FieldLocation::Element, mesh.topology.volume_elements.len()),
            (FieldLocation::Face, mesh.topology.boundary_faces.len()),
            (FieldLocation::Edge, mesh.topology.boundary_edges.len()),
        ] {
            let SignalShape::TopologyField {
                location: declared, ..
            } = shape_mut(&mut signal)
            else {
                unreachable!()
            };
            *declared = location;
            let PhysicalSignal::Temperature(value) = &mut signal else {
                unreachable!("thermal domain fixture")
            };
            value.representation = SignalRepresentation::Constant(SignalValue::Tensor {
                shape: vec![count, 1],
                values: vec![Temperature::new(293.15); count],
            });
            let binding = SignalTopologyBinding::derive(&signal, &mesh)
                .unwrap()
                .unwrap();
            assert_eq!(binding.entity_count(), count);
            binding.validate_for(&signal, &mesh).unwrap();
            assert!(digests.insert(binding.order_digest()));
        }
    }
}

#[test]
fn foreign_mesh_shape_count_width_and_order_never_rebind() {
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let signal = signal(&mesh, false);
    let binding = SignalTopologyBinding::derive(&signal, &mesh)
        .unwrap()
        .unwrap();
    for (key, replacement) in [
        (
            "mesh_digest",
            serde_json::to_value(Digest::sha256("foreign mesh")).unwrap(),
        ),
        (
            "location",
            serde_json::to_value(FieldLocation::Face).unwrap(),
        ),
        (
            "entity_count",
            serde_json::json!(binding.entity_count() + 1),
        ),
        ("component_count", serde_json::json!(2)),
        (
            "order_digest",
            serde_json::to_value(Digest::sha256("foreign rows")).unwrap(),
        ),
    ] {
        let mut wire = serde_json::to_value(&binding).unwrap();
        wire[key] = replacement;
        let forged: SignalTopologyBinding = serde_json::from_value(wire).unwrap();
        assert!(
            forged.validate_for(&signal, &mesh).is_err(),
            "substitution {key}"
        );
    }
    // Tet4 has four nodes and four faces: shape/count agreement alone cannot
    // authenticate a substituted domain carrying the original node-order digest.
    let mut face_signal = signal.clone();
    let SignalShape::TopologyField { location, .. } = shape_mut(&mut face_signal) else {
        unreachable!()
    };
    *location = FieldLocation::Face;
    let mut face_binding = binding.clone();
    face_binding.location = FieldLocation::Face;
    face_binding.validate_shape(&face_signal).unwrap();
    assert!(face_binding.validate_for(&face_signal, &mesh).is_err());
    assert!(binding
        .validate_for(
            &signal,
            &canonical_tetrahedron_solver_mesh(ElementOrder::Tet10)
        )
        .is_err());
    let mut wrong = signal.clone();
    let SignalShape::TopologyField { topology_id, .. } = shape_mut(&mut wrong) else {
        unreachable!()
    };
    *topology_id = "named-but-unresolved-topology".into();
    assert!(SignalTopologyBinding::derive(&wrong, &mesh).is_err());
}

#[test]
fn non_topology_shapes_stay_unbound_and_unsupported_locations_reject() {
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let mut signal = signal(&mesh, false);
    let binding = SignalTopologyBinding::derive(&signal, &mesh)
        .unwrap()
        .unwrap();
    for shape in [
        SignalShape::Scalar,
        SignalShape::Vector3,
        SignalShape::Tensor(vec![2, 3]),
    ] {
        *shape_mut(&mut signal) = shape;
        assert_eq!(SignalTopologyBinding::derive(&signal, &mesh).unwrap(), None);
        assert!(binding.validate_shape(&signal).is_err());
    }
    for location in [
        FieldLocation::Point,
        FieldLocation::IntegrationPoint,
        FieldLocation::Interface,
        FieldLocation::Connector,
        FieldLocation::Constraint,
        FieldLocation::ReferencePoint,
        FieldLocation::Mode,
        FieldLocation::Global,
    ] {
        *shape_mut(&mut signal) = SignalShape::TopologyField {
            topology_id: binding.mesh_digest().to_string(),
            location,
            component_count: 3,
        };
        assert!(SignalTopologyBinding::derive(&signal, &mesh).is_err());
    }
}

#[test]
fn dimensions_overflow_and_unknown_transport_fields_reject() {
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let signal = signal(&mesh, false);
    let binding = SignalTopologyBinding::derive(&signal, &mesh)
        .unwrap()
        .unwrap();
    for (key, value) in [
        ("entity_count", 0usize),
        ("component_count", 0),
        ("entity_count", usize::MAX),
    ] {
        let mut wire = serde_json::to_value(&binding).unwrap();
        wire[key] = serde_json::json!(value);
        assert!(serde_json::from_value::<SignalTopologyBinding>(wire).is_err());
    }
    let mut wire = serde_json::to_value(&binding).unwrap();
    wire["untrusted_order"] = serde_json::json!(true);
    assert!(serde_json::from_value::<SignalTopologyBinding>(wire).is_err());
}
