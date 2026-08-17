use super::super::super::exact_topology_tests::{model, topology};
use super::super::tests::registry;
use super::super::*;
use super::test_support::BudgetControl;
use super::*;
use crate::model::{
    ExactCoedge, ExactEdge, ExactWire, PersistentEntityId, PersistentEntityKind,
    TopologicalOrientation,
};

#[test]
fn oriented_pcurve_winding_classifies_outer_boundary_and_hole() {
    let (registry, topology, model) = circular_face_with_hole();
    let evaluator = PortableExactEvaluator::new(&registry, &topology, &model).unwrap();
    let id = TrimClassifierId::new("trim:1").unwrap();
    let control = BudgetControl::generous();
    assert_eq!(
        evaluator
            .classify(&id, [0.5, 0.0], 1.0e-9, &control)
            .unwrap(),
        TrimDomainLocation::Inside
    );
    assert_eq!(
        evaluator
            .classify(&id, [0.0, 0.0], 1.0e-9, &control)
            .unwrap(),
        TrimDomainLocation::Outside
    );
    assert_eq!(
        evaluator
            .classify(&id, [0.25, 0.0], 1.0e-9, &control)
            .unwrap(),
        TrimDomainLocation::OnBoundary
    );
    assert_eq!(
        evaluator
            .classify(&id, [1.0, 0.0], 1.0e-9, &control)
            .unwrap(),
        TrimDomainLocation::OnBoundary
    );
    assert_eq!(
        evaluator
            .classify(&id, [2.0, 0.0], 1.0e-9, &control)
            .unwrap(),
        TrimDomainLocation::Outside
    );
}

#[test]
fn classifier_and_mass_properties_preserve_typed_failures() {
    let portable_registry = registry();
    let topology = topology();
    let evaluator = PortableExactEvaluator::new(&portable_registry, &topology, &model()).unwrap();
    let trim_id = TrimClassifierId::new("trim:1").unwrap();
    let error = evaluator
        .classify(
            &trim_id,
            [0.0, 0.0],
            1.0e-9,
            &BudgetControl::with_limits(u64::MAX, 0, u64::MAX),
        )
        .unwrap_err();
    assert_eq!(error.kind, GeometryEvaluationErrorKind::BudgetExceeded);
    let error = evaluator
        .classify(
            &trim_id,
            [0.0, 0.0],
            1.0e-9,
            &BudgetControl::allocation_limited(0),
        )
        .unwrap_err();
    assert_eq!(error.kind, GeometryEvaluationErrorKind::BudgetExceeded);
    let error = evaluator
        .classify(&trim_id, [0.0, 0.0], 1.0e-9, &BudgetControl::cancelled())
        .unwrap_err();
    assert_eq!(error.kind, GeometryEvaluationErrorKind::Cancelled);
    let error = evaluator
        .classify(&trim_id, [0.0, 0.0], 0.0, &BudgetControl::generous())
        .unwrap_err();
    assert_eq!(error.kind, GeometryEvaluationErrorKind::InvalidResult);

    let properties = evaluator
        .mass_properties(
            &MassPropertiesEvaluatorId::new("mass:1").unwrap(),
            &BudgetControl::generous(),
        )
        .unwrap();
    assert_eq!(properties.volume_m3, 1.0);
    assert_eq!(properties.centroid_m, [0.5, 0.5, 0.5]);
    let error = evaluator
        .mass_properties(
            &MassPropertiesEvaluatorId::new("mass:1").unwrap(),
            &BudgetControl::cancelled(),
        )
        .unwrap_err();
    assert_eq!(error.kind, GeometryEvaluationErrorKind::Cancelled);

    let mut kernel_registry = registry();
    kernel_registry.trim_classifiers[0].implementation =
        ExactTrimClassifierImplementation::Kernel {
            reference: KernelEvaluatorRef {
                entity_token: "face:1".into(),
                representation_digest: [4; 32],
            },
        };
    kernel_registry.mass_properties[0].implementation = ExactMassPropertiesImplementation::Kernel {
        reference: KernelEvaluatorRef {
            entity_token: "body:1".into(),
            representation_digest: [5; 32],
        },
    };
    let evaluator = PortableExactEvaluator::new(&kernel_registry, &topology, &model()).unwrap();
    let error = evaluator
        .classify(&trim_id, [0.0, 0.0], 1.0e-9, &BudgetControl::generous())
        .unwrap_err();
    assert_eq!(error.kind, GeometryEvaluationErrorKind::KernelUnavailable);
    let error = evaluator
        .mass_properties(
            &MassPropertiesEvaluatorId::new("mass:1").unwrap(),
            &BudgetControl::generous(),
        )
        .unwrap_err();
    assert_eq!(error.kind, GeometryEvaluationErrorKind::KernelUnavailable);
}

fn circular_face_with_hole() -> (ExactEvaluatorRegistry, ExactBRepTopology, ExactBRepModel) {
    let mut topology = topology();
    let inner_wire_id = part_id(PersistentEntityKind::Wire, "wire-hole");
    let inner_coedge_id = part_id(PersistentEntityKind::Coedge, "coedge-hole");
    let inner_edge_id = part_id(PersistentEntityKind::Edge, "edge-hole");
    topology.faces[0].inner_wire_ids.push(inner_wire_id.clone());
    topology.wires.push(ExactWire {
        id: inner_wire_id,
        orientation: TopologicalOrientation::Reversed,
        coedge_ids: vec![inner_coedge_id.clone()],
    });
    topology.coedges.push(ExactCoedge {
        id: inner_coedge_id,
        face_id: topology.faces[0].id.clone(),
        edge_id: inner_edge_id.clone(),
        orientation: TopologicalOrientation::Forward,
        pcurve_evaluator_id: PcurveEvaluatorId::new("pcurve:2").unwrap(),
        seam_image: None,
    });
    topology.edges.push(ExactEdge {
        id: inner_edge_id,
        curve_evaluator_id: CurveEvaluatorId::new("curve:2").unwrap(),
        start_vertex_id: Some(topology.vertices[0].id.clone()),
        end_vertex_id: Some(topology.vertices[0].id.clone()),
        is_closed: true,
        is_periodic: true,
        is_degenerate: false,
    });
    topology.wires.sort_by(|left, right| left.id.cmp(&right.id));
    topology
        .coedges
        .sort_by(|left, right| left.id.cmp(&right.id));
    topology.edges.sort_by(|left, right| left.id.cmp(&right.id));

    let domain = ParameterRange {
        start: 0.0,
        end: std::f64::consts::TAU,
    };
    let mut registry = registry();
    registry.curves.push(ExactCurveEvaluatorRecord {
        id: CurveEvaluatorId::new("curve:2").unwrap(),
        implementation: ExactCurveImplementation::Portable {
            definition: ExactCurveDefinition::Circle {
                center_m: [0.0; 3],
                x_axis: [1.0, 0.0, 0.0],
                y_axis: [0.0, 1.0, 0.0],
                radius_m: 0.25,
                domain,
            },
        },
    });
    registry.pcurves.push(ExactPcurveEvaluatorRecord {
        id: PcurveEvaluatorId::new("pcurve:2").unwrap(),
        implementation: ExactPcurveImplementation::Portable {
            definition: ExactPcurveDefinition::Circle {
                center_uv: [0.0, 0.0],
                x_axis_uv: [1.0, 0.0],
                y_axis_uv: [0.0, 1.0],
                radius_uv: 0.25,
                domain,
            },
        },
    });
    registry
        .curves
        .sort_by(|left, right| left.id.cmp(&right.id));
    registry
        .pcurves
        .sort_by(|left, right| left.id.cmp(&right.id));

    let mut model = model();
    model.wire_count = 2;
    model.coedge_count = 2;
    model.edge_count = 2;
    (registry, topology, model)
}

fn part_id(kind: PersistentEntityKind, source_topology_id: &str) -> PersistentEntityId {
    PersistentEntityId {
        kind,
        source_topology_id: source_topology_id.into(),
        assembly_path: vec!["root".into(), "instance".into()],
    }
}
