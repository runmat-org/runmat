use super::*;

/// A spherical octant sheet bounded by two meridians and the equator, with one
/// authoritative surface singularity at the north pole.
pub fn exact_spherical_octant() -> (GeometryDocument, ExactBRepTopology, ExactEvaluatorRegistry) {
    let (mut document, mut topology, mut evaluators) = exact_circle();
    let face_id = topology.faces[0].id.clone();
    let vertex_ids = [
        id(PersistentEntityKind::Vertex, "vertex:octant:0"),
        id(PersistentEntityKind::Vertex, "vertex:octant:1"),
        id(PersistentEntityKind::Vertex, "vertex:octant:2"),
    ];
    let edge_ids = [
        id(PersistentEntityKind::Edge, "edge:octant:0"),
        id(PersistentEntityKind::Edge, "edge:octant:1"),
        id(PersistentEntityKind::Edge, "edge:octant:2"),
    ];
    let coedge_ids = [
        id(PersistentEntityKind::Coedge, "coedge:octant:0"),
        id(PersistentEntityKind::Coedge, "coedge:octant:1"),
        id(PersistentEntityKind::Coedge, "coedge:octant:2"),
    ];
    let arc_domain = parameter(0.0, std::f64::consts::FRAC_PI_2);

    topology.bodies[0].is_sheet_body = true;
    topology.bodies[0].lump_ids.clear();
    topology.bodies[0].sheet_shell_ids = vec![topology.shells[0].id.clone()];
    topology.lumps.clear();
    topology.solids.clear();
    topology.regions.clear();
    topology.faces[0].periodic_u = true;
    topology.faces[0].has_singularity = true;
    topology.wires[0].coedge_ids = coedge_ids.to_vec();
    topology.coedges = coedge_ids
        .iter()
        .zip(&edge_ids)
        .enumerate()
        .map(|(index, (coedge_id, edge_id))| ExactCoedge {
            id: coedge_id.clone(),
            face_id: face_id.clone(),
            edge_id: edge_id.clone(),
            orientation: TopologicalOrientation::Forward,
            pcurve_evaluator_id: PcurveEvaluatorId::new(format!("pcurve:{}", index + 1)).unwrap(),
            seam_image: None,
        })
        .collect();
    topology.edges = vec![
        edge(&edge_ids[0], "curve:1", &vertex_ids[0], &vertex_ids[1]),
        edge(&edge_ids[1], "curve:2", &vertex_ids[1], &vertex_ids[2]),
        edge(&edge_ids[2], "curve:3", &vertex_ids[2], &vertex_ids[0]),
    ];
    topology.vertices = vec![
        vertex(&vertex_ids[0], [1.0, 0.0, 0.0]),
        vertex(&vertex_ids[1], [0.0, 0.0, 1.0]),
        vertex(&vertex_ids[2], [0.0, 1.0, 0.0]),
    ];

    evaluators.curves = vec![
        circle_curve("curve:1", [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], arc_domain),
        circle_curve("curve:2", [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], arc_domain),
        circle_curve("curve:3", [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], arc_domain),
    ];
    evaluators.pcurves = vec![
        line_pcurve("pcurve:1", [0.0, 0.0], [0.0, 1.0], arc_domain),
        line_pcurve(
            "pcurve:2",
            [std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2],
            [0.0, -1.0],
            arc_domain,
        ),
        line_pcurve(
            "pcurve:3",
            [std::f64::consts::FRAC_PI_2, 0.0],
            [-1.0, 0.0],
            arc_domain,
        ),
    ];
    evaluators.surfaces[0].implementation = ExactSurfaceImplementation::Portable {
        definition: ExactSurfaceDefinition::Sphere {
            center_m: [0.0; 3],
            x_axis: [1.0, 0.0, 0.0],
            y_axis: [0.0, 1.0, 0.0],
            z_axis: [0.0, 0.0, 1.0],
            radius_m: 1.0,
            domains: [
                parameter(0.0, std::f64::consts::TAU),
                parameter(-std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2),
            ],
        },
    };
    let ExactMassPropertiesImplementation::KernelValidated { properties, .. } =
        &mut evaluators.mass_properties[0].implementation
    else {
        unreachable!()
    };
    properties.volume_m3 = 0.0;
    properties.surface_area_m2 = std::f64::consts::FRAC_PI_2;

    let GeometryModel::ExactBRep { model } = &mut document.model else {
        unreachable!()
    };
    model.artifact.digest = GeometryDigest::from_bytes([3; 32]);
    model.lump_count = 0;
    model.solid_count = 0;
    model.region_count = 0;
    model.coedge_count = 3;
    model.edge_count = 3;
    model.vertex_count = 3;
    document.source.content_digest = GeometryDigest::from_bytes([8; 32]);
    (document, topology, evaluators)
}

fn edge(
    id: &PersistentEntityId,
    evaluator_id: &str,
    start: &PersistentEntityId,
    end: &PersistentEntityId,
) -> ExactEdge {
    ExactEdge {
        id: id.clone(),
        curve_evaluator_id: CurveEvaluatorId::new(evaluator_id).unwrap(),
        start_vertex_id: Some(start.clone()),
        end_vertex_id: Some(end.clone()),
        is_closed: false,
        is_periodic: false,
        is_degenerate: false,
    }
}

fn vertex(id: &PersistentEntityId, point_m: [f64; 3]) -> ExactVertex {
    ExactVertex {
        id: id.clone(),
        point_m,
        tolerance_m: 1.0e-8,
    }
}

fn circle_curve(
    evaluator_id: &str,
    x_axis: [f64; 3],
    y_axis: [f64; 3],
    domain: ParameterRange,
) -> ExactCurveEvaluatorRecord {
    ExactCurveEvaluatorRecord {
        id: CurveEvaluatorId::new(evaluator_id).unwrap(),
        implementation: ExactCurveImplementation::Portable {
            definition: ExactCurveDefinition::Circle {
                center_m: [0.0; 3],
                x_axis,
                y_axis,
                radius_m: 1.0,
                domain,
                periodic: false,
            },
        },
    }
}

fn line_pcurve(
    evaluator_id: &str,
    origin_uv: [f64; 2],
    direction_uv_per_parameter: [f64; 2],
    domain: ParameterRange,
) -> ExactPcurveEvaluatorRecord {
    ExactPcurveEvaluatorRecord {
        id: PcurveEvaluatorId::new(evaluator_id).unwrap(),
        implementation: ExactPcurveImplementation::Portable {
            definition: ExactPcurveDefinition::Line {
                origin_uv,
                direction_uv_per_parameter,
                domain,
            },
        },
    }
}
