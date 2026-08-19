use super::*;

const POINTS: [[f64; 3]; 4] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
];
const FACETS: [[usize; 3]; 4] = [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]];
const EDGES: [[usize; 2]; 6] = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]];
const FACE_UV: [[f64; 2]; 3] = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];

/// A closed four-face exact tetrahedron with portable line, plane, pcurve, trim, and mass
/// evaluators. It is intentionally backend-neutral for complete meshing-stage conformance tests.
pub fn exact_tetrahedron() -> (GeometryDocument, ExactBRepTopology, ExactEvaluatorRegistry) {
    let root = id(PersistentEntityKind::Assembly, "root");
    let part = id(PersistentEntityKind::Assembly, "part");
    let instance = id(PersistentEntityKind::Instance, "tetrahedron-instance");
    let body = id(PersistentEntityKind::Body, "tetrahedron-body");
    let lump = id(PersistentEntityKind::Lump, "tetrahedron-lump");
    let solid = id(PersistentEntityKind::Solid, "tetrahedron-solid");
    let region = id(PersistentEntityKind::Region, "tetrahedron-region");
    let shell = id(PersistentEntityKind::Shell, "tetrahedron-shell");
    let vertices = (0..POINTS.len())
        .map(|index| {
            id(
                PersistentEntityKind::Vertex,
                &format!("tetrahedron-vertex:{index}"),
            )
        })
        .collect::<Vec<_>>();
    let edges = (0..EDGES.len())
        .map(|index| {
            id(
                PersistentEntityKind::Edge,
                &format!("tetrahedron-edge:{index}"),
            )
        })
        .collect::<Vec<_>>();
    let faces = (0..FACETS.len())
        .map(|index| {
            id(
                PersistentEntityKind::Face,
                &format!("tetrahedron-face:{index}"),
            )
        })
        .collect::<Vec<_>>();
    let wires = (0..FACETS.len())
        .map(|index| {
            id(
                PersistentEntityKind::Wire,
                &format!("tetrahedron-wire:{index}"),
            )
        })
        .collect::<Vec<_>>();

    let mut coedges = Vec::new();
    let mut exact_wires = Vec::new();
    let mut pcurves = Vec::new();
    for (face_index, facet) in FACETS.iter().enumerate() {
        let mut coedge_ids = Vec::new();
        for local_edge in 0..3 {
            let from = facet[local_edge];
            let to = facet[(local_edge + 1) % 3];
            let edge_index = edge_index(from, to);
            let coedge_id = id(
                PersistentEntityKind::Coedge,
                &format!("tetrahedron-coedge:{face_index}:{local_edge}"),
            );
            let pcurve_id =
                PcurveEvaluatorId::new(format!("tetrahedron-pcurve:{face_index}:{local_edge}"))
                    .unwrap();
            let (origin_uv, destination_uv) = if from < to {
                (FACE_UV[local_edge], FACE_UV[(local_edge + 1) % 3])
            } else {
                (FACE_UV[(local_edge + 1) % 3], FACE_UV[local_edge])
            };
            pcurves.push(ExactPcurveEvaluatorRecord {
                id: pcurve_id.clone(),
                implementation: ExactPcurveImplementation::Portable {
                    definition: ExactPcurveDefinition::Line {
                        origin_uv,
                        direction_uv_per_parameter: subtract2(destination_uv, origin_uv),
                        domain: parameter(0.0, 1.0),
                    },
                },
            });
            coedge_ids.push(coedge_id.clone());
            coedges.push(ExactCoedge {
                id: coedge_id,
                face_id: faces[face_index].clone(),
                edge_id: edges[edge_index].clone(),
                orientation: if from < to {
                    TopologicalOrientation::Forward
                } else {
                    TopologicalOrientation::Reversed
                },
                pcurve_evaluator_id: pcurve_id,
                seam_image: None,
            });
        }
        exact_wires.push(ExactWire {
            id: wires[face_index].clone(),
            orientation: TopologicalOrientation::Forward,
            coedge_ids,
        });
    }

    let topology = ExactBRepTopology {
        schema_version: EXACT_BREP_TOPOLOGY_SCHEMA_VERSION,
        root_assembly_id: root.clone(),
        assemblies: vec![
            ExactAssembly {
                id: part.clone(),
                definition_digest: [31; 32],
                body_ids: vec![body.clone()],
                child_instance_ids: Vec::new(),
            },
            ExactAssembly {
                id: root.clone(),
                definition_digest: [32; 32],
                body_ids: Vec::new(),
                child_instance_ids: vec![instance.clone()],
            },
        ],
        instances: vec![ExactInstance {
            id: instance,
            parent_assembly_id: root,
            instantiated_assembly_id: part,
            transform: GeometryTransform([
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ]),
        }],
        bodies: vec![ExactBody {
            id: body,
            mass_properties_evaluator_id: MassPropertiesEvaluatorId::new(
                "tetrahedron-mass-properties",
            )
            .unwrap(),
            lump_ids: vec![lump.clone()],
            is_sheet_body: false,
            sheet_shell_ids: Vec::new(),
        }],
        lumps: vec![ExactLump {
            id: lump,
            solid_ids: vec![solid.clone()],
        }],
        solids: vec![ExactSolid {
            id: solid.clone(),
            outer_shell_id: shell.clone(),
            void_shell_ids: Vec::new(),
        }],
        regions: vec![ExactRegion {
            id: region,
            solid_id: solid,
        }],
        shells: vec![ExactShell {
            id: shell,
            orientation: TopologicalOrientation::Forward,
            face_uses: faces
                .iter()
                .map(|face| OrientedEntityUse {
                    entity_id: face.clone(),
                    orientation: TopologicalOrientation::Forward,
                })
                .collect(),
        }],
        faces: faces
            .iter()
            .enumerate()
            .map(|(index, face)| ExactFace {
                id: face.clone(),
                orientation: TopologicalOrientation::Forward,
                surface_evaluator_id: SurfaceEvaluatorId::new(format!(
                    "tetrahedron-surface:{index}"
                ))
                .unwrap(),
                trim_classifier_id: TrimClassifierId::new(format!("tetrahedron-trim:{index}"))
                    .unwrap(),
                outer_wire_id: wires[index].clone(),
                inner_wire_ids: Vec::new(),
                periodic_u: false,
                periodic_v: false,
                has_singularity: false,
            })
            .collect(),
        wires: exact_wires,
        coedges,
        edges: EDGES
            .iter()
            .enumerate()
            .map(|(index, endpoints)| ExactEdge {
                id: edges[index].clone(),
                curve_evaluator_id: CurveEvaluatorId::new(format!("tetrahedron-curve:{index}"))
                    .unwrap(),
                start_vertex_id: Some(vertices[endpoints[0]].clone()),
                end_vertex_id: Some(vertices[endpoints[1]].clone()),
                is_closed: false,
                is_periodic: false,
                is_degenerate: false,
            })
            .collect(),
        vertices: POINTS
            .iter()
            .enumerate()
            .map(|(index, point_m)| ExactVertex {
                id: vertices[index].clone(),
                point_m: *point_m,
                tolerance_m: 1.0e-10,
            })
            .collect(),
        interfaces: Vec::new(),
        contacts: Vec::new(),
    };

    let evaluators = ExactEvaluatorRegistry {
        schema_version: EXACT_EVALUATOR_REGISTRY_SCHEMA_VERSION,
        kernel_abi: "occt-v1".into(),
        curves: EDGES
            .iter()
            .enumerate()
            .map(|(index, endpoints)| ExactCurveEvaluatorRecord {
                id: CurveEvaluatorId::new(format!("tetrahedron-curve:{index}")).unwrap(),
                implementation: ExactCurveImplementation::Portable {
                    definition: ExactCurveDefinition::Line {
                        origin_m: POINTS[endpoints[0]],
                        direction_m_per_parameter: subtract3(
                            POINTS[endpoints[1]],
                            POINTS[endpoints[0]],
                        ),
                        domain: parameter(0.0, 1.0),
                    },
                },
            })
            .collect(),
        pcurves,
        surfaces: FACETS
            .iter()
            .enumerate()
            .map(|(index, facet)| ExactSurfaceEvaluatorRecord {
                id: SurfaceEvaluatorId::new(format!("tetrahedron-surface:{index}")).unwrap(),
                implementation: ExactSurfaceImplementation::Portable {
                    definition: ExactSurfaceDefinition::Plane {
                        origin_m: POINTS[facet[0]],
                        u_axis_m_per_parameter: subtract3(POINTS[facet[1]], POINTS[facet[0]]),
                        v_axis_m_per_parameter: subtract3(POINTS[facet[2]], POINTS[facet[0]]),
                        domains: [parameter(-0.25, 1.25), parameter(-0.25, 1.25)],
                    },
                },
            })
            .collect(),
        trim_classifiers: (0..FACETS.len())
            .map(|index| ExactTrimClassifierRecord {
                id: TrimClassifierId::new(format!("tetrahedron-trim:{index}")).unwrap(),
                implementation: ExactTrimClassifierImplementation::OrientedPcurveWinding,
            })
            .collect(),
        mass_properties: vec![ExactMassPropertiesRecord {
            id: MassPropertiesEvaluatorId::new("tetrahedron-mass-properties").unwrap(),
            implementation: ExactMassPropertiesImplementation::KernelValidated {
                properties: BodyMassProperties {
                    volume_m3: 1.0 / 6.0,
                    surface_area_m2: 1.5 + 3.0_f64.sqrt() * 0.5,
                    centroid_m: [0.25, 0.25, 0.25],
                    inertia_about_centroid_m5: [1.0 / 80.0; 6],
                },
                validation_digest: [33; 32],
            },
        }],
    };

    let mut model = exact_model();
    model.artifact.digest = GeometryDigest::from_bytes([34; 32]);
    model.assembly_count = 2;
    model.instance_count = 1;
    model.body_count = 1;
    model.lump_count = 1;
    model.solid_count = 1;
    model.region_count = 1;
    model.shell_count = 1;
    model.face_count = 4;
    model.wire_count = 4;
    model.coedge_count = 12;
    model.edge_count = 6;
    model.vertex_count = 4;
    let mut document = document(model);
    document.source.content_digest = GeometryDigest::from_bytes([35; 32]);
    (document, topology, evaluators)
}

fn edge_index(left: usize, right: usize) -> usize {
    let mut endpoints = [left, right];
    endpoints.sort_unstable();
    EDGES
        .iter()
        .position(|candidate| *candidate == endpoints)
        .expect("tetrahedron facet edge belongs to the canonical edge inventory")
}

fn subtract2(left: [f64; 2], right: [f64; 2]) -> [f64; 2] {
    [left[0] - right[0], left[1] - right[1]]
}

fn subtract3(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

#[cfg(test)]
mod tests {
    #[test]
    fn exact_tetrahedron_is_a_complete_closed_geometry_contract() {
        let (document, topology, evaluators) = super::exact_tetrahedron();
        let runmat_geometry_core::GeometryModel::ExactBRep { model } = &document.model else {
            panic!("tetrahedron fixture must remain exact")
        };
        topology.validate_against(model).unwrap();
        topology.validate_solid_shell_boundaries().unwrap();
        evaluators.validate_against(&topology, model).unwrap();
    }
}
