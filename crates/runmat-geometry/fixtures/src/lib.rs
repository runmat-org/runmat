//! Backend-neutral authoritative geometry fixtures for cross-crate conformance tests.

use runmat_geometry_core::*;

/// A closed circular edge on a planar face with a complete exact evaluator inventory.
pub fn exact_circle() -> (GeometryDocument, ExactBRepTopology, ExactEvaluatorRegistry) {
    let root = id(PersistentEntityKind::Assembly, "root");
    let part = id(PersistentEntityKind::Assembly, "part");
    let instance = id(PersistentEntityKind::Instance, "instance");
    let body = id(PersistentEntityKind::Body, "body");
    let lump = id(PersistentEntityKind::Lump, "lump");
    let solid = id(PersistentEntityKind::Solid, "solid");
    let shell = id(PersistentEntityKind::Shell, "shell");
    let face = id(PersistentEntityKind::Face, "face");
    let wire = id(PersistentEntityKind::Wire, "wire");
    let coedge = id(PersistentEntityKind::Coedge, "coedge");
    let edge = id(PersistentEntityKind::Edge, "edge");
    let vertex = id(PersistentEntityKind::Vertex, "vertex");
    let topology = ExactBRepTopology {
        schema_version: EXACT_BREP_TOPOLOGY_SCHEMA_VERSION,
        root_assembly_id: root.clone(),
        assemblies: vec![
            ExactAssembly {
                id: part.clone(),
                definition_digest: [1; 32],
                body_ids: vec![body.clone()],
                child_instance_ids: Vec::new(),
            },
            ExactAssembly {
                id: root.clone(),
                definition_digest: [2; 32],
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
            mass_properties_evaluator_id: MassPropertiesEvaluatorId::new("mass:1").unwrap(),
            lump_ids: vec![lump.clone()],
            is_sheet_body: false,
            sheet_shell_ids: Vec::new(),
        }],
        lumps: vec![ExactLump {
            id: lump,
            solid_ids: vec![solid.clone()],
        }],
        solids: vec![ExactSolid {
            id: solid,
            outer_shell_id: shell.clone(),
            void_shell_ids: Vec::new(),
        }],
        shells: vec![ExactShell {
            id: shell,
            orientation: TopologicalOrientation::Forward,
            face_uses: vec![OrientedEntityUse {
                entity_id: face.clone(),
                orientation: TopologicalOrientation::Forward,
            }],
        }],
        faces: vec![ExactFace {
            id: face.clone(),
            orientation: TopologicalOrientation::Forward,
            surface_evaluator_id: SurfaceEvaluatorId::new("surface:1").unwrap(),
            trim_classifier_id: TrimClassifierId::new("trim:1").unwrap(),
            outer_wire_id: wire.clone(),
            inner_wire_ids: Vec::new(),
            periodic_u: false,
            periodic_v: false,
            has_singularity: false,
        }],
        wires: vec![ExactWire {
            id: wire,
            orientation: TopologicalOrientation::Forward,
            coedge_ids: vec![coedge.clone()],
        }],
        coedges: vec![ExactCoedge {
            id: coedge,
            face_id: face,
            edge_id: edge.clone(),
            orientation: TopologicalOrientation::Forward,
            pcurve_evaluator_id: PcurveEvaluatorId::new("pcurve:1").unwrap(),
            seam_image: None,
        }],
        edges: vec![ExactEdge {
            id: edge,
            curve_evaluator_id: CurveEvaluatorId::new("curve:1").unwrap(),
            start_vertex_id: Some(vertex.clone()),
            end_vertex_id: Some(vertex.clone()),
            is_closed: true,
            is_periodic: true,
            is_degenerate: false,
        }],
        vertices: vec![ExactVertex {
            id: vertex,
            point_m: [1.0, 0.0, 0.0],
            tolerance_m: 1.0e-8,
        }],
        interfaces: Vec::new(),
        contacts: Vec::new(),
    };
    (document(exact_model()), topology, evaluators())
}

/// A closed outward-oriented tetrahedron with authoritative faceted topology.
pub fn faceted_tetrahedron() -> (GeometryDocument, FacetedSolid) {
    let shell_id = id(PersistentEntityKind::Shell, "faceted:shell:0");
    let solid = FacetedSolid {
        schema_version: FACETED_SOLID_SCHEMA_VERSION,
        vertices: [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        .into_iter()
        .enumerate()
        .map(|(index, coordinates_m)| FacetedVertex {
            id: id(
                PersistentEntityKind::Vertex,
                &format!("faceted:vertex:{index}"),
            ),
            coordinates_m,
        })
        .collect(),
        triangles: [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]
            .into_iter()
            .enumerate()
            .map(|(index, vertex_indices)| FacetedTriangle {
                id: id(PersistentEntityKind::Face, &format!("faceted:face:{index}")),
                vertex_indices,
                shell_id: shell_id.clone(),
            })
            .collect(),
        shells: vec![FacetedShell {
            id: shell_id,
            orientation: FacetedShellOrientation::Outward,
            triangle_indices: vec![0, 1, 2, 3],
        }],
    };
    let document = GeometryDocument {
        schema_version: GEOMETRY_DOCUMENT_SCHEMA_VERSION,
        source: GeometrySourceIdentity {
            content_digest: GeometryDigest::from_bytes([9; 32]),
            format: GeometrySourceFormat::Stl,
            importer_version: "runmat-faceted-fixture/1".into(),
            kernel_version: None,
            source_units: UnitSystem::Meter,
            meters_per_source_unit: 1.0,
        },
        revision: GeometryRevisionIdentity {
            revision: 1,
            persistent_mapping_version: 1,
            parent_document_digest: None,
        },
        tolerance: GeometryTolerancePolicy {
            source_tolerance_m: 0.0,
            absolute_floor_m: 1.0e-12,
            model_relative_term: 1.0e-12,
            requested_deviation_m: 1.0e-4,
            maximum_healing_displacement_m: 0.0,
        },
        healing: GeometryHealingPolicy {
            algorithm_version: "none/1".into(),
            sew: false,
            repair_orientation: false,
            consolidate_duplicates: false,
            repair_tolerance_scale_gaps: false,
            simplify_short_edges_and_sliver_faces: false,
        },
        model: GeometryModel::FacetedSolid {
            model: FacetedSolidModel {
                artifact: GeometryObjectRef {
                    digest: GeometryDigest::from_bytes([1; 32]),
                    encoded_length: 1,
                    media_type: FACETED_SOLID_MEDIA_TYPE.into(),
                    schema_version: GEOMETRY_PRIMARY_ARTIFACT_SCHEMA_VERSION,
                },
                vertex_count: 4,
                triangle_count: 4,
                shell_count: 1,
                is_watertight: true,
                is_oriented: true,
            },
        },
        display_tessellations: Vec::new(),
    };
    (document, solid)
}

fn id(kind: PersistentEntityKind, name: &str) -> PersistentEntityId {
    PersistentEntityId {
        kind,
        source_topology_id: name.into(),
        assembly_path: if kind == PersistentEntityKind::Assembly && name == "root" {
            Vec::new()
        } else {
            vec!["part".into()]
        },
    }
}

fn parameter(start: f64, end: f64) -> ParameterRange {
    ParameterRange { start, end }
}

fn evaluators() -> ExactEvaluatorRegistry {
    ExactEvaluatorRegistry {
        schema_version: EXACT_EVALUATOR_REGISTRY_SCHEMA_VERSION,
        kernel_abi: "occt-v1".into(),
        curves: vec![ExactCurveEvaluatorRecord {
            id: CurveEvaluatorId::new("curve:1").unwrap(),
            implementation: ExactCurveImplementation::Portable {
                definition: ExactCurveDefinition::Circle {
                    center_m: [0.0, 0.0, 0.0],
                    x_axis: [1.0, 0.0, 0.0],
                    y_axis: [0.0, 1.0, 0.0],
                    radius_m: 1.0,
                    domain: parameter(0.0, std::f64::consts::TAU),
                },
            },
        }],
        pcurves: vec![ExactPcurveEvaluatorRecord {
            id: PcurveEvaluatorId::new("pcurve:1").unwrap(),
            implementation: ExactPcurveImplementation::Portable {
                definition: ExactPcurveDefinition::Circle {
                    center_uv: [0.0, 0.0],
                    x_axis_uv: [1.0, 0.0],
                    y_axis_uv: [0.0, 1.0],
                    radius_uv: 1.0,
                    domain: parameter(0.0, std::f64::consts::TAU),
                },
            },
        }],
        surfaces: vec![ExactSurfaceEvaluatorRecord {
            id: SurfaceEvaluatorId::new("surface:1").unwrap(),
            implementation: ExactSurfaceImplementation::Portable {
                definition: ExactSurfaceDefinition::Plane {
                    origin_m: [0.0, 0.0, 0.0],
                    u_axis_m_per_parameter: [1.0, 0.0, 0.0],
                    v_axis_m_per_parameter: [0.0, 1.0, 0.0],
                    domains: [parameter(-2.0, 2.0), parameter(-2.0, 2.0)],
                },
            },
        }],
        trim_classifiers: vec![ExactTrimClassifierRecord {
            id: TrimClassifierId::new("trim:1").unwrap(),
            implementation: ExactTrimClassifierImplementation::OrientedPcurveWinding,
        }],
        mass_properties: vec![ExactMassPropertiesRecord {
            id: MassPropertiesEvaluatorId::new("mass:1").unwrap(),
            implementation: ExactMassPropertiesImplementation::KernelValidated {
                properties: BodyMassProperties {
                    volume_m3: 1.0,
                    surface_area_m2: 6.0,
                    centroid_m: [0.5, 0.5, 0.5],
                    inertia_about_centroid_m5: [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 0.0, 0.0, 0.0],
                },
                validation_digest: [9; 32],
            },
        }],
    }
}

fn exact_model() -> ExactBRepModel {
    ExactBRepModel {
        artifact: GeometryObjectRef {
            digest: GeometryDigest::from_bytes([1; 32]),
            encoded_length: 1,
            media_type: EXACT_BREP_MEDIA_TYPE.into(),
            schema_version: GEOMETRY_PRIMARY_ARTIFACT_SCHEMA_VERSION,
        },
        kernel_abi: "occt-v1".into(),
        capabilities: ExactGeometryCapabilities {
            curve_point: true,
            curve_tangent: true,
            curve_curvature: true,
            curve_arc_length: true,
            curve_inverse_projection: true,
            pcurve_point: true,
            pcurve_derivatives: true,
            surface_point: true,
            surface_first_derivatives: true,
            surface_second_derivatives: true,
            surface_normal: true,
            surface_principal_curvature: true,
            surface_uv_bounds: true,
            surface_periodicity: true,
            surface_closest_point: true,
            trim_domain_classification: true,
            mass_properties: true,
        },
        assembly_count: 2,
        instance_count: 1,
        body_count: 1,
        lump_count: 1,
        solid_count: 1,
        shell_count: 1,
        face_count: 1,
        wire_count: 1,
        coedge_count: 1,
        edge_count: 1,
        vertex_count: 1,
        interface_count: 0,
        contact_count: 0,
    }
}

fn document(model: ExactBRepModel) -> GeometryDocument {
    GeometryDocument {
        schema_version: GEOMETRY_DOCUMENT_SCHEMA_VERSION,
        source: GeometrySourceIdentity {
            content_digest: GeometryDigest::from_bytes([7; 32]),
            format: GeometrySourceFormat::Step,
            importer_version: "step-import/3".into(),
            kernel_version: Some("occt/7.9".into()),
            source_units: UnitSystem::Meter,
            meters_per_source_unit: 1.0,
        },
        revision: GeometryRevisionIdentity {
            revision: 1,
            persistent_mapping_version: 1,
            parent_document_digest: None,
        },
        tolerance: GeometryTolerancePolicy {
            source_tolerance_m: 1.0e-8,
            absolute_floor_m: 1.0e-10,
            model_relative_term: 1.0e-9,
            requested_deviation_m: 1.0e-5,
            maximum_healing_displacement_m: 1.0e-6,
        },
        healing: GeometryHealingPolicy {
            algorithm_version: "occt-healing/1".into(),
            sew: true,
            repair_orientation: true,
            consolidate_duplicates: true,
            repair_tolerance_scale_gaps: true,
            simplify_short_edges_and_sliver_faces: false,
        },
        model: GeometryModel::ExactBRep { model },
        display_tessellations: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_circle_is_a_complete_authoritative_fixture() {
        let (document, topology, evaluators) = exact_circle();
        document.validate().unwrap();
        let GeometryModel::ExactBRep { model } = &document.model else {
            panic!("exact-circle fixture must remain exact")
        };
        topology.validate_against(model).unwrap();
        evaluators.validate_against(&topology, model).unwrap();
    }
}
