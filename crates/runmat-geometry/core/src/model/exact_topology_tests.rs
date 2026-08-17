use super::*;

fn id(kind: PersistentEntityKind, name: &str) -> PersistentEntityId {
    PersistentEntityId {
        kind,
        source_topology_id: name.into(),
        assembly_path: vec!["root".into()],
    }
}

fn part_id(kind: PersistentEntityKind, name: &str) -> PersistentEntityId {
    PersistentEntityId {
        kind,
        source_topology_id: name.into(),
        assembly_path: vec!["root".into(), "instance".into()],
    }
}

fn curve_evaluator(name: &str) -> CurveEvaluatorId {
    CurveEvaluatorId::new(name).unwrap()
}

fn pcurve_evaluator(name: &str) -> PcurveEvaluatorId {
    PcurveEvaluatorId::new(name).unwrap()
}

fn surface_evaluator(name: &str) -> SurfaceEvaluatorId {
    SurfaceEvaluatorId::new(name).unwrap()
}

fn trim_classifier(name: &str) -> TrimClassifierId {
    TrimClassifierId::new(name).unwrap()
}

fn mass_properties_evaluator(name: &str) -> MassPropertiesEvaluatorId {
    MassPropertiesEvaluatorId::new(name).unwrap()
}

pub(super) fn topology() -> ExactBRepTopology {
    let part_assembly = part_id(PersistentEntityKind::Assembly, "assembly-part");
    let root_assembly = id(PersistentEntityKind::Assembly, "assembly-root");
    let instance = part_id(PersistentEntityKind::Instance, "instance");
    let body = part_id(PersistentEntityKind::Body, "body");
    let lump = part_id(PersistentEntityKind::Lump, "lump");
    let solid = part_id(PersistentEntityKind::Solid, "solid");
    let shell = part_id(PersistentEntityKind::Shell, "shell");
    let face = part_id(PersistentEntityKind::Face, "face");
    let wire = part_id(PersistentEntityKind::Wire, "wire");
    let coedge = part_id(PersistentEntityKind::Coedge, "coedge");
    let edge = part_id(PersistentEntityKind::Edge, "edge");
    let vertex = part_id(PersistentEntityKind::Vertex, "vertex");
    ExactBRepTopology {
        schema_version: EXACT_BREP_TOPOLOGY_SCHEMA_VERSION,
        root_assembly_id: root_assembly.clone(),
        assemblies: vec![
            ExactAssembly {
                id: part_assembly.clone(),
                definition_digest: [1; 32],
                body_ids: vec![body.clone()],
                child_instance_ids: Vec::new(),
            },
            ExactAssembly {
                id: root_assembly.clone(),
                definition_digest: [2; 32],
                body_ids: Vec::new(),
                child_instance_ids: vec![instance.clone()],
            },
        ],
        instances: vec![ExactInstance {
            id: instance,
            parent_assembly_id: root_assembly,
            instantiated_assembly_id: part_assembly,
            transform: GeometryTransform([
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ]),
        }],
        bodies: vec![ExactBody {
            id: body,
            mass_properties_evaluator_id: mass_properties_evaluator("mass:1"),
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
            id: part_id(PersistentEntityKind::Region, "region-a"),
            solid_id: solid,
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
            surface_evaluator_id: surface_evaluator("surface:1"),
            trim_classifier_id: trim_classifier("trim:1"),
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
            pcurve_evaluator_id: pcurve_evaluator("pcurve:1"),
            seam_image: None,
        }],
        edges: vec![ExactEdge {
            id: edge,
            curve_evaluator_id: curve_evaluator("curve:1"),
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
    }
}

pub(super) fn model() -> ExactBRepModel {
    ExactBRepModel {
        artifact: GeometryObjectRef {
            digest: GeometryDigest::from_bytes([1; 32]),
            encoded_length: 4096,
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
        region_count: 1,
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

fn add_second_interface_side(topology: &mut ExactBRepTopology) {
    let body = part_id(PersistentEntityKind::Body, "body-b");
    let lump = part_id(PersistentEntityKind::Lump, "lump-b");
    let solid = part_id(PersistentEntityKind::Solid, "solid-b");
    let shell = part_id(PersistentEntityKind::Shell, "shell-b");
    topology.assemblies[0].body_ids.push(body.clone());
    topology.bodies.push(ExactBody {
        id: body,
        mass_properties_evaluator_id: mass_properties_evaluator("mass:2"),
        lump_ids: vec![lump.clone()],
        is_sheet_body: false,
        sheet_shell_ids: Vec::new(),
    });
    topology.lumps.push(ExactLump {
        id: lump,
        solid_ids: vec![solid.clone()],
    });
    topology.solids.push(ExactSolid {
        id: solid.clone(),
        outer_shell_id: shell.clone(),
        void_shell_ids: Vec::new(),
    });
    topology.regions.push(ExactRegion {
        id: part_id(PersistentEntityKind::Region, "region-b"),
        solid_id: solid,
    });
    topology.shells.push(ExactShell {
        id: shell,
        orientation: TopologicalOrientation::Forward,
        face_uses: vec![OrientedEntityUse {
            entity_id: topology.faces[0].id.clone(),
            orientation: TopologicalOrientation::Reversed,
        }],
    });
}

fn add_independent_contact_face(topology: &mut ExactBRepTopology) -> PersistentEntityId {
    let face = part_id(PersistentEntityKind::Face, "face-contact");
    let wire = part_id(PersistentEntityKind::Wire, "wire-contact");
    let coedge = part_id(PersistentEntityKind::Coedge, "coedge-contact");
    let edge = part_id(PersistentEntityKind::Edge, "edge-contact");
    let vertex = part_id(PersistentEntityKind::Vertex, "vertex-contact");
    topology.shells[0].face_uses.push(OrientedEntityUse {
        entity_id: face.clone(),
        orientation: TopologicalOrientation::Forward,
    });
    topology.faces.push(ExactFace {
        id: face.clone(),
        orientation: TopologicalOrientation::Forward,
        surface_evaluator_id: surface_evaluator("surface:contact"),
        trim_classifier_id: trim_classifier("trim:contact"),
        outer_wire_id: wire.clone(),
        inner_wire_ids: Vec::new(),
        periodic_u: false,
        periodic_v: false,
        has_singularity: false,
    });
    topology.wires.push(ExactWire {
        id: wire,
        orientation: TopologicalOrientation::Forward,
        coedge_ids: vec![coedge.clone()],
    });
    topology.coedges.push(ExactCoedge {
        id: coedge,
        face_id: face.clone(),
        edge_id: edge.clone(),
        orientation: TopologicalOrientation::Forward,
        pcurve_evaluator_id: pcurve_evaluator("pcurve:contact"),
        seam_image: None,
    });
    topology.edges.push(ExactEdge {
        id: edge,
        curve_evaluator_id: curve_evaluator("curve:contact"),
        start_vertex_id: Some(vertex.clone()),
        end_vertex_id: Some(vertex.clone()),
        is_closed: true,
        is_periodic: true,
        is_degenerate: false,
    });
    topology.vertices.push(ExactVertex {
        id: vertex,
        point_m: [1.0, 0.0, 0.0],
        tolerance_m: 1.0e-8,
    });
    face
}

#[test]
fn explicit_topology_validates_and_round_trips() {
    let topology = topology();
    topology.validate_against(&model()).unwrap();
    assert_eq!(
        serde_json::from_str::<ExactBRepTopology>(&serde_json::to_string(&topology).unwrap())
            .unwrap(),
        topology
    );
}

#[test]
fn assembly_occurrences_are_rooted_owned_and_affine() {
    let mut root_part = topology();
    root_part.root_assembly_id = root_part.assemblies[0].id.clone();
    root_part.assemblies.pop();
    root_part.instances.clear();
    let mut root_summary = model();
    root_summary.assembly_count = 1;
    root_summary.instance_count = 0;
    root_part.validate_against(&root_summary).unwrap();

    let mut invalid = topology();
    invalid.assemblies[0].definition_digest = [0; 32];
    assert!(invalid.validate_against(&model()).is_err());

    let mut cyclic = topology();
    cyclic.instances[0].instantiated_assembly_id = cyclic.root_assembly_id.clone();
    assert!(cyclic.validate_against(&model()).is_err());

    let mut non_affine = topology();
    non_affine.instances[0].transform.0[15] = 2.0;
    assert!(non_affine.validate_against(&model()).is_err());

    let mut singular = topology();
    singular.instances[0].transform.0[0] = 0.0;
    assert!(singular.validate_against(&model()).is_err());

    let mut orphaned_body = topology();
    orphaned_body.assemblies[0].body_ids.clear();
    assert!(orphaned_body.validate_against(&model()).is_err());
}

#[test]
fn sheet_bodies_own_shells_without_fake_solids() {
    let mut sheet = topology();
    sheet.bodies[0].is_sheet_body = true;
    sheet.bodies[0].lump_ids.clear();
    sheet.bodies[0].sheet_shell_ids = vec![sheet.shells[0].id.clone()];
    sheet.lumps.clear();
    sheet.solids.clear();
    sheet.regions.clear();
    let mut summary = model();
    summary.lump_count = 0;
    summary.solid_count = 0;
    summary.region_count = 0;
    sheet.validate_against(&summary).unwrap();

    sheet.bodies[0].lump_ids = vec![id(PersistentEntityKind::Lump, "missing")];
    assert!(sheet.validate_against(&summary).is_err());
}

#[test]
fn solid_shell_boundary_validation_rejects_open_boundaries_but_allows_sheets() {
    let mut open = topology();
    assert!(open.validate_solid_shell_boundaries().is_err());

    open.bodies[0].is_sheet_body = true;
    open.bodies[0].lump_ids.clear();
    open.bodies[0].sheet_shell_ids = vec![open.shells[0].id.clone()];
    open.lumps.clear();
    open.solids.clear();
    open.validate_solid_shell_boundaries().unwrap();
}

#[test]
fn dangling_kind_confused_and_wrong_face_incidence_fail() {
    let mut dangling = topology();
    dangling.coedges[0].edge_id = id(PersistentEntityKind::Edge, "missing");
    assert!(dangling.validate_against(&model()).is_err());
    let mut confused = topology();
    confused.faces[0].outer_wire_id = id(PersistentEntityKind::Edge, "edge");
    assert!(confused.validate_against(&model()).is_err());
    let mut wrong_face = topology();
    wrong_face.coedges[0].face_id = id(PersistentEntityKind::Face, "other");
    assert!(wrong_face.validate_against(&model()).is_err());
}

#[test]
fn shared_interfaces_require_opposite_oriented_region_uses() {
    let mut topology = topology();
    add_second_interface_side(&mut topology);
    topology.interfaces.push(ExactSharedInterface {
        face_id: topology.faces[0].id.clone(),
        side_a_region_id: topology.regions[0].id.clone(),
        side_b_region_id: topology.regions[1].id.clone(),
        side_a_orientation: TopologicalOrientation::Forward,
        side_b_orientation: TopologicalOrientation::Reversed,
    });
    let mut summary = model();
    summary.body_count = 2;
    summary.lump_count = 2;
    summary.solid_count = 2;
    summary.region_count = 2;
    summary.shell_count = 2;
    summary.interface_count = 1;
    topology.validate_against(&summary).unwrap();
    topology.interfaces[0].side_b_orientation = TopologicalOrientation::Forward;
    assert!(topology.validate_against(&summary).is_err());
    topology.interfaces[0].side_b_orientation = TopologicalOrientation::Reversed;
    topology.shells[1].face_uses[0].orientation = TopologicalOrientation::Forward;
    assert!(topology.validate_against(&summary).is_err());
}

#[test]
fn shared_interfaces_reject_unknown_regions() {
    let mut topology = topology();
    add_second_interface_side(&mut topology);
    topology.interfaces.push(ExactSharedInterface {
        face_id: topology.faces[0].id.clone(),
        side_a_region_id: topology.regions[0].id.clone(),
        side_b_region_id: part_id(PersistentEntityKind::Region, "unknown"),
        side_a_orientation: TopologicalOrientation::Forward,
        side_b_orientation: TopologicalOrientation::Reversed,
    });
    let mut summary = model();
    summary.body_count = 2;
    summary.lump_count = 2;
    summary.solid_count = 2;
    summary.region_count = 2;
    summary.shell_count = 2;
    summary.interface_count = 1;
    assert!(topology.validate_against(&summary).is_err());
}

#[test]
fn exact_regions_require_one_to_one_solid_ownership() {
    let mut topology = topology();
    add_second_interface_side(&mut topology);
    let mut summary = model();
    summary.body_count = 2;
    summary.lump_count = 2;
    summary.solid_count = 2;
    summary.region_count = 2;
    summary.shell_count = 2;

    topology.regions[1].solid_id = topology.regions[0].solid_id.clone();
    assert!(topology.validate_against(&summary).is_err());
}

#[test]
fn contacts_are_authored_from_canonical_exact_source_faces() {
    let mut topology = topology();
    let side_a_face = topology.faces[0].id.clone();
    let side_b_face = add_independent_contact_face(&mut topology);
    topology.contacts = author_exact_contacts(
        &topology,
        &[ExactContactDefinition {
            side_a_face_ids: vec![side_b_face.clone()],
            side_b_face_ids: vec![side_a_face.clone()],
        }],
    )
    .unwrap();
    let authored = topology.contacts[0].clone();
    let mut summary = model();
    summary.face_count = 2;
    summary.wire_count = 2;
    summary.coedge_count = 2;
    summary.edge_count = 2;
    summary.vertex_count = 2;
    summary.contact_count = 1;
    topology.validate_against(&summary).unwrap();

    let reversed = author_exact_contacts(
        &topology,
        &[ExactContactDefinition {
            side_a_face_ids: vec![side_a_face.clone()],
            side_b_face_ids: vec![side_b_face.clone()],
        }],
    )
    .unwrap();
    assert_eq!(reversed, vec![authored.clone()]);

    topology.contacts[0].side_b_face_ids = topology.contacts[0].side_a_face_ids.clone();
    assert!(topology.validate_against(&summary).is_err());
    topology.contacts[0] = authored.clone();
    topology.contacts[0].pairing_schema_version += 1;
    assert!(topology.validate_against(&summary).is_err());
    topology.contacts[0] = authored.clone();
    topology.contacts[0].pairing_contract_digest = [0; 32];
    assert!(topology.validate_against(&summary).is_err());
    topology.contacts[0] = authored;
    topology.contacts[0].id.source_topology_id = "contact:tampered".into();
    assert!(topology.validate_against(&summary).is_err());
}

#[test]
fn contact_authoring_rejects_unknown_interface_and_reused_faces() {
    let mut topology = topology();
    let first = topology.faces[0].id.clone();
    let second = add_independent_contact_face(&mut topology);
    let unknown = part_id(PersistentEntityKind::Face, "unknown");
    assert!(author_exact_contacts(
        &topology,
        &[ExactContactDefinition {
            side_a_face_ids: Vec::new(),
            side_b_face_ids: vec![first.clone()],
        }]
    )
    .is_err());
    assert!(author_exact_contacts(
        &topology,
        &[ExactContactDefinition {
            side_a_face_ids: vec![first.clone()],
            side_b_face_ids: vec![unknown],
        }]
    )
    .is_err());

    assert!(author_exact_contacts(
        &topology,
        &[
            ExactContactDefinition {
                side_a_face_ids: vec![first.clone()],
                side_b_face_ids: vec![second.clone()],
            },
            ExactContactDefinition {
                side_a_face_ids: vec![first.clone()],
                side_b_face_ids: vec![second.clone()],
            },
        ]
    )
    .is_err());

    add_second_interface_side(&mut topology);
    topology.interfaces.push(ExactSharedInterface {
        face_id: first.clone(),
        side_a_region_id: topology.regions[0].id.clone(),
        side_b_region_id: topology.regions[1].id.clone(),
        side_a_orientation: TopologicalOrientation::Forward,
        side_b_orientation: TopologicalOrientation::Reversed,
    });
    assert!(author_exact_contacts(
        &topology,
        &[ExactContactDefinition {
            side_a_face_ids: vec![first],
            side_b_face_ids: vec![second],
        }]
    )
    .is_err());
}

#[test]
fn noncanonical_order_and_summary_substitution_fail() {
    let mut noncanonical = topology();
    noncanonical.vertices.push(ExactVertex {
        id: id(PersistentEntityKind::Vertex, "a-before-existing"),
        point_m: [1.0, 0.0, 0.0],
        tolerance_m: 0.0,
    });
    let mut summary = model();
    summary.vertex_count = 2;
    assert!(noncanonical.validate_against(&summary).is_err());
    summary = model();
    summary.face_count = 2;
    assert!(topology().validate_against(&summary).is_err());
}

#[test]
fn occurrence_transform_resolves_world_points_and_vectors() {
    let mut topology = topology();
    topology.instances[0].transform = GeometryTransform([
        0.0, -2.0, 0.0, 3.0, 2.0, 0.0, 0.0, 4.0, 0.0, 0.0, 2.0, 5.0, 0.0, 0.0, 0.0, 1.0,
    ]);
    let transform = topology.world_transform_for(&topology.edges[0].id).unwrap();
    assert_eq!(
        transform.transform_point([1.0, 2.0, 3.0]),
        [-1.0, 6.0, 11.0]
    );
    assert_eq!(
        transform.transform_vector([1.0, 2.0, 3.0]),
        [-4.0, 2.0, 6.0]
    );

    let unknown = PersistentEntityId {
        kind: PersistentEntityKind::Edge,
        source_topology_id: "unknown".into(),
        assembly_path: vec!["absent".into()],
    };
    assert!(topology.world_transform_for(&unknown).is_err());
}

#[test]
fn affine_composition_applies_parent_before_local() {
    let parent = GeometryTransform([
        2.0, 0.0, 0.0, 10.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]);
    let local = GeometryTransform([
        1.0, 0.0, 0.0, 3.0, 0.0, 1.0, 0.0, 4.0, 0.0, 0.0, 1.0, 5.0, 0.0, 0.0, 0.0, 1.0,
    ]);
    assert_eq!(
        parent.compose(&local).transform_point([1.0, 1.0, 1.0]),
        [18.0, 10.0, 12.0]
    );
}
