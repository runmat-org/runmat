use super::*;

#[test]
fn face_incidence_reaches_exact_boundary_region_and_occurrence_ancestry() {
    let (_, topology, _) = runmat_geometry_fixtures::exact_circle();
    let incidence = TopologyMetricIncidence::new(&topology);
    let entities = incidence.incident_face_entities(&topology.faces[0]);

    for entity in topology
        .faces
        .iter()
        .map(|value| &value.id)
        .chain(topology.wires.iter().map(|value| &value.id))
        .chain(topology.coedges.iter().map(|value| &value.id))
        .chain(topology.edges.iter().map(|value| &value.id))
        .chain(topology.vertices.iter().map(|value| &value.id))
        .chain(topology.shells.iter().map(|value| &value.id))
        .chain(topology.solids.iter().map(|value| &value.id))
        .chain(topology.regions.iter().map(|value| &value.id))
        .chain(topology.lumps.iter().map(|value| &value.id))
        .chain(topology.bodies.iter().map(|value| &value.id))
        .chain(topology.assemblies.iter().map(|value| &value.id))
        .chain(topology.instances.iter().map(|value| &value.id))
    {
        assert!(
            entities.contains(entity),
            "missing incident entity {entity:?}"
        );
        assert!(incidence.knows(entity));
    }
}

#[test]
fn faces_sharing_an_exact_edge_are_symmetric_metric_neighbors() {
    let (_, mut topology, _) = runmat_geometry_fixtures::exact_circle();
    let first_face_id = topology.faces[0].id.clone();
    let mut second_face = topology.faces[0].clone();
    second_face.id.source_topology_id = "face:2".into();
    let mut second_coedge = topology.coedges[0].clone();
    second_coedge.id.source_topology_id = "coedge:2".into();
    second_coedge.face_id = second_face.id.clone();
    topology.faces.push(second_face.clone());
    topology.faces.sort_by(|left, right| left.id.cmp(&right.id));
    topology.coedges.push(second_coedge);
    topology
        .coedges
        .sort_by(|left, right| left.id.cmp(&right.id));

    let adjacency = TopologyMetricIncidence::face_adjacency(&topology);
    assert!(adjacency[&first_face_id].contains(&second_face.id));
    assert!(adjacency[&second_face.id].contains(&first_face_id));
}
