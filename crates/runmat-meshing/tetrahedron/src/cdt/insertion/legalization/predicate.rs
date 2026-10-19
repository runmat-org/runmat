//! Sphere decisions distinguish canonical insertion from physical admission.
use super::*;

#[derive(Clone, Copy)]
pub(super) enum SphereRelationPolicy {
    CanonicalInsertion,
    PhysicalAdmission,
}

pub(super) fn repair_sphere_relation(
    topology: &DelaunayVolumeTopology,
    vertices: [u32; 4],
    node: crate::cdt::DelaunayVolumeNode,
    policy: SphereRelationPolicy,
    work: &mut InsertionWork<'_>,
) -> Result<PredicateSign, DelaunayInsertionError> {
    match policy {
        SphereRelationPolicy::CanonicalInsertion => {
            circumsphere_relation(topology, vertices, node, work)
        }
        SphereRelationPolicy::PhysicalAdmission => {
            work.predicate()?;
            let points = vertices.map(|vertex| topology.nodes[vertex as usize]);
            insphere_volume_nodes([points[0], points[1], points[2], points[3], node])
                .map_err(super::predicate_error)
        }
    }
}
