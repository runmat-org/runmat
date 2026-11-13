//! Exact sphere-preserving reconnections. Every new simplex has the same
//! physical circumsphere as an old simplex: no empty-sphere admission can
//! change, including faces outside the cavity. Geometry coverage and protected
//! interfaces remain the caller's responsibility; mutation checks incidence.
use super::topology_access::TetrahedralTopology;
use runmat_meshing_core::quality::predicate::{PredicateSign, SpatialPredicateError};

fn shares_physical_sphere(
    topology: &impl TetrahedralTopology,
    reference: [u32; 4],
    fifth: u32,
) -> Result<bool, SpatialPredicateError> {
    topology
        .insphere_nodes([
            topology.nodes()[reference[0] as usize],
            topology.nodes()[reference[1] as usize],
            topology.nodes()[reference[2] as usize],
            topology.nodes()[reference[3] as usize],
            topology.nodes()[fifth as usize],
        ])
        .map(|sign| sign == PredicateSign::Zero)
}

/// A convex two-to-three retriangulation retains the entire six-face cavity
/// boundary. Equal physical spheres preserve interior and exterior admission.
pub(in crate::cdt) fn admits_interior_diagonal(
    topology: &impl TetrahedralTopology,
    reference: [u32; 4],
    endpoints: [u32; 2],
    face: [u32; 3],
) -> Result<bool, SpatialPredicateError> {
    Ok(interior_diagonal_geometry(topology, endpoints, face)?
        && shares_physical_sphere(topology, reference, endpoints[1])?)
}

pub(in crate::cdt) fn interior_diagonal_geometry(
    topology: &impl TetrahedralTopology,
    endpoints: [u32; 2],
    face: [u32; 3],
) -> Result<bool, SpatialPredicateError> {
    let [left, right] = endpoints;
    let mut sign = None;
    for edge in [[face[0], face[1]], [face[1], face[2]], [face[2], face[0]]] {
        let current = orient(topology, [left, right, edge[0], edge[1]])?;
        if current == PredicateSign::Zero || sign.is_some_and(|previous| previous != current) {
            return Ok(false);
        }
        sign = Some(current);
    }
    Ok(true)
}

/// Both diagonals must lie inside the same convex planar quadrilateral. This
/// proves that changing its hull triangulation preserves physical coverage.
pub(in crate::cdt) fn admits_boundary_diagonal(
    topology: &impl TetrahedralTopology,
    reference: [u32; 4],
    endpoints: [u32; 2],
    old_diagonal: [u32; 2],
    apex: u32,
) -> Result<bool, SpatialPredicateError> {
    Ok(
        boundary_diagonal_geometry(topology, endpoints, old_diagonal, apex)?
            && shares_physical_sphere(topology, reference, endpoints[1])?,
    )
}

pub(in crate::cdt) fn boundary_diagonal_geometry(
    topology: &impl TetrahedralTopology,
    endpoints: [u32; 2],
    old_diagonal: [u32; 2],
    apex: u32,
) -> Result<bool, SpatialPredicateError> {
    let [left, right] = endpoints;
    let [first, second] = old_diagonal;
    Ok(
        orient(topology, [left, right, first, second])? == PredicateSign::Zero
            && opposite_sides(topology, [first, second, apex], endpoints)?
            && opposite_sides(topology, [left, right, apex], old_diagonal)?,
    )
}

fn opposite_sides(
    topology: &impl TetrahedralTopology,
    face: [u32; 3],
    endpoints: [u32; 2],
) -> Result<bool, SpatialPredicateError> {
    let left = orient(topology, [face[0], face[1], face[2], endpoints[0]])?;
    let right = orient(topology, [face[0], face[1], face[2], endpoints[1]])?;
    Ok(left != PredicateSign::Zero && right != PredicateSign::Zero && left != right)
}
fn orient(
    topology: &impl TetrahedralTopology,
    vertices: [u32; 4],
) -> Result<PredicateSign, SpatialPredicateError> {
    topology.orient_nodes(vertices.map(|vertex| topology.nodes()[vertex as usize]))
}
