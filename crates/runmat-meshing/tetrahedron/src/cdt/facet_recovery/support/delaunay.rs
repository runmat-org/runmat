//! The exact 2D Delaunay triangulation of a facet's boundary polygon.
//!
//! The polygon is the source triangle with the recovered segment nodes on its edges. Corners are
//! explicit and segment nodes are constructed on the source edges, so every polygon node lies
//! exactly in the source triangle's plane. A constrained Delaunay tetrahedralization restricts to
//! each facet as the facet's 2D Delaunay triangulation; a support with another diagonal needs
//! Steiner points that only a non-Delaunay choice made necessary. Each in-circle decision lifts the
//! circle to a sphere through an explicit point off the plane and evaluates the constructed
//! in-sphere predicate exactly, so the decision agrees with the volume predicates.
use super::*;
use runmat_meshing_core::quality::predicate::{
    insphere3d_constructed, orient3d_constructed, Point3Construction,
};

pub(super) struct FacetProjection {
    pub(super) axes: [usize; 2],
    pub(super) sign: PredicateSign,
}

/// Returns `None` when a cocircular tie or a degenerate chord leaves the triangulation
/// undetermined; the caller then keeps its topology-guided ear clipping.
pub(super) fn triangulate_delaunay(
    recovery: &FacetSupportView<'_, impl TetrahedralTopology>,
    boundary: &[StableDigest],
    source: [StableDigest; 3],
    projection: FacetProjection,
    source_edge_membership: &BTreeMap<StableDigest, BTreeSet<usize>>,
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<Option<Vec<DelaunayRecoveredFacetTriangle>>, DelaunayFacetRecoveryError> {
    let FacetProjection {
        axes,
        sign: polygon_sign,
    } = projection;
    let count = boundary.len();
    if count < 4 {
        return Ok(None);
    }
    let Some(apex) = lifted_apex(recovery, source, constraint_index)? else {
        return Ok(None);
    };
    let node = |index: usize| -> Result<SpatialPredicatePoint, DelaunayFacetRecoveryError> {
        node_index(recovery.topology, boundary[index])
            .map(|position| recovery.topology.nodes()[position].predicate_point())
            .ok_or_else(|| {
                invalid(
                    constraint_index,
                    "facet support node is missing from topology",
                )
            })
    };
    // A chord between two nodes of one source edge would run along that edge over the nodes
    // between them; only adjacent boundary nodes may share a source edge.
    let chord_admissible = |first: usize, last: usize| {
        last == first + 1
            || source_edge_membership
                .get(&boundary[first])
                .zip(source_edge_membership.get(&boundary[last]))
                .is_none_or(|(first, last)| first.intersection(last).next().is_none())
    };
    let mut triangles = Vec::with_capacity(count - 2);
    // The closing boundary edge (count - 1, 0) is the base of the whole polygon.
    let mut pending = vec![(0_usize, count - 1)];
    while let Some((first, last)) = pending.pop() {
        if last == first + 1 {
            continue;
        }
        let mut apex_index = None;
        for middle in first + 1..last {
            work.support_step(constraint_index)?;
            let triangle = [boundary[first], boundary[middle], boundary[last]];
            if !chord_admissible(first, middle)
                || !chord_admissible(middle, last)
                || shares_source_edge(triangle, source_edge_membership)
                || orientation(recovery, triangle, axes, constraint_index, work)? != polygon_sign
            {
                continue;
            }
            let Some(current) = apex_index else {
                apex_index = Some(middle);
                continue;
            };
            // Circles through the base chord are nested on the polygon's side, so replacing
            // the apex by any node strictly inside its circle ends at the empty circle.
            match in_circle(
                [node(first)?, node(current)?, node(last)?],
                apex,
                node(middle)?,
                constraint_index,
            )? {
                PredicateSign::Positive => apex_index = Some(middle),
                PredicateSign::Zero => return Ok(None),
                PredicateSign::Negative => {}
            }
        }
        let Some(middle) = apex_index else {
            return Ok(None);
        };
        for other in (first + 1..last).filter(|other| *other != middle) {
            work.support_step(constraint_index)?;
            if in_circle(
                [node(first)?, node(middle)?, node(last)?],
                apex,
                node(other)?,
                constraint_index,
            )? != PredicateSign::Negative
            {
                return Ok(None);
            }
        }
        triangles.push(DelaunayRecoveredFacetTriangle {
            node_identities: [boundary[first], boundary[middle], boundary[last]],
        });
        pending.push((middle, last));
        pending.push((first, middle));
    }
    Ok(Some(triangles))
}

/// An explicit point off the source triangle's plane, at the triangle's scale along its normal.
fn lifted_apex(
    recovery: &FacetSupportView<'_, impl TetrahedralTopology>,
    source: [StableDigest; 3],
    constraint_index: u32,
) -> Result<Option<SpatialPredicatePoint>, DelaunayFacetRecoveryError> {
    let [a, b, c] = points(recovery, source, constraint_index)?;
    let sub = |p: [f64; 3], q: [f64; 3]| [p[0] - q[0], p[1] - q[1], p[2] - q[2]];
    let dot = |p: [f64; 3], q: [f64; 3]| p[0] * q[0] + p[1] * q[1] + p[2] * q[2];
    let (ab, ac) = (
        sub(b.coordinates, a.coordinates),
        sub(c.coordinates, a.coordinates),
    );
    let normal = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ];
    let length = dot(normal, normal).sqrt();
    let scale = dot(ab, ab).max(dot(ac, ac)).sqrt();
    if !(length > 0.0 && length.is_finite() && scale > 0.0) {
        return Ok(None);
    }
    let coordinates = [0, 1, 2].map(|axis| a.coordinates[axis] + normal[axis] / length * scale);
    let apex = SpatialPredicatePoint {
        identity: StableDigest::from_bytes([0xa5; 32]),
        coordinates,
        construction: Point3Construction::Explicit,
    };
    // Rounding can only move the apex along the plane by far less than its height.
    match orient3d_constructed([a, b, c, apex]) {
        Ok(PredicateSign::Zero) | Err(_) => Ok(None),
        Ok(_) => Ok(Some(apex)),
    }
}

/// The sign of `query` against the circumcircle of the coplanar `triangle`: positive strictly
/// inside, zero on it, negative outside.
fn in_circle(
    triangle: [SpatialPredicatePoint; 3],
    apex: SpatialPredicatePoint,
    query: SpatialPredicatePoint,
    constraint_index: u32,
) -> Result<PredicateSign, DelaunayFacetRecoveryError> {
    let failure = |failure| {
        error(
            DelaunayFacetRecoveryErrorKind::InvalidTopology,
            Some(constraint_index),
            format!("facet support in-circle predicate failed: {failure:?}"),
        )
    };
    let [a, b, c] = triangle;
    let orientation = orient3d_constructed([a, b, c, apex]).map_err(failure)?;
    let sphere = insphere3d_constructed([a, b, c, apex, query]).map_err(failure)?;
    Ok(match (orientation, sphere) {
        (PredicateSign::Zero, _) | (_, PredicateSign::Zero) => PredicateSign::Zero,
        (orientation, sphere) if orientation == sphere => PredicateSign::Positive,
        _ => PredicateSign::Negative,
    })
}
