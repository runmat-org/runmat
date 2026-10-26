use super::work::HullWork;
use super::{error, DelaunayPointSetError, DelaunayPointSetErrorKind, DelaunayVolumeNode};
use runmat_meshing_core::quality::predicate::PredicateSign;

pub(super) fn initial_tetrahedron(
    nodes: &[DelaunayVolumeNode],
    work: &mut HullWork<'_>,
) -> Result<[usize; 4], DelaunayPointSetError> {
    let first = nodes[0].coordinates_m.canonical_value();
    let second = nodes[1].coordinates_m.canonical_value();
    let mut third = None;
    for (index, node) in nodes.iter().enumerate().skip(2) {
        let point = node.coordinates_m.canonical_value();
        let projections = [
            [
                [first[0], first[1]],
                [second[0], second[1]],
                [point[0], point[1]],
            ],
            [
                [first[1], first[2]],
                [second[1], second[2]],
                [point[1], point[2]],
            ],
            [
                [first[2], first[0]],
                [second[2], second[0]],
                [point[2], point[0]],
            ],
        ];
        let mut noncollinear = false;
        for projection in projections {
            noncollinear |= work.orient2d(projection)? != PredicateSign::Zero;
        }
        if noncollinear {
            third = Some(index);
            break;
        }
    }
    let third = third.ok_or_else(|| {
        error(
            DelaunayPointSetErrorKind::InsufficientDimension,
            "all point-set nodes are collinear",
        )
    })?;
    for index in (third + 1)..nodes.len() {
        if work.orient3d([
            first,
            second,
            nodes[third].coordinates_m.canonical_value(),
            nodes[index].coordinates_m.canonical_value(),
        ])? != PredicateSign::Zero
        {
            return Ok([0, 1, third, index]);
        }
    }
    Err(error(
        DelaunayPointSetErrorKind::InsufficientDimension,
        "all point-set nodes are coplanar",
    ))
}
