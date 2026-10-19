use std::collections::BTreeMap;

use runmat_meshing_core::{predicate::orient2d, PredicateSign};

use crate::{ExactFaceDelaunayTriangle, ExactFacePslg};

#[derive(Clone, Copy, Debug)]
pub(super) struct EdgeUse {
    pub triangle_index: usize,
    pub opposite_vertex: u32,
}

#[cfg(test)]
pub(super) fn edge_uses(
    triangles: &[ExactFaceDelaunayTriangle],
) -> BTreeMap<[u32; 2], Vec<EdgeUse>> {
    edge_uses_with_checkpoint(triangles, || Ok::<(), std::convert::Infallible>(()))
        .unwrap_or_else(|never| match never {})
}

/// Derive the same canonical incidence while admitting cancellation during the walk.
pub(super) fn edge_uses_with_checkpoint<E>(
    triangles: &[ExactFaceDelaunayTriangle],
    mut checkpoint: impl FnMut() -> Result<(), E>,
) -> Result<BTreeMap<[u32; 2], Vec<EdgeUse>>, E> {
    let mut uses = BTreeMap::<[u32; 2], Vec<EdgeUse>>::new();
    for (triangle_index, triangle) in triangles.iter().enumerate() {
        checkpoint()?;
        let indices = triangle.vertex_indices;
        for position in 0..3 {
            uses.entry(sorted_edge([
                indices[position],
                indices[(position + 1) % 3],
            ]))
            .or_default()
            .push(EdgeUse {
                triangle_index,
                opposite_vertex: indices[(position + 2) % 3],
            });
        }
    }
    Ok(uses)
}

pub(super) fn flip_edge(
    triangles: &mut [ExactFaceDelaunayTriangle],
    pslg: &ExactFacePslg,
    edge: [u32; 2],
    uses: &[EdgeUse],
    existing_edges: &BTreeMap<[u32; 2], Vec<EdgeUse>>,
) -> Result<Option<[u32; 2]>, runmat_meshing_core::PlanarPredicateError> {
    if uses.len() != 2 {
        return Ok(None);
    }
    let diagonal = sorted_edge([uses[0].opposite_vertex, uses[1].opposite_vertex]);
    if diagonal[0] == diagonal[1] || existing_edges.contains_key(&diagonal) {
        return Ok(None);
    }
    let Some(left) = oriented_triangle([diagonal[0], diagonal[1], edge[0]], pslg)? else {
        return Ok(None);
    };
    let Some(right) = oriented_triangle([diagonal[1], diagonal[0], edge[1]], pslg)? else {
        return Ok(None);
    };
    triangles[uses[0].triangle_index] = ExactFaceDelaunayTriangle {
        vertex_indices: canonical_triangle(left),
    };
    triangles[uses[1].triangle_index] = ExactFaceDelaunayTriangle {
        vertex_indices: canonical_triangle(right),
    };
    triangles.sort();
    Ok(Some(diagonal))
}

pub(super) fn properly_crosses(
    left: [u32; 2],
    right: [u32; 2],
    pslg: &ExactFacePslg,
) -> Result<bool, runmat_meshing_core::PlanarPredicateError> {
    if left.iter().any(|vertex| right.contains(vertex)) {
        return Ok(false);
    }
    let left_uv = left.map(|index| pslg.vertices[index as usize].uv);
    let right_uv = right.map(|index| pslg.vertices[index as usize].uv);
    let signs = [
        orient2d([left_uv[0], left_uv[1], right_uv[0]])?,
        orient2d([left_uv[0], left_uv[1], right_uv[1]])?,
        orient2d([right_uv[0], right_uv[1], left_uv[0]])?,
        orient2d([right_uv[0], right_uv[1], left_uv[1]])?,
    ];
    Ok(opposite(signs[0], signs[1]) && opposite(signs[2], signs[3]))
}

pub(super) fn sorted_edge(mut edge: [u32; 2]) -> [u32; 2] {
    edge.sort_unstable();
    edge
}

pub(super) fn oriented_triangle(
    mut triangle: [u32; 3],
    pslg: &ExactFacePslg,
) -> Result<Option<[u32; 3]>, runmat_meshing_core::PlanarPredicateError> {
    match orient2d(triangle.map(|index| pslg.vertices[index as usize].uv))? {
        PredicateSign::Positive => Ok(Some(triangle)),
        PredicateSign::Negative => {
            triangle.swap(0, 1);
            Ok(Some(triangle))
        }
        PredicateSign::Zero => Ok(None),
    }
}

pub(super) fn canonical_triangle(triangle: [u32; 3]) -> [u32; 3] {
    let position = triangle
        .iter()
        .enumerate()
        .min_by_key(|(_, index)| *index)
        .map(|(position, _)| position)
        .expect("triangle has three vertices");
    [
        triangle[position],
        triangle[(position + 1) % 3],
        triangle[(position + 2) % 3],
    ]
}

fn opposite(left: PredicateSign, right: PredicateSign) -> bool {
    matches!(
        (left, right),
        (PredicateSign::Negative, PredicateSign::Positive)
            | (PredicateSign::Positive, PredicateSign::Negative)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cooperative_incidence_preserves_shared_edges_and_stops_inside_the_walk() {
        let triangles = [
            ExactFaceDelaunayTriangle {
                vertex_indices: [0, 1, 2],
            },
            ExactFaceDelaunayTriangle {
                vertex_indices: [1, 3, 2],
            },
            ExactFaceDelaunayTriangle {
                vertex_indices: [2, 3, 4],
            },
        ];
        let expected = edge_uses(&triangles);
        let mut probes = 0;
        let actual = edge_uses_with_checkpoint(&triangles, || {
            probes += 1;
            Ok::<(), &'static str>(())
        })
        .unwrap();
        assert_eq!(probes, triangles.len());
        assert_eq!(
            actual.keys().collect::<Vec<_>>(),
            expected.keys().collect::<Vec<_>>()
        );
        for (edge, uses) in actual {
            let facts = |uses: &[EdgeUse]| {
                uses.iter()
                    .map(|use_| (use_.triangle_index, use_.opposite_vertex))
                    .collect::<Vec<_>>()
            };
            assert_eq!(facts(&uses), facts(&expected[&edge]));
        }
        let mut probes = 0;
        let error = edge_uses_with_checkpoint(&triangles, || {
            probes += 1;
            if probes == 2 {
                Err("cancelled")
            } else {
                Ok(())
            }
        })
        .unwrap_err();
        assert_eq!(error, "cancelled");
        assert_eq!(probes, 2);
    }
}
