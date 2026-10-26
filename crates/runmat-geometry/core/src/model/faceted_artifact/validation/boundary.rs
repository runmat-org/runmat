use super::*;
use crate::FacetedShellOrientation;
#[derive(Clone, Copy)]
struct EdgeUse {
    key: (u32, u32),
    triangle: usize,
    from: u32,
    to: u32,
}
pub(super) fn validate(
    solid: &FacetedSolid,
    shells: &[(&PersistentEntityId, usize)],
    work: ValidationWork<'_>,
) -> Result<(), GeometryContractError> {
    let count = solid.triangles.len();
    let capacity = count
        .checked_mul(3)
        .ok_or_else(|| invalid("faceted solid boundary", "edge inventory overflow"))?;
    let mut edges = work.buffer(capacity)?;
    for (triangle, entry) in solid.triangles.iter().enumerate() {
        work.checkpoint()?;
        let [a, b, c] = entry.vertex_indices;
        for (from, to) in [(a, b), (b, c), (c, a)] {
            edges.push(EdgeUse {
                key: (from.min(to), from.max(to)),
                triangle,
                from,
                to,
            });
        }
    }
    edges.sort_unstable_by_key(|entry| (entry.key, entry.triangle, entry.from, entry.to));
    let mut neighbors = work.buffer(count)?;
    for _ in 0..count {
        neighbors.push([usize::MAX; 3]);
    }
    for pair in edges.chunks(2) {
        work.checkpoint()?;
        if pair.len() != 2
            || pair[0].key != pair[1].key
            || pair[0].from != pair[1].to
            || pair[0].to != pair[1].from
        {
            return Err(invalid(
                "faceted solid boundary",
                "every edge must have exactly two oppositely oriented triangle uses",
            ));
        }
        let left = pair[0].triangle;
        let right = pair[1].triangle;
        if solid.triangles[left].shell_id != solid.triangles[right].shell_id {
            return Err(invalid(
                "faceted solid boundary",
                "an edge cannot cross shell identities",
            ));
        }
        for (from, to) in [(left, right), (right, left)] {
            let slot = neighbors[from]
                .iter_mut()
                .find(|value| **value == usize::MAX)
                .ok_or_else(|| {
                    invalid("faceted solid boundary", "triangle has excess edge uses")
                })?;
            *slot = to;
        }
    }
    // Pairing alone must not let four or more uses of the same edge pass.
    if edges
        .windows(3)
        .any(|triple| triple[0].key == triple[2].key)
    {
        return Err(invalid(
            "faceted solid boundary",
            "every edge must have exactly two oppositely oriented triangle uses",
        ));
    }
    drop(edges);
    let mut visited = work.buffer(count)?;
    for _ in 0..count {
        visited.push(0u8);
    }
    let mut frontier = work.buffer(count)?;
    for &(_, shell_index) in shells {
        work.checkpoint()?;
        let expected = &solid.shells[shell_index].triangle_indices;
        let first = expected[0] as usize;
        visited[first] = 1;
        frontier.push(first);
        let mut reached = 0;
        while let Some(current) = frontier.pop() {
            work.checkpoint()?;
            reached += 1;
            for &next in &neighbors[current] {
                if next == usize::MAX {
                    return Err(invalid(
                        "faceted solid boundary",
                        "triangle adjacency is incomplete",
                    ));
                }
                if std::mem::replace(&mut visited[next], 1) == 0 {
                    frontier.push(next);
                }
            }
        }
        if reached != expected.len() {
            return Err(invalid(
                "faceted shell",
                "each shell identity must describe one connected component",
            ));
        }
        validate_shell_orientation(solid, shell_index)?;
    }
    Ok(())
}

fn validate_shell_orientation(
    solid: &FacetedSolid,
    shell_index: usize,
) -> Result<(), GeometryContractError> {
    let shell = &solid.shells[shell_index];
    let anchor_triangle = &solid.triangles[shell.triangle_indices[0] as usize];
    let origin = solid.vertices[anchor_triangle.vertex_indices[0] as usize]
        .coordinates_m
        .canonical_value();
    let mut signed_six_volume = 0.0;
    for &triangle_index in &shell.triangle_indices {
        let indices = solid.triangles[triangle_index as usize].vertex_indices;
        let [a, b, c] = indices.map(|index| {
            let point = solid.vertices[index as usize]
                .coordinates_m
                .canonical_value();
            [
                point[0] - origin[0],
                point[1] - origin[1],
                point[2] - origin[2],
            ]
        });
        let cross = [
            b[1] * c[2] - b[2] * c[1],
            b[2] * c[0] - b[0] * c[2],
            b[0] * c[1] - b[1] * c[0],
        ];
        signed_six_volume += a[0] * cross[0] + a[1] * cross[1] + a[2] * cross[2];
    }
    if !signed_six_volume.is_finite() || signed_six_volume == 0.0 {
        return Err(invalid(
            "faceted shell volume",
            "a closed shell must enclose finite nonzero oriented volume",
        ));
    }
    let measured = if signed_six_volume > 0.0 {
        FacetedShellOrientation::Outward
    } else {
        FacetedShellOrientation::Inward
    };
    if shell.orientation != measured {
        return Err(invalid(
            "faceted shell orientation",
            "declared orientation disagrees with the independent signed-volume measure",
        ));
    }
    Ok(())
}
