use std::collections::{BTreeMap, BTreeSet, VecDeque};

use super::{FacetedShellOrientation, FacetedSolid, FACETED_SOLID_SCHEMA_VERSION};
use crate::{FacetedSolidModel, GeometryContractError, PersistentEntityId, PersistentEntityKind};

pub(super) fn validate_faceted_solid(
    solid: &FacetedSolid,
    model: &FacetedSolidModel,
) -> Result<(), GeometryContractError> {
    if solid.schema_version != FACETED_SOLID_SCHEMA_VERSION {
        return Err(invalid("faceted solid schema", "unsupported version"));
    }
    if !model.is_watertight || !model.is_oriented {
        return Err(invalid(
            "faceted solid model",
            "the document must declare the proven topology properties",
        ));
    }
    if solid.vertices.len() as u64 != model.vertex_count
        || solid.triangles.len() as u64 != model.triangle_count
        || solid.shells.len() as u64 != model.shell_count
        || solid.vertices.len() < 4
        || solid.triangles.len() < 4
        || solid.shells.is_empty()
    {
        return Err(invalid(
            "faceted solid inventory",
            "payload counts must match a non-empty document model",
        ));
    }
    validate_vertices(solid)?;
    let shell_indices = validate_shells(solid)?;
    validate_triangles(solid, &shell_indices)?;
    validate_closed_oriented_shells(solid, &shell_indices)
}

fn validate_vertices(solid: &FacetedSolid) -> Result<(), GeometryContractError> {
    let mut ids = BTreeSet::new();
    for vertex in &solid.vertices {
        validate_id(&vertex.id, PersistentEntityKind::Vertex, &mut ids)?;
        if vertex.coordinates_m.iter().any(|value| !value.is_finite()) {
            return Err(invalid("faceted vertex", "coordinates must be finite"));
        }
    }
    Ok(())
}

fn validate_shells(
    solid: &FacetedSolid,
) -> Result<BTreeMap<PersistentEntityId, usize>, GeometryContractError> {
    let mut ids = BTreeSet::new();
    let mut indices = BTreeMap::new();
    let mut covered = vec![false; solid.triangles.len()];
    for (shell_index, shell) in solid.shells.iter().enumerate() {
        validate_id(&shell.id, PersistentEntityKind::Shell, &mut ids)?;
        if shell.triangle_indices.is_empty()
            || shell
                .triangle_indices
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
        {
            return Err(invalid(
                "faceted shell",
                "triangle indices must be non-empty, sorted, and unique",
            ));
        }
        for &triangle_index in &shell.triangle_indices {
            let covered = covered
                .get_mut(triangle_index as usize)
                .ok_or_else(|| invalid("faceted shell", "triangle index is outside the payload"))?;
            if std::mem::replace(covered, true) {
                return Err(invalid(
                    "faceted shell",
                    "each triangle must belong to exactly one shell",
                ));
            }
            if solid.triangles[triangle_index as usize].shell_id != shell.id {
                return Err(invalid(
                    "faceted shell",
                    "triangle membership and shell identity disagree",
                ));
            }
        }
        indices.insert(shell.id.clone(), shell_index);
    }
    if covered.into_iter().any(|value| !value) {
        return Err(invalid(
            "faceted shell",
            "every triangle must belong to a shell",
        ));
    }
    Ok(indices)
}

fn validate_triangles(
    solid: &FacetedSolid,
    shell_indices: &BTreeMap<PersistentEntityId, usize>,
) -> Result<(), GeometryContractError> {
    let mut ids = BTreeSet::new();
    let mut used_vertices = vec![false; solid.vertices.len()];
    for triangle in &solid.triangles {
        validate_id(&triangle.id, PersistentEntityKind::Face, &mut ids)?;
        if !shell_indices.contains_key(&triangle.shell_id) {
            return Err(invalid(
                "faceted triangle",
                "shell identity is absent from the payload",
            ));
        }
        let [a, b, c] = triangle.vertex_indices;
        if a == b || b == c || c == a {
            return Err(invalid("faceted triangle", "vertices must be distinct"));
        }
        for index in [a, b, c] {
            let Some(used) = used_vertices.get_mut(index as usize) else {
                return Err(invalid(
                    "faceted triangle",
                    "vertex index is outside the payload",
                ));
            };
            *used = true;
        }
        let points = [a, b, c].map(|index| {
            solid
                .vertices
                .get(index as usize)
                .map(|vertex| vertex.coordinates_m)
        });
        let [Some(a), Some(b), Some(c)] = points else {
            return Err(invalid(
                "faceted triangle",
                "vertex index is outside the payload",
            ));
        };
        let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let cross = [
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        ];
        if cross.iter().any(|value| !value.is_finite()) || cross.iter().all(|value| *value == 0.0) {
            return Err(invalid("faceted triangle", "triangle is degenerate"));
        }
    }
    if used_vertices.into_iter().any(|used| !used) {
        return Err(invalid(
            "faceted solid inventory",
            "every vertex must be referenced by the authoritative topology",
        ));
    }
    Ok(())
}

fn validate_closed_oriented_shells(
    solid: &FacetedSolid,
    shell_indices: &BTreeMap<PersistentEntityId, usize>,
) -> Result<(), GeometryContractError> {
    let mut edges = BTreeMap::<(u32, u32), Vec<(usize, u32, u32)>>::new();
    let mut neighbors = vec![BTreeSet::new(); solid.triangles.len()];
    for (triangle_index, triangle) in solid.triangles.iter().enumerate() {
        let [a, b, c] = triangle.vertex_indices;
        for (from, to) in [(a, b), (b, c), (c, a)] {
            edges
                .entry((from.min(to), from.max(to)))
                .or_default()
                .push((triangle_index, from, to));
        }
    }
    for uses in edges.values() {
        if uses.len() != 2 || uses[0].1 != uses[1].2 || uses[0].2 != uses[1].1 {
            return Err(invalid(
                "faceted solid boundary",
                "every edge must have exactly two oppositely oriented triangle uses",
            ));
        }
        let left = uses[0].0;
        let right = uses[1].0;
        if solid.triangles[left].shell_id != solid.triangles[right].shell_id {
            return Err(invalid(
                "faceted solid boundary",
                "an edge cannot cross shell identities",
            ));
        }
        neighbors[left].insert(right);
        neighbors[right].insert(left);
    }
    for (shell_id, &shell_index) in shell_indices {
        let expected = &solid.shells[shell_index].triangle_indices;
        let mut visited = BTreeSet::new();
        let mut queue = VecDeque::from([expected[0] as usize]);
        while let Some(current) = queue.pop_front() {
            if !visited.insert(current) {
                continue;
            }
            for &neighbor in &neighbors[current] {
                if solid.triangles[neighbor].shell_id == *shell_id {
                    queue.push_back(neighbor);
                }
            }
        }
        if visited.len() != expected.len() {
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
    let origin = solid.vertices[anchor_triangle.vertex_indices[0] as usize].coordinates_m;
    let mut signed_six_volume = 0.0;
    for &triangle_index in &shell.triangle_indices {
        let indices = solid.triangles[triangle_index as usize].vertex_indices;
        let [a, b, c] = indices.map(|index| {
            let point = solid.vertices[index as usize].coordinates_m;
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

fn validate_id(
    id: &PersistentEntityId,
    expected_kind: PersistentEntityKind,
    ids: &mut BTreeSet<PersistentEntityId>,
) -> Result<(), GeometryContractError> {
    id.validate()?;
    if id.kind != expected_kind || !ids.insert(id.clone()) {
        return Err(invalid(
            "faceted entity identity",
            "identity kind must match its role and be unique",
        ));
    }
    Ok(())
}

fn invalid(field: &str, reason: &str) -> GeometryContractError {
    GeometryContractError::invalid(field, reason)
}
