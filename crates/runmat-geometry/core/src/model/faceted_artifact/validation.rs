use super::{FacetedSolid, FACETED_SOLID_SCHEMA_VERSION};
use crate::model::validation_storage::{ValidationBuffer, ValidationWork};
use crate::{FacetedSolidModel, GeometryContractError, PersistentEntityId, PersistentEntityKind};
mod boundary;

pub(super) fn validate_faceted_solid(
    solid: &FacetedSolid,
    model: &FacetedSolidModel,
) -> Result<(), GeometryContractError> {
    validate_admitted(solid, model, ValidationWork(None))
}
pub(super) fn validate_admitted(
    solid: &FacetedSolid,
    model: &FacetedSolidModel,
    work: ValidationWork<'_>,
) -> Result<(), GeometryContractError> {
    work.checkpoint()?;
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
    validate_ids(
        solid.vertices.iter().map(|vertex| &vertex.id),
        PersistentEntityKind::Vertex,
        work,
    )?;
    for vertex in &solid.vertices {
        work.checkpoint()?;
        if vertex
            .coordinates_m
            .canonical_value()
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err(invalid("faceted vertex", "coordinates must be finite"));
        }
    }
    let shells = validate_shells(solid, work)?;
    validate_triangles(solid, &shells, work)?;
    boundary::validate(solid, &shells, work)
}
fn validate_ids<'a>(
    ids: impl ExactSizeIterator<Item = &'a PersistentEntityId>,
    kind: PersistentEntityKind,
    work: ValidationWork<'a>,
) -> Result<(), GeometryContractError> {
    let mut ordered = work.buffer(ids.len())?;
    for id in ids {
        work.checkpoint()?;
        id.validate()?;
        if id.kind != kind {
            return Err(invalid(
                "faceted entity identity",
                "identity kind must match its role and be unique",
            ));
        }
        ordered.push(id);
    }
    ordered.sort_unstable();
    if ordered.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(invalid(
            "faceted entity identity",
            "identity kind must match its role and be unique",
        ));
    }
    Ok(())
}
fn validate_shells<'a>(
    solid: &'a FacetedSolid,
    work: ValidationWork<'a>,
) -> Result<ValidationBuffer<'a, (&'a PersistentEntityId, usize)>, GeometryContractError> {
    validate_ids(
        solid.shells.iter().map(|shell| &shell.id),
        PersistentEntityKind::Shell,
        work,
    )?;
    let mut indices = work.buffer(solid.shells.len())?;
    let mut covered = work.buffer(solid.triangles.len())?;
    for _ in 0..solid.triangles.len() {
        covered.push(0u8);
    }
    for (index, shell) in solid.shells.iter().enumerate() {
        work.checkpoint()?;
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
            work.checkpoint()?;
            let covered = covered
                .get_mut(triangle_index as usize)
                .ok_or_else(|| invalid("faceted shell", "triangle index is outside the payload"))?;
            if std::mem::replace(covered, 1) != 0 {
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
        indices.push((&shell.id, index));
    }
    if covered.contains(&0) {
        return Err(invalid(
            "faceted shell",
            "every triangle must belong to a shell",
        ));
    }
    indices.sort_unstable_by_key(|entry| entry.0);
    Ok(indices)
}
fn validate_triangles(
    solid: &FacetedSolid,
    shells: &[(&PersistentEntityId, usize)],
    work: ValidationWork<'_>,
) -> Result<(), GeometryContractError> {
    validate_ids(
        solid.triangles.iter().map(|triangle| &triangle.id),
        PersistentEntityKind::Face,
        work,
    )?;
    let mut used_vertices = work.buffer(solid.vertices.len())?;
    for _ in 0..solid.vertices.len() {
        used_vertices.push(0u8);
    }
    for triangle in &solid.triangles {
        work.checkpoint()?;
        if shells
            .binary_search_by(|entry| entry.0.cmp(&triangle.shell_id))
            .is_err()
        {
            return Err(invalid(
                "faceted triangle",
                "shell identity is absent from the payload",
            ));
        }
        let [a, b, c] = triangle.vertex_indices;
        if a == b || b == c || c == a {
            return Err(invalid("faceted triangle", "vertices must be distinct"));
        }
        for slot in [a, b, c] {
            let used = used_vertices.get_mut(slot as usize).ok_or_else(|| {
                invalid("faceted triangle", "vertex index is outside the payload")
            })?;
            *used = 1;
        }
        let [a, b, c] = [a, b, c].map(|slot| {
            solid.vertices[slot as usize]
                .coordinates_m
                .canonical_value()
        });
        let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let cross = [
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        ];
        if cross.iter().any(|value| !value.is_finite()) || cross.iter().all(|&value| value == 0.0) {
            return Err(invalid("faceted triangle", "triangle is degenerate"));
        }
    }
    if used_vertices.contains(&0) {
        return Err(invalid(
            "faceted solid inventory",
            "every vertex must be referenced by the authoritative topology",
        ));
    }
    Ok(())
}
fn invalid(field: &str, reason: &str) -> GeometryContractError {
    GeometryContractError::invalid(field, reason)
}
