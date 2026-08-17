use std::collections::{BTreeMap, BTreeSet, VecDeque};

use runmat_geometry_core::{
    FacetedShell, FacetedShellOrientation, FacetedSolid, FacetedTriangle, FacetedVertex,
    PersistentEntityId, PersistentEntityKind, SurfaceMesh, FACETED_SOLID_SCHEMA_VERSION,
};

use crate::import::{GeometryImportContext, GeometryImportError};

type CoordinateKey = [u64; 3];

pub(super) fn canonical_faceted_solid(
    meshes: &[SurfaceMesh],
    meters_per_source_unit: f64,
    context: &GeometryImportContext,
) -> Result<FacetedSolid, GeometryImportError> {
    if meshes.is_empty() {
        return Err(invalid("faceted source has no triangle payload"));
    }
    let mut coordinates = BTreeMap::<CoordinateKey, [f64; 3]>::new();
    let mut source_triangles = Vec::<[CoordinateKey; 3]>::new();
    for mesh in meshes {
        for (index, triangle) in mesh.triangles.iter().enumerate() {
            checkpoint(context, index)?;
            let points = triangle.map(|vertex_index| {
                mesh.vertices
                    .get(vertex_index as usize)
                    .copied()
                    .map(|point| point.map(|value| normalize_zero(value * meters_per_source_unit)))
            });
            let [Some(a), Some(b), Some(c)] = points else {
                return Err(invalid(
                    "faceted triangle references an unknown source vertex",
                ));
            };
            for point in [a, b, c] {
                if point.iter().any(|value| !value.is_finite()) {
                    return Err(invalid(
                        "faceted source contains a non-finite normalized vertex",
                    ));
                }
                coordinates.entry(key(point)).or_insert(point);
            }
            let mut triangle = [key(a), key(b), key(c)];
            rotate_minimum_first(&mut triangle);
            source_triangles.push(triangle);
        }
    }
    let ordered_coordinates = coordinates.into_iter().collect::<Vec<_>>();
    let vertex_index = ordered_coordinates
        .iter()
        .enumerate()
        .map(|(index, (key, _))| (*key, index as u32))
        .collect::<BTreeMap<_, _>>();
    let mut triangles = source_triangles
        .into_iter()
        .map(|triangle| triangle.map(|point| vertex_index[&point]))
        .collect::<Vec<_>>();
    triangles.sort();
    let shell_triangles = shell_components(&triangles, context)?;
    let shell_ids = (0..shell_triangles.len())
        .map(|index| entity_id(PersistentEntityKind::Shell, "shell", index))
        .collect::<Vec<_>>();
    let mut triangle_shells = vec![0usize; triangles.len()];
    for (shell_index, members) in shell_triangles.iter().enumerate() {
        for &triangle_index in members {
            triangle_shells[triangle_index as usize] = shell_index;
        }
    }
    let vertices = ordered_coordinates
        .into_iter()
        .enumerate()
        .map(|(index, (_, coordinates_m))| FacetedVertex {
            id: entity_id(PersistentEntityKind::Vertex, "vertex", index),
            coordinates_m,
        })
        .collect::<Vec<_>>();
    let triangle_records = triangles
        .iter()
        .enumerate()
        .map(|(index, &vertex_indices)| FacetedTriangle {
            id: entity_id(PersistentEntityKind::Face, "face", index),
            vertex_indices,
            shell_id: shell_ids[triangle_shells[index]].clone(),
        })
        .collect::<Vec<_>>();
    let shells = shell_triangles
        .into_iter()
        .enumerate()
        .map(|(index, triangle_indices)| {
            Ok(FacetedShell {
                id: shell_ids[index].clone(),
                orientation: shell_orientation(&vertices, &triangles, &triangle_indices)?,
                triangle_indices,
            })
        })
        .collect::<Result<Vec<_>, GeometryImportError>>()?;
    Ok(FacetedSolid {
        schema_version: FACETED_SOLID_SCHEMA_VERSION,
        vertices,
        triangles: triangle_records,
        shells,
    })
}

fn shell_components(
    triangles: &[[u32; 3]],
    context: &GeometryImportContext,
) -> Result<Vec<Vec<u32>>, GeometryImportError> {
    let mut edges = BTreeMap::<(u32, u32), Vec<(usize, u32, u32)>>::new();
    for (triangle_index, &[a, b, c]) in triangles.iter().enumerate() {
        checkpoint(context, triangle_index)?;
        if a == b || b == c || c == a {
            return Err(invalid("faceted source contains a collapsed triangle"));
        }
        for (from, to) in [(a, b), (b, c), (c, a)] {
            edges
                .entry((from.min(to), from.max(to)))
                .or_default()
                .push((triangle_index, from, to));
        }
    }
    let mut neighbors = vec![BTreeSet::new(); triangles.len()];
    for (edge, uses) in edges {
        if uses.len() != 2 {
            return Err(invalid(format!(
                "faceted source edge {}:{} has {} uses instead of two",
                edge.0,
                edge.1,
                uses.len()
            )));
        }
        if uses[0].1 != uses[1].2 || uses[0].2 != uses[1].1 {
            return Err(invalid(format!(
                "faceted source edge {}:{} does not have opposite oriented uses",
                edge.0, edge.1
            )));
        }
        neighbors[uses[0].0].insert(uses[1].0);
        neighbors[uses[1].0].insert(uses[0].0);
    }
    let mut unseen = (0..triangles.len()).collect::<BTreeSet<_>>();
    let mut components = Vec::new();
    while let Some(&seed) = unseen.first() {
        let mut members = Vec::new();
        let mut queue = VecDeque::from([seed]);
        while let Some(current) = queue.pop_front() {
            if !unseen.remove(&current) {
                continue;
            }
            members.push(current as u32);
            queue.extend(neighbors[current].iter().copied());
        }
        members.sort_unstable();
        components.push(members);
    }
    Ok(components)
}

fn shell_orientation(
    vertices: &[FacetedVertex],
    triangles: &[[u32; 3]],
    members: &[u32],
) -> Result<FacetedShellOrientation, GeometryImportError> {
    let origin = vertices[triangles[members[0] as usize][0] as usize].coordinates_m;
    let mut signed_six_volume = 0.0_f64;
    for &triangle_index in members {
        let [a, b, c] = triangles[triangle_index as usize].map(|vertex_index| {
            let point = vertices[vertex_index as usize].coordinates_m;
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
            "faceted shell does not enclose finite nonzero volume",
        ));
    }
    Ok(if signed_six_volume > 0.0 {
        FacetedShellOrientation::Outward
    } else {
        FacetedShellOrientation::Inward
    })
}

fn entity_id(kind: PersistentEntityKind, role: &str, index: usize) -> PersistentEntityId {
    PersistentEntityId {
        kind,
        source_topology_id: format!("faceted:{role}:{index:016x}"),
        assembly_path: vec!["root".into()],
    }
}

fn rotate_minimum_first(triangle: &mut [CoordinateKey; 3]) {
    let minimum = triangle
        .iter()
        .enumerate()
        .min_by_key(|(_, point)| *point)
        .map(|(index, _)| index)
        .unwrap_or(0);
    triangle.rotate_left(minimum);
}

fn key(point: [f64; 3]) -> CoordinateKey {
    point.map(f64::to_bits)
}

fn normalize_zero(value: f64) -> f64 {
    if value == 0.0 {
        0.0
    } else {
        value
    }
}

fn checkpoint(context: &GeometryImportContext, index: usize) -> Result<(), GeometryImportError> {
    if index & 0x3ff == 0 {
        context.check_cancelled()?;
    }
    Ok(())
}

fn invalid(reason: impl Into<String>) -> GeometryImportError {
    GeometryImportError::InvalidGeometry(reason.into())
}
