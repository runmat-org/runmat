use crate::report::{ImportDiagnostic, ImportDiagnosticSeverity};
use runmat_geometry_core::{EntityIdRange, EntityKind, Region, RegionEntityMapping, SurfaceMesh};
use std::collections::BTreeMap;

use super::{
    build_asset, build_result, capacity_guard, check_cancelled_periodic, is_degenerate_triangle,
    parse_f64, push_entity_range, push_mesh_count_diagnostics, push_utf8_bom_stripped_diagnostic,
    strip_utf8_bom_text, GeometryImportContext, GeometryImportError, GeometryImportOptions,
};

pub(super) fn import_obj(
    path: &str,
    bytes: &[u8],
    options: GeometryImportOptions,
    context: &GeometryImportContext,
) -> Result<crate::report::ImportResult, GeometryImportError> {
    context.check_cancelled()?;
    let text = std::str::from_utf8(bytes)
        .map_err(|_| GeometryImportError::ParseFailed("invalid UTF-8 OBJ payload".to_string()))?;

    let mut diagnostics = Vec::<ImportDiagnostic>::new();
    let (text, stripped_bom) = strip_utf8_bom_text(text);
    if stripped_bom {
        push_utf8_bom_stripped_diagnostic(&mut diagnostics, "obj");
    }
    let mut vertex_pool = Vec::<[f64; 3]>::new();
    let mut triangles = Vec::<[u32; 3]>::new();
    let mut triangle_count = 0u64;
    let mut regions = ObjRegionTracker::default();

    for (line_idx, line) in text.lines().enumerate() {
        check_cancelled_periodic(context, line_idx)?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if let Some(label) = trimmed.strip_prefix("o ") {
            regions.select("object", label);
            continue;
        }
        if let Some(label) = trimmed.strip_prefix("g ") {
            regions.select("group", label);
            continue;
        }
        if let Some(label) = trimmed.strip_prefix("usemtl ") {
            regions.select("material", label);
            continue;
        }
        if trimmed.starts_with("v ") {
            let mut parts = trimmed.split_whitespace();
            let _ = parts.next();
            let x = parse_f64(parts.next().ok_or_else(|| {
                GeometryImportError::ParseFailed(format!(
                    "missing OBJ vertex x component at line {}",
                    line_idx + 1
                ))
            })?)?;
            let y = parse_f64(parts.next().ok_or_else(|| {
                GeometryImportError::ParseFailed(format!(
                    "missing OBJ vertex y component at line {}",
                    line_idx + 1
                ))
            })?)?;
            let z = parse_f64(parts.next().ok_or_else(|| {
                GeometryImportError::ParseFailed(format!(
                    "missing OBJ vertex z component at line {}",
                    line_idx + 1
                ))
            })?)?;
            vertex_pool.push([x, y, z]);
            continue;
        }
        if !trimmed.starts_with("f ") {
            continue;
        }

        let mut face_indices = Vec::<usize>::new();
        for token in trimmed.split_whitespace().skip(1) {
            let index = parse_face_index(token, vertex_pool.len()).map_err(|reason| {
                GeometryImportError::ParseFailed(format!(
                    "invalid OBJ face index at line {}: {}",
                    line_idx + 1,
                    reason
                ))
            })?;
            face_indices.push(index);
        }

        if face_indices.len() < 3 {
            return Err(GeometryImportError::ParseFailed(format!(
                "OBJ face requires at least 3 vertices at line {}",
                line_idx + 1
            )));
        }

        let pivot = face_indices[0];
        for i in 1..(face_indices.len() - 1) {
            check_cancelled_periodic(context, i)?;
            let tri = [pivot, face_indices[i], face_indices[i + 1]];
            capacity_guard(triangle_count + 1, &options)?;

            let vertices = [
                vertex_pool[tri[0]],
                vertex_pool[tri[1]],
                vertex_pool[tri[2]],
            ];
            if is_degenerate_triangle(&vertices) {
                diagnostics.push(ImportDiagnostic {
                    code: "GEOMETRY_NORMALIZE_DEGENERATE_REMOVED".to_string(),
                    severity: ImportDiagnosticSeverity::Warning,
                    message: "Removed degenerate OBJ face triangle during import".to_string(),
                });
            } else {
                let entity_id = triangles.len() as u64;
                triangles.push([
                    u32::try_from(tri[0]).map_err(|_| {
                        GeometryImportError::ParseFailed(
                            "OBJ vertex index exceeds render mesh index range".to_string(),
                        )
                    })?,
                    u32::try_from(tri[1]).map_err(|_| {
                        GeometryImportError::ParseFailed(
                            "OBJ vertex index exceeds render mesh index range".to_string(),
                        )
                    })?,
                    u32::try_from(tri[2]).map_err(|_| {
                        GeometryImportError::ParseFailed(
                            "OBJ vertex index exceeds render mesh index range".to_string(),
                        )
                    })?,
                ]);
                regions.record_face(entity_id);
                triangle_count += 1;
            }
        }
    }

    push_mesh_count_diagnostics(
        &mut diagnostics,
        "obj",
        vertex_pool.len() as u64,
        triangle_count,
    );
    let mut asset = build_asset(
        path,
        "obj/v1",
        options.units,
        options.tessellation_profile.clone(),
        vertex_pool.len() as u64,
        triangle_count,
        vec![SurfaceMesh::new("mesh_1", vertex_pool, triangles)],
        diagnostics.clone(),
    );
    if let Some((mapped_regions, mapped_entities)) = regions.into_geometry_regions("mesh_1") {
        asset.regions = mapped_regions;
        asset.region_entity_mappings = mapped_entities;
    }
    Ok(build_result(asset, diagnostics))
}

#[derive(Debug, Default)]
struct ObjRegionTracker {
    selected_region_id: Option<String>,
    regions: Vec<Region>,
    region_ids_by_key: BTreeMap<String, String>,
    ranges_by_region_id: BTreeMap<String, Vec<EntityIdRange>>,
}

impl ObjRegionTracker {
    fn select(&mut self, kind: &str, raw_label: &str) {
        let label = raw_label.split_whitespace().collect::<Vec<_>>().join(" ");
        if label.is_empty() {
            self.selected_region_id = None;
            return;
        }
        let key = format!("{kind}:{label}");
        let region_id = if let Some(region_id) = self.region_ids_by_key.get(&key) {
            region_id.clone()
        } else {
            let mut candidate = format!("region_{}", stable_region_slug(&label));
            if self
                .regions
                .iter()
                .any(|region| region.region_id == candidate)
            {
                candidate = format!("{}_{}", candidate, self.regions.len() + 1);
            }
            self.region_ids_by_key.insert(key, candidate.clone());
            self.regions.push(Region {
                region_id: candidate.clone(),
                name: label,
                tag: Some(format!("obj_{kind}")),
                cad_ownership: None,
            });
            candidate
        };
        self.selected_region_id = Some(region_id);
    }

    fn record_face(&mut self, entity_id: u64) {
        let region_id = self
            .selected_region_id
            .clone()
            .unwrap_or_else(|| self.default_region_id());
        push_entity_range(
            self.ranges_by_region_id.entry(region_id).or_default(),
            entity_id,
        );
    }

    fn default_region_id(&mut self) -> String {
        const KEY: &str = "default:Default Region";
        if let Some(region_id) = self.region_ids_by_key.get(KEY) {
            return region_id.clone();
        }
        let region_id = "region_default".to_string();
        self.region_ids_by_key
            .insert(KEY.to_string(), region_id.clone());
        self.regions.push(Region {
            region_id: region_id.clone(),
            name: "Default Region".to_string(),
            tag: Some("mesh_default".to_string()),
            cad_ownership: None,
        });
        region_id
    }

    fn into_geometry_regions(
        self,
        mesh_id: &str,
    ) -> Option<(Vec<Region>, Vec<RegionEntityMapping>)> {
        if self.regions.is_empty() {
            return None;
        }
        let mappings = self
            .regions
            .iter()
            .filter_map(|region| {
                let ranges = self
                    .ranges_by_region_id
                    .get(&region.region_id)
                    .cloned()
                    .unwrap_or_default();
                if ranges.is_empty() {
                    None
                } else {
                    Some(RegionEntityMapping::new(
                        region.region_id.clone(),
                        mesh_id.to_string(),
                        EntityKind::Face,
                        ranges,
                    ))
                }
            })
            .collect::<Vec<_>>();
        Some((self.regions, mappings))
    }
}

fn stable_region_slug(value: &str) -> String {
    let mut slug = String::new();
    let mut previous_separator = false;
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() {
            slug.push(ch.to_ascii_lowercase());
            previous_separator = false;
        } else if !previous_separator && !slug.is_empty() {
            slug.push('_');
            previous_separator = true;
        }
    }
    while slug.ends_with('_') {
        slug.pop();
    }
    if slug.is_empty() {
        "unnamed".to_string()
    } else {
        slug
    }
}

fn parse_face_index(token: &str, vertex_count: usize) -> Result<usize, String> {
    if vertex_count == 0 {
        return Err("face declared before vertices".to_string());
    }
    let raw_index = token
        .split('/')
        .next()
        .ok_or_else(|| "missing face index token".to_string())?;
    let index = raw_index
        .parse::<isize>()
        .map_err(|_| format!("invalid face index '{}'", raw_index))?;
    if index == 0 {
        return Err("OBJ indices are 1-based and cannot be zero".to_string());
    }

    let resolved = if index > 0 {
        index - 1
    } else {
        vertex_count as isize + index
    };
    if resolved < 0 || resolved >= vertex_count as isize {
        return Err(format!(
            "face index '{}' resolves out of bounds for {} vertices",
            raw_index, vertex_count
        ));
    }
    Ok(resolved as usize)
}

#[cfg(test)]
mod tests {
    use super::{parse_face_index, stable_region_slug};

    #[test]
    fn parse_face_index_supports_positive_and_negative_indices() {
        assert_eq!(parse_face_index("1/2/3", 4).expect("positive index"), 0);
        assert_eq!(parse_face_index("-1", 4).expect("negative index"), 3);
    }

    #[test]
    fn stable_region_slug_normalizes_labels() {
        assert_eq!(stable_region_slug("Bracket A"), "bracket_a");
        assert_eq!(stable_region_slug("  @@@  "), "unnamed");
    }
}
