//! Read-only, request-local geometry observations for the study agent.
//! Never mutates the user's interactive scene or treats tessellation as exact CAD.
use crate::core::PipelineType;
use crate::geometry_scene::{GeometryScenePresentation, GeometrySceneSection};
use crate::{GeometryScene, GeometrySceneChunk};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashSet;

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct Visibility {
    #[serde(default)]
    mode: Option<String>,
    #[serde(default)]
    hidden_ids: Vec<String>,
    #[serde(default)]
    isolated_ids: Vec<String>,
}

fn known_ids(scene: &GeometryScene) -> HashSet<&str> {
    scene
        .chunks
        .iter()
        .flat_map(|chunk| {
            std::iter::once(chunk.chunk_id.as_str())
                .chain(chunk.mesh_id.as_deref())
                .chain(chunk.owner_node_ids.iter().map(String::as_str))
                .chain(chunk.regions.iter().map(|region| region.region_id.as_str()))
        })
        .collect()
}

fn validate_ids(scene: &GeometryScene, ids: &[String]) -> Result<(), String> {
    if ids.len() > 64 {
        return Err("At most 64 geometry IDs may be inspected at once".into());
    }
    let known = known_ids(scene);
    for id in ids {
        if !known.contains(id.as_str()) {
            return Err(format!("Unknown geometry ID '{id}'; query current geometry IDs before selecting or isolating it"));
        }
    }
    Ok(())
}

fn matching_triangles(chunk: &GeometrySceneChunk, ids: &[String], count: usize) -> Vec<bool> {
    if ids.iter().any(|id| {
        id == &chunk.chunk_id
            || chunk.mesh_id.as_ref() == Some(id)
            || chunk.owner_node_ids.contains(id)
    }) {
        return vec![true; count];
    }
    let mut mask = vec![false; count];
    for region in chunk
        .regions
        .iter()
        .filter(|region| ids.contains(&region.region_id))
    {
        for range in &region.triangle_ranges {
            let start = (range.start as usize).min(count);
            let end = start.saturating_add(range.count as usize).min(count);
            mask[start..end].fill(true);
        }
    }
    mask
}

pub fn observation_scene(
    scene: &GeometryScene,
    visibility: Option<Value>,
    section: Option<Value>,
    selected: Vec<String>,
) -> Result<GeometryScene, String> {
    let visibility: Visibility = visibility
        .map(serde_json::from_value)
        .transpose()
        .map_err(|error| format!("Invalid geometry visibility: {error}"))?
        .unwrap_or_default();
    if !matches!(
        visibility.mode.as_deref(),
        None | Some("all" | "hidden" | "isolated")
    ) {
        return Err("Unsupported geometry visibility mode".into());
    }
    if visibility.mode.as_deref() == Some("isolated") && visibility.isolated_ids.is_empty() {
        return Err("Isolation requires at least one geometry ID".into());
    }
    validate_ids(scene, &selected)?;
    validate_ids(scene, &visibility.hidden_ids)?;
    validate_ids(scene, &visibility.isolated_ids)?;
    let section: Option<GeometrySceneSection> = section
        .map(serde_json::from_value)
        .transpose()
        .map_err(|error| format!("Invalid section plane: {error}"))?;
    if let Some(section) = &section {
        let plane = &section.plane;
        let norm: f32 = plane.normal.iter().map(|v| v * v).sum();
        if !norm.is_finite()
            || norm <= 1e-12
            || plane
                .origin
                .is_some_and(|origin| origin.iter().any(|v| !v.is_finite()))
            || plane.offset.is_some_and(|v| !v.is_finite())
        {
            return Err(
                "Section plane must have a finite, nonzero normal and finite coordinates".into(),
            );
        }
    }
    if selected.is_empty()
        && visibility.hidden_ids.is_empty()
        && visibility.mode.as_deref() != Some("isolated")
        && section.is_none()
    {
        return Ok(scene.clone());
    }
    let palette = [
        [0.98, 0.45, 0.12, 1.0],
        [0.1, 0.8, 0.95, 1.0],
        [0.2, 0.85, 0.4, 1.0],
        [0.9, 0.25, 0.8, 1.0],
        [1.0, 0.85, 0.1, 1.0],
        [0.3, 0.45, 1.0, 1.0],
    ];
    let colors = GeometryScenePresentation {
        selected_region_ids: selected.clone(),
        ..Default::default()
    };
    let clipping = GeometryScenePresentation {
        section: Some(section),
        ..Default::default()
    };
    let mut callouts = selected
        .iter()
        .enumerate()
        .map(|(index, id)| crate::geometry_scene::GeometrySceneCallout {
            region_id: id.clone(),
            color: palette[index % palette.len()],
            triangles: Vec::new(),
        })
        .collect::<Vec<_>>();
    let mut chunks = Vec::new();
    for chunk in &scene.chunks {
        if !chunk.visible || chunk.render_data.pipeline_type != PipelineType::Triangles {
            continue;
        }
        let count = chunk.triangle_count();
        let hidden = matching_triangles(chunk, &visibility.hidden_ids, count);
        let isolated = matching_triangles(chunk, &visibility.isolated_ids, count);
        let selected_masks = selected
            .iter()
            .map(|id| matching_triangles(chunk, std::slice::from_ref(id), count))
            .collect::<Vec<_>>();
        let data = chunk.render_data_with_presentation(&colors);
        let mut vertices = Vec::new();
        for triangle in 0..count {
            if hidden[triangle]
                || (visibility.mode.as_deref() == Some("isolated") && !isolated[triangle])
            {
                continue;
            }
            for corner in 0..3 {
                let at = triangle * 3 + corner;
                let index = match &data.indices {
                    Some(indices) => *indices
                        .get(at)
                        .ok_or("Geometry contains an invalid triangle index")?
                        as usize,
                    None => at,
                };
                let mut vertex = *data
                    .vertices
                    .get(index)
                    .ok_or("Geometry contains an invalid triangle index")?;
                if let Some(selected_index) = selected_masks.iter().position(|mask| mask[triangle])
                {
                    vertex.color = palette[selected_index % palette.len()];
                }
                vertices.push(vertex);
            }
            for (index, mask) in selected_masks.iter().enumerate() {
                if mask[triangle] {
                    let at = vertices.len() - 3;
                    callouts[index].triangles.push([
                        vertices[at].position,
                        vertices[at + 1].position,
                        vertices[at + 2].position,
                    ]);
                }
            }
        }
        if vertices.is_empty() {
            continue;
        }
        let indices = (0..vertices.len())
            .map(|i| u32::try_from(i).map_err(|_| "Geometry observation is too large"))
            .collect::<Result<Vec<_>, _>>()?;
        let filtered = GeometrySceneChunk::indexed_triangles(
            chunk.chunk_id.clone(),
            vertices,
            indices,
            chunk.material.clone(),
        );
        let clipped = filtered.render_data_with_presentation(&clipping);
        if clipped.vertices.is_empty() || clipped.draw_calls.is_empty() {
            continue;
        }
        let mut baked = GeometrySceneChunk::from_render_data(chunk.chunk_id.clone(), clipped);
        // Keep camera fitting tied to the actual observed vertices.
        let mut min = baked.vertices[0].position;
        let mut max = min;
        for vertex in &baked.vertices {
            for axis in 0..3 {
                min[axis] = min[axis].min(vertex.position[axis]);
                max[axis] = max[axis].max(vertex.position[axis]);
            }
        }
        baked.bounds = crate::core::BoundingBox {
            min: min.into(),
            max: max.into(),
        };
        baked.render_data.bounds = Some(baked.bounds);
        // Tessellation edges are not CAD feature edges. Keep these observations
        // shaded so internal triangulation does not obscure face highlights.
        chunks.push(baked);
    }
    if chunks.is_empty() {
        return Err("No surface triangles remain for the requested visibility and section; change the view or inspected IDs".into());
    }
    // Different selections can have identical vertex counts. Their cache keys
    // still need to differ or a retained renderer can reuse the previous colors.
    use std::hash::{Hash, Hasher};
    let mut identity = std::collections::hash_map::DefaultHasher::new();
    selected.hash(&mut identity);
    visibility.mode.hash(&mut identity);
    visibility.hidden_ids.hash(&mut identity);
    visibility.isolated_ids.hash(&mut identity);
    serde_json::to_string(&clipping.section)
        .map_err(|error| error.to_string())?
        .hash(&mut identity);
    let mut observed = GeometryScene::new(
        format!(
            "{}:agent-observation:{:016x}",
            scene.scene_id,
            identity.finish()
        ),
        scene.revision,
        chunks,
    );
    observed.title = scene.title.clone();
    observed.show_grid = scene.show_grid;
    observed.axis_equal = scene.axis_equal;
    observed.observation_callouts = callouts;
    observed.physical_context = scene.physical_context.clone();
    Ok(observed)
}

pub fn measure_scene(scene: &GeometryScene, kind: &str, ids: &[String]) -> Result<Value, String> {
    if !matches!(kind, "area" | "bounds") {
        return Err(format!("Measurement '{kind}' is unavailable; supported tessellation measurements are area and bounds"));
    }
    use runmat_types::physical::{
        Area, Position, QuantityKind, SourceQuantity, Translation, UnitAttribution, UnitDefinition,
    };
    let context=scene.physical_context.as_ref().ok_or("Geometry scene has no preserved physical unit context; reopen the geometry to inspect its source units")?;
    context
        .source_unit_evidence
        .validate()
        .map_err(|e| e.to_string())?;
    if !context.source_unit_evidence.physical_scale_resolved() {
        return Err("Geometry source units are unresolved; specify the physical interpretation before measuring".into());
    }
    let units = context.coordinate_units;
    let length_unit = units.unit();
    let scale = units.meters_per_unit();
    validate_ids(scene, ids)?;
    let mut area = 0.0f64;
    let mut weighted_center = [0.0; 3];
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    let mut triangles = 0usize;
    for chunk in &scene.chunks {
        if !chunk.visible || chunk.render_data.pipeline_type != PipelineType::Triangles {
            continue;
        }
        let mask = matching_triangles(chunk, ids, chunk.triangle_count());
        for (triangle, matched) in mask.iter().enumerate() {
            if !ids.is_empty() && !matched {
                continue;
            }
            let mut p = [[0.0; 3]; 3];
            for (corner, point) in p.iter_mut().enumerate() {
                let at = triangle * 3 + corner;
                let index = match &chunk.indices {
                    Some(indices) => *indices
                        .get(at)
                        .ok_or("Geometry contains an invalid triangle index")?
                        as usize,
                    None => at,
                };
                let vertex = chunk
                    .vertices
                    .get(index)
                    .ok_or("Geometry contains an invalid triangle index")?;
                for axis in 0..3 {
                    point[axis] = f64::from(vertex.position[axis]);
                    min[axis] = min[axis].min(point[axis]);
                    max[axis] = max[axis].max(point[axis]);
                }
            }
            let u = [p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]];
            let v = [p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]];
            let cross = [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ];
            let a = cross.iter().map(|x| x * x).sum::<f64>().sqrt() * 0.5;
            area += a;
            triangles += 1;
            for (axis, component) in weighted_center.iter_mut().enumerate() {
                *component += a * (p[0][axis] + p[1][axis] + p[2][axis]) / 3.0;
            }
        }
    }
    if triangles == 0 || !area.is_finite() || area <= 0.0 {
        return Err("The requested geometry has no measurable surface triangles".into());
    }
    let center = weighted_center.map(|v| v / area);
    let canonical_bounds = json!({"min":Position::new(min.map(|v|v*scale)),"max":Position::new(max.map(|v|v*scale)),"size":Translation::new(std::array::from_fn::<_,3,_>(|i|(max[i]-min[i])*scale))});
    let area_source = SourceQuantity::try_new(
        QuantityKind::Area,
        area,
        UnitDefinition::try_new(QuantityKind::Area.dimension(), scale * scale, 0.0)
            .map_err(|e| e.to_string())?,
        format!("{}^2", length_unit.symbol()),
        UnitAttribution::CanonicalComputation,
    )
    .map_err(|e| e.to_string())?;
    Ok(
        json!({"kind":kind,"status":"measured","method":"surface_tessellation","approximate":true,"region_ids":ids,"triangle_count":triangles,
        "value":if kind=="area"{json!({"canonical":Area::new(area*scale*scale),"source":area_source})}else{canonical_bounds.clone()},
        "bounds":canonical_bounds,"coordinateUnits":units,"centroid":Position::new(center.map(|v|v*scale)),
        "note":"Measured from imported surface tessellation, not exact CAD. Canonical values are SI; the source measure retains the preview coordinate conversion. Not mesh-convergence or engineering-acceptance evidence."}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry_scene::{GeometrySceneRegion, GeometrySceneTriangleRange};
    fn fixture() -> GeometryScene {
        let positions = [
            [0., 0., 0.],
            [2., 0., 0.],
            [0., 2., 0.],
            [10., 0., 0.],
            [12., 0., 0.],
            [10., 2., 0.],
        ];
        let vertices = positions
            .into_iter()
            .map(|p| crate::geometry_scene_vertex(p, [0.5, 0.5, 0.5, 1.], [0., 0., 1.]))
            .collect();
        let chunk = GeometrySceneChunk::indexed_triangles(
            "part",
            vertices,
            vec![0, 1, 2, 3, 4, 5],
            Default::default(),
        )
        .with_regions(vec![
            GeometrySceneRegion::new(
                "face_a",
                None,
                None,
                vec![GeometrySceneTriangleRange::new(0, 1)],
            ),
            GeometrySceneRegion::new(
                "face_b",
                None,
                None,
                vec![GeometrySceneTriangleRange::new(1, 1)],
            ),
        ]);
        GeometryScene::new("fixture", 1, vec![chunk]).with_physical_context(
            crate::geometry_scene::GeometryPhysicalContext {
                coordinate_units: runmat_types::physical::CoordinateUnits::new(
                    runmat_types::physical::LengthUnit::Millimeter,
                ),
                source_unit_evidence: runmat_types::physical::LengthUnitEvidence::explicit_fallback(
                    runmat_types::physical::LengthUnit::Millimeter,
                    "authored geometry fixture",
                ),
            },
        )
    }
    #[test]
    fn isolation_changes_geometry_and_selection_colors_without_mutating_source() {
        let scene = fixture();
        let observed = observation_scene(
            &scene,
            Some(json!({"mode":"isolated","isolated_ids":["face_b"]})),
            None,
            vec!["face_b".into()],
        )
        .unwrap();
        let triangles = observed
            .chunks
            .iter()
            .filter(|chunk| chunk.render_data.pipeline_type == PipelineType::Triangles)
            .collect::<Vec<_>>();
        assert_eq!(triangles.len(), 1);
        assert!(observed
            .chunks
            .iter()
            .all(|chunk| chunk.render_data.pipeline_type == PipelineType::Triangles));
        assert_eq!(triangles[0].triangle_count(), 1);
        assert!(observed.bounds.min.x >= 10.0);
        assert!(triangles[0]
            .vertices
            .iter()
            .all(|v| v.color == [0.98, 0.45, 0.12, 1.0]));
        assert_eq!(scene.chunks[0].triangle_count(), 2);
        assert!(observation_scene(
            &scene,
            Some(json!({"mode":"isolated","isolated_ids":["missing"]})),
            None,
            vec![]
        )
        .is_err());
    }
    #[test]
    fn sectioning_clips_the_returned_surface_instead_of_ignoring_the_plane() {
        let observed = observation_scene(
            &fixture(),
            Some(json!({"mode":"isolated","isolated_ids":["face_b"]})),
            Some(json!({"plane":{"normal":[1,0,0],"origin":[11,0,0]}})),
            vec![],
        )
        .unwrap();
        let measurement = measure_scene(&observed, "area", &[]).unwrap();
        let area = measurement["value"]["canonical"]["square_meters"]
            .as_f64()
            .unwrap();
        assert!(area > 0.0 && area < 2.0e-6);
        assert!(observation_scene(
            &fixture(),
            None,
            Some(json!({"plane":{"normal":[0,0,0]}})),
            vec![]
        )
        .is_err());
    }
    #[test]
    fn measurements_report_real_values_units_and_tessellation_limits() {
        let result = measure_scene(&fixture(), "area", &["face_a".into()]).unwrap();
        assert_eq!(result["value"]["canonical"]["square_meters"], 2.0e-6);
        assert_eq!(result["value"]["source"]["source_symbol"], "mm^2");
        assert_eq!(result["approximate"], true);
        assert_eq!(result["bounds"]["max"]["meters"], json!([0.002, 0.002, 0.]));
        assert!(measure_scene(&fixture(), "area", &["missing".into()]).is_err());
        assert!(measure_scene(&fixture(), "distance", &[]).is_err());
    }
}
