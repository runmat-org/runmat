//! `evaluate.zebra`: black and white bands over every face, banded by the angle between the
//! exact surface normal and a light direction, so the bands run on across an edge where the
//! faces are tangent and break where they meet at a crease. Each edge between two faces is a row
//! with its continuity, found by the kernel: G0 (the faces meet), G1 (tangent) or G2 (curvature
//! continuous).
//!
//! Values: `stripes` (bands over the half-turn from facing the light to facing away),
//! `direction` (a plane, axis or face: the light; the Top plane's normal when not picked).

use super::field::{field_face, field_meshes, measure, model_size, SurfaceField, RAMP_STRIPES};
use super::{Analysis, Row};
use runmat_cad::evaluate::{AnalysisRequest, AnalysisResult, EdgeContinuity, FieldQuantity};
use serde_json::{json, Map, Value};
use std::f64::consts::PI;

/// Zebra is drawn with triangles no longer than this share of the model's size, and short
/// enough that a triangle's corner normals turn by at most a third of a band.
const EDGE_SHARE: f64 = 1.0 / 40.0;

/// The band (0 black, 1 white) of a unit normal for `stripes` bands against the light `light`.
pub(super) fn band(normal: [f64; 3], light: [f64; 3], stripes: u32) -> u32 {
    let c = (normal[0] * light[0] + normal[1] * light[1] + normal[2] * light[2]).clamp(-1.0, 1.0);
    let s = f64::from(stripes) * c.acos() / PI;
    (s.floor() as u32).min(stripes.saturating_sub(1)) % 2
}

fn continuity_text(c: EdgeContinuity) -> &'static str {
    match c {
        EdgeContinuity::G0 => "G0 · the faces meet at a crease",
        EdgeContinuity::G1 => "G1 · tangent",
        EdgeContinuity::G2 => "G2 · curvature continuous",
    }
}

pub(super) fn zebra(
    path: &str,
    text: &str,
    configuration: Option<&str>,
    values: &Map<String, Value>,
) -> Result<Analysis, String> {
    let part = crate::runtime::regenerate_part_in(path, text, configuration, &Default::default())
        .map_err(|d| d.first().map(|d| d.message.clone()).unwrap_or_default())?;
    let stripes = values
        .get("stripes")
        .and_then(Value::as_u64)
        .unwrap_or(20)
        .clamp(2, 200) as u32;
    let pick = crate::features::picks_of(values, "direction")
        .into_iter()
        .next()
        .unwrap_or_else(|| "origin.xy".into());
    let light = crate::draft_analysis::pull_direction(&part.display, &pick)?;
    let bodies: Vec<_> = part
        .display
        .bodies
        .iter()
        .filter(|b| !b.flat_pattern)
        .collect();
    if bodies.is_empty() {
        return Err("the part has no bodies to show".into());
    }
    let turn = PI / f64::from(stripes) / 3.0;
    let meshes = field_meshes(&bodies, model_size(&part.display) * EDGE_SHARE, Some(turn));
    let points = measure(&part, &meshes, FieldQuantity::Normal)?;
    let faces: Vec<_> = meshes
        .iter()
        .zip(&points)
        .filter_map(|(m, p)| {
            field_face(m, p, |q| {
                q.normal.map(|n| f64::from(band(n, light, stripes)))
            })
        })
        .collect();
    if faces.is_empty() {
        return Err("no face of the part could be measured".into());
    }
    // The continuity of each edge between two faces, per body.
    let mut rows = Vec::new();
    let mut counts = [0usize; 3];
    let mut found = Vec::new();
    for body in &bodies {
        let Some(state) = part.result.state.bodies.get(body.index as usize) else {
            continue;
        };
        let edges = match part
            .kernel
            .analyze(&AnalysisRequest::Zebra, &[&state.handle])
        {
            Ok(AnalysisResult::Zebra { edges }) => edges,
            Ok(other) => return Err(format!("the kernel answered {other:?}")),
            Err(e) => return Err(format!("the kernel could not check the edges: {e:?}")),
        };
        for (index, c) in edges {
            let Some(edge) = body.edges.iter().find(|e| e.index == index) else {
                continue;
            };
            // Edges on one face (a seam) aren't between faces.
            if edge.faces[0] == edge.faces[1] {
                continue;
            }
            counts[c as usize] += 1;
            let face_label = |i: u32| {
                body.faces
                    .iter()
                    .find(|f| f.index == i)
                    .map(|f| crate::face_labels::label(body.index, f))
                    .unwrap_or_else(|| format!("Face {}", i + 1))
            };
            found.push(json!({ "edge": edge.reference, "continuity": c }));
            rows.push(Row {
                cells: vec![
                    format!("Edge {}", index + 1),
                    format!(
                        "{} · {}",
                        face_label(edge.faces[0]),
                        face_label(edge.faces[1])
                    ),
                    continuity_text(c).into(),
                ],
                items: edge.reference.iter().cloned().collect(),
                severity: None,
                legend: None,
                reference: edge.reference.clone(),
            });
        }
    }
    // Creases first: they are where the bands break.
    rows.sort_by_key(|r| !r.cells[2].starts_with("G0"));
    let summary = format!(
        "{} edges between faces · {} G0 (crease) · {} G1 (tangent) · {} G2 (curvature continuous)",
        rows.len(),
        counts[0],
        counts[1],
        counts[2]
    );
    Ok(Analysis {
        title: "Zebra stripes".into(),
        summary,
        columns: ["Edge", "Between", "Continuity"].map(String::from).to_vec(),
        rows,
        problems: Vec::new(),
        data: json!({ "stripes": stripes, "light": light, "edges": found }),
        colors: None,
        settings: vec![
            format!("Stripes: {stripes}"),
            format!("Light: {}", super::pull_label(&pick, light, None)),
        ],
        arrow: None,
        dimensions: Vec::new(),
        field: Some(SurfaceField {
            kind: "stripes",
            label: "Zebra stripes".into(),
            unit: String::new(),
            range: [0.0, 1.0],
            ramp: RAMP_STRIPES.to_vec(),
            faces,
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::band;

    #[test]
    fn bands_alternate_with_the_normals_angle_to_the_light() {
        let up = [0.0, 0.0, 1.0];
        assert_eq!(band(up, up, 10), 0);
        // 18° to 36° is the second band of ten.
        let tilted = [(25f64).to_radians().sin(), 0.0, (25f64).to_radians().cos()];
        assert_eq!(band(tilted, up, 10), 1);
        // Facing away is the last band (the tenth: index 9, odd).
        assert_eq!(band([0.0, 0.0, -1.0], up, 10), 1);
    }
}
