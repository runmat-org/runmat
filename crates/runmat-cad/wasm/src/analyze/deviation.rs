//! `evaluate.deviation`: the angle between the two faces along each picked edge (every edge
//! between two faces when none is picked), from the exact surface normals of both faces at the
//! edge's points. A tangent edge measures 0°; a sharp edge of a box, 90°. Each edge is a row with
//! its least, greatest and mean angle; edges whose greatest angle passes `tolerance` are flagged.
//!
//! Values: `edges` (picks), `tolerance` (degrees; the angle under which faces count as
//! tangent).

use super::{Analysis, Row};
use runmat_cad::evaluate::{AnalysisRequest, AnalysisResult, FacePoints, FieldQuantity};
use serde_json::{json, Map, Value};

/// The angle between two unit normals, degrees.
pub(super) fn angle(a: [f64; 3], b: [f64; 3]) -> f64 {
    (a[0] * b[0] + a[1] * b[1] + a[2] * b[2])
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees()
}

pub(super) fn deviation(
    path: &str,
    text: &str,
    configuration: Option<&str>,
    values: &Map<String, Value>,
) -> Result<Analysis, String> {
    let part = crate::runtime::regenerate_part_in(path, text, configuration, &Default::default())
        .map_err(|d| d.first().map(|d| d.message.clone()).unwrap_or_default())?;
    let tolerance = values
        .get("tolerance")
        .and_then(Value::as_f64)
        .unwrap_or(0.5);
    let picks = crate::features::picks_of(values, "edges");
    let mut rows = Vec::new();
    let mut found = Vec::new();
    let mut over = 0usize;
    let mut matched = 0usize;
    for body in part.display.bodies.iter().filter(|b| !b.flat_pattern) {
        let Some(state) = part.result.state.bodies.get(body.index as usize) else {
            continue;
        };
        for edge in &body.edges {
            let picked = edge.reference.as_ref().is_some_and(|r| picks.contains(r));
            if !picks.is_empty() && !picked {
                continue;
            }
            matched += usize::from(picked);
            if edge.faces[0] == edge.faces[1] {
                if picked {
                    rows.push(Row {
                        cells: vec![
                            format!("Edge {}", edge.index + 1),
                            "—".into(),
                            "—".into(),
                            "—".into(),
                            "A seam on one face".into(),
                        ],
                        items: edge.reference.iter().cloned().collect(),
                        severity: Some("info"),
                        legend: None,
                        reference: edge.reference.clone(),
                    });
                }
                continue;
            }
            let points: Vec<[f64; 3]> = edge
                .points
                .chunks_exact(3)
                .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
                .collect();
            if points.is_empty() {
                continue;
            }
            let request = AnalysisRequest::SurfaceField {
                quantity: FieldQuantity::Normal,
                faces: edge
                    .faces
                    .iter()
                    .map(|f| FacePoints {
                        face: *f,
                        points: points.clone(),
                    })
                    .collect(),
            };
            let normals = match part.kernel.analyze(&request, &[&state.handle]) {
                Ok(AnalysisResult::SurfaceField { faces }) => faces,
                Ok(other) => return Err(format!("the kernel answered {other:?}")),
                Err(e) => return Err(format!("the kernel could not measure the edge: {e:?}")),
            };
            let angles: Vec<f64> = normals[0]
                .iter()
                .zip(&normals[1])
                .filter_map(|(a, b)| Some(angle(a.normal?, b.normal?)))
                .collect();
            if angles.is_empty() {
                continue;
            }
            let lo = angles.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = angles.iter().copied().fold(0.0, f64::max);
            let mean = angles.iter().sum::<f64>() / angles.len() as f64;
            let flagged = hi > tolerance;
            over += usize::from(flagged);
            let face_label = |i: u32| {
                body.faces
                    .iter()
                    .find(|f| f.index == i)
                    .map(|f| crate::face_labels::label(body.index, f))
                    .unwrap_or_else(|| format!("Face {}", i + 1))
            };
            found.push(json!({
                "edge": edge.reference, "min": lo, "max": hi, "mean": mean, "points": angles.len(),
            }));
            rows.push(Row {
                cells: vec![
                    format!("Edge {}", edge.index + 1),
                    format!("{lo:.3}°"),
                    format!("{hi:.3}°"),
                    format!("{mean:.3}°"),
                    format!(
                        "{} · {}",
                        face_label(edge.faces[0]),
                        face_label(edge.faces[1])
                    ),
                ],
                items: edge.reference.iter().cloned().collect(),
                severity: flagged.then_some("info"),
                legend: None,
                reference: edge.reference.clone(),
            });
        }
    }
    if !picks.is_empty() && matched == 0 {
        return Err("the picked edges aren't edges of this part".into());
    }
    if rows.is_empty() {
        return Err("the part has no edges between two faces".into());
    }
    // The sharpest edges first.
    rows.sort_by(|a, b| {
        let g = |r: &Row| {
            r.cells[2]
                .trim_end_matches('°')
                .parse::<f64>()
                .unwrap_or(-1.0)
        };
        g(b).partial_cmp(&g(a)).unwrap_or(std::cmp::Ordering::Equal)
    });
    let plural = |n: usize| if n == 1 { "" } else { "s" };
    let tangent = rows.len() - over;
    Ok(Analysis {
        title: "Deviation".into(),
        summary: format!(
            "{} edge{} · {tangent} within {tolerance}° (tangent) · {over} over",
            rows.len(),
            plural(rows.len())
        ),
        columns: [
            "Edge",
            "Least angle",
            "Greatest angle",
            "Mean angle",
            "Between",
        ]
        .map(String::from)
        .to_vec(),
        rows,
        problems: Vec::new(),
        data: json!({ "tolerance": tolerance, "edges": found }),
        colors: None,
        settings: vec![
            format!("Tangent within: {tolerance}°"),
            if picks.is_empty() {
                "Edges: every edge between two faces".into()
            } else {
                format!("Edges: {} picked", picks.len())
            },
        ],
        arrow: None,
        dimensions: Vec::new(),
        field: None,
    })
}
