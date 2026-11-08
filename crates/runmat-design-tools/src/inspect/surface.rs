//! The `surface` section: where a ray meets a part's or an assembly's bodies, with the face and
//! its normal there ([`runmat_cad_wasm::probe`]), so a part can be seated on a curved body. Also
//! the occurrence names a check is limited to (`changed`), resolved to occurrence paths.

use crate::{name_matches, Files, ToolError};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{json, Value};

/// A ray, in the design's millimetres.
#[derive(Clone, Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Ray {
    /// The start point, x, y, z in mm (the assembly's frame for an assembly). Start outside the
    /// body, for example below a lower shell to find its underside.
    pub from: [f64; 3],
    /// The direction, x, y, z (any length): `[0, 0, 1]` looks up.
    pub direction: [f64; 3],
    /// Assemblies: only these occurrences' bodies (names or paths; `*` globs). Every part's
    /// otherwise.
    #[serde(default)]
    pub occurrences: Option<Vec<String>>,
}

/// The occurrence paths `names` match: an exact path, else each path the name matches
/// (case-insensitive substring, or a `*` glob). A name that matches nothing is an error that
/// lists the occurrences.
pub(super) fn occurrences_named(names: &[String], paths: &[String]) -> Result<Vec<String>, ToolError> {
    let mut out = Vec::new();
    for name in names {
        let matched: Vec<&String> = if paths.contains(name) {
            paths.iter().filter(|p| *p == name).collect()
        } else {
            paths.iter().filter(|p| name_matches(Some(name), p)).collect()
        };
        if matched.is_empty() {
            return Err(ToolError::invalid(format!(
                "no occurrence matches `{name}`; the assembly's occurrences are {}",
                paths.join(", ")
            )));
        }
        for p in matched {
            if !out.contains(p) {
                out.push(p.clone());
            }
        }
    }
    Ok(out)
}

fn round(v: f64, places: i32) -> f64 {
    let k = 10f64.powi(places);
    (v * k).round() / k
}

/// The `surface` section: the hits nearest first (at most 8), points and distances in mm.
pub(super) fn surface(
    ray: Option<&Ray>,
    path: &str,
    text: &str,
    files: &Files,
    occurrence_paths: Option<&[String]>,
) -> Result<Value, ToolError> {
    let ray = ray.ok_or_else(|| ToolError::invalid("the surface section needs a `ray`: `from` (mm) and `direction`"))?;
    let only = match (&ray.occurrences, occurrence_paths) {
        (Some(names), Some(paths)) => Some(occurrences_named(names, paths)?),
        _ => None,
    };
    let from = ray.from.map(|v| v / 1000.0);
    let hits = runmat_cad_wasm::probe::probe(path, text, files, from, ray.direction, only.as_deref())
        .map_err(|e| ToolError::new("not_compiled", e))?;
    let rows: Vec<Value> = hits
        .iter()
        .take(8)
        .map(|h| {
            let mut row = json!({
                "point_mm": h.point.map(|v| round(v * 1000.0, 2)),
                "normal": h.normal.map(|v| round(v, 4)),
                "distance_mm": round(h.distance * 1000.0, 2),
                "entering": h.entering,
            });
            if !h.occurrence.is_empty() {
                row["occurrence"] = json!(h.occurrence);
            }
            if let Some(face) = &h.face {
                row["face"] = json!(face);
            }
            if let Some(reference) = &h.reference {
                row["reference"] = json!(reference);
            }
            row
        })
        .collect();
    Ok(json!({
        "hits": rows,
        "note": if rows.is_empty() {
            "the ray meets no body"
        } else {
            "the first hit is the surface the ray reaches first; points lie on the display tessellation, within its chordal tolerance of the exact surface"
        },
    }))
}
