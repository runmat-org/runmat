//! `assembly.visualization`: the assembly's part occurrences sorted by a property, largest
//! first, and colored by it on the model while the results are open.
//!
//! - `mass`: each occurrence's mass ([`crate::mass`]); one without a density has none.
//! - `volume`: each occurrence's volume.
//! - `cost`: the occurrence's `cost` property, else its part output's `cost` property (a number
//!   or a quantity, read as written).
//! - `count`: how many occurrences of the same part and configuration the assembly holds.
//!
//! The colors run in five equal bands from the smallest value (blue) to the largest (red);
//! occurrences with no value are grey.

use super::{Analysis, FaceColor, FaceColors, LegendEntry, Row};
use crate::mass::{assembly_mass, PartMass};
use crate::runtime::{resolve_path, Files};
use runmat_cad::model::items::{Property, PropertyValue};
use runmat_cad::value::ValueExpr;
use serde_json::{Map, Value};

const BANDS: [[f32; 4]; 5] = [
    [0.23, 0.42, 0.85, 1.0],
    [0.20, 0.68, 0.78, 1.0],
    [0.36, 0.72, 0.36, 1.0],
    [0.93, 0.70, 0.20, 1.0],
    [0.86, 0.27, 0.22, 1.0],
];
const NONE: [f32; 4] = [0.6, 0.6, 0.6, 1.0];

fn number(value: &PropertyValue) -> Option<(f64, String)> {
    match value {
        PropertyValue::Quantity(ValueExpr::Quantity { value, unit }) => {
            Some((*value, unit.clone()))
        }
        PropertyValue::Quantity(ValueExpr::Number { value }) => Some((*value, String::new())),
        PropertyValue::Text(t) => t.trim().parse().ok().map(|v| (v, String::new())),
        _ => None,
    }
}

fn cost_of(list: &[Property]) -> Option<(f64, String)> {
    list.iter()
        .find(|p| p.name.as_str() == "cost")
        .and_then(|p| number(&p.value))
}

/// The `cost` of part occurrence `p`: its top-level component's property, else its part
/// output's.
fn cost(path: &str, text: &str, files: &Files, p: &PartMass) -> Option<(f64, String)> {
    let top = p.path.split('/').next().unwrap_or(&p.path);
    let assembly = runmat_cad::syntax::parse_assembly(path, text).document?;
    if let Some(c) = assembly
        .occurrences
        .iter()
        .find(|o| o.name == top)
        .and_then(|o| cost_of(&o.properties))
    {
        return Some(c);
    }
    let file = resolve_path(path, &p.file, files);
    let part = runmat_cad::syntax::parse_part(&file, files.get(&file)?).document?;
    part.parts.iter().find_map(|o| cost_of(&o.properties))
}

pub(super) fn visualization(
    path: &str,
    text: &str,
    files: &Files,
    configuration: Option<&str>,
    values: &Map<String, Value>,
) -> Result<Analysis, String> {
    let by = values.get("by").and_then(Value::as_str).unwrap_or("mass");
    let report = assembly_mass(path, text, files, configuration)?;
    let (label, unit): (&str, String) = match by {
        "mass" => ("Mass", "kg".into()),
        "volume" => ("Volume", "cm³".into()),
        "cost" => ("Cost", String::new()),
        "count" => ("Count", String::new()),
        other => return Err(format!("components can't be sorted by `{other}`")),
    };
    let mut cost_unit = String::new();
    let mut measured: Vec<(&PartMass, Option<f64>)> = report
        .parts
        .iter()
        .map(|p| {
            let v = match by {
                "mass" => p.mass,
                "volume" => Some(p.volume * 1e6),
                "cost" => cost(path, text, files, p).map(|(v, u)| {
                    if cost_unit.is_empty() {
                        cost_unit = u;
                    }
                    v
                }),
                _ => Some(
                    report
                        .parts
                        .iter()
                        .filter(|q| q.file == p.file && q.configuration == p.configuration)
                        .count() as f64,
                ),
            };
            (p, v)
        })
        .collect();
    let unit = if by == "cost" { cost_unit } else { unit };
    measured.sort_by(|a, b| {
        b.1.unwrap_or(f64::NEG_INFINITY)
            .total_cmp(&a.1.unwrap_or(f64::NEG_INFINITY))
            .then_with(|| a.0.path.cmp(&b.0.path))
    });
    let known: Vec<f64> = measured.iter().filter_map(|m| m.1).collect();
    let (lo, hi) = known
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), v| {
            (l.min(*v), h.max(*v))
        });
    let band = |v: f64| {
        if !(hi > lo) {
            return BANDS.len() - 1;
        }
        (((v - lo) / (hi - lo)) * BANDS.len() as f64)
            .floor()
            .min(4.0) as usize
    };
    let show = |v: f64| {
        let text = if by == "count" {
            format!("{v}")
        } else {
            let r = if v.abs() >= 100.0 {
                format!("{v:.1}")
            } else {
                format!("{v:.3}")
            };
            r.trim_end_matches('0').trim_end_matches('.').to_owned()
        };
        if unit.is_empty() {
            text
        } else {
            format!("{text} {unit}")
        }
    };
    let width = (hi - lo) / BANDS.len() as f64;
    let mut legend: Vec<LegendEntry> = (0..BANDS.len())
        .map(|i| LegendEntry {
            label: if hi > lo {
                format!(
                    "{} to {}",
                    show(lo + width * i as f64),
                    show(lo + width * (i + 1) as f64)
                )
            } else {
                known.first().map(|v| show(*v)).unwrap_or_default()
            },
            color: BANDS[i],
        })
        .collect();
    if !(hi > lo) {
        legend = legend.split_off(BANDS.len() - 1);
    }
    let offset = if hi > lo { 0 } else { BANDS.len() - 1 };
    let none_entry = legend.len();
    let missing = measured.iter().any(|m| m.1.is_none());
    if missing {
        legend.push(LegendEntry {
            label: "No value".into(),
            color: NONE,
        });
    }
    let mut faces = Vec::new();
    let rows: Vec<Row> = measured
        .iter()
        .map(|(p, v)| {
            let entry = v.map_or(none_entry, |v| band(v) - offset);
            if let Some(item) = &p.item {
                faces.push(FaceColor {
                    item: item.clone(),
                    entry,
                });
            }
            Row {
                cells: vec![
                    p.path.clone(),
                    p.file.rsplit('/').next().unwrap_or(&p.file).to_owned(),
                    v.map(show).unwrap_or_else(|| "—".into()),
                ],
                items: p.item.iter().cloned().collect(),
                severity: None,
                legend: Some(entry),
                reference: None,
            }
        })
        .collect();
    let summary = format!(
        "{} part occurrence{} by {}{}",
        rows.len(),
        if rows.len() == 1 { "" } else { "s" },
        label.to_lowercase(),
        if missing {
            format!(
                " · {} with no value",
                measured.iter().filter(|m| m.1.is_none()).count()
            )
        } else {
            String::new()
        }
    );
    Ok(Analysis {
        title: format!("Assembly visualization: {}", label.to_lowercase()),
        summary,
        columns: ["Component", "Part", label].map(String::from).to_vec(),
        rows,
        problems: if by == "mass" {
            report.problems.clone()
        } else {
            Vec::new()
        },
        data: Value::Null,
        colors: Some(FaceColors { legend, faces }),
        settings: vec![format!("Sorted by {}", label.to_lowercase())],
        arrow: None,
        dimensions: Vec::new(),
        field: None,
    })
}
