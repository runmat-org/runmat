//! A part's sheet-metal defaults: the gauge table and gauge, and the bend table or K-factor
//! that every sheet body of the part follows, kept as data in the project.
//!
//! Written in the part as `sheet_metal = { gauges = steel, gauge = "16", bend_table = din6935 }`.
//!
//! - `gauges`: a built-in gauge table ([`BUILTIN_GAUGES`]) or a CSV file in the project with the
//!   columns `gauge,thickness_mm,bend_radius_mm,k_factor`.
//! - `gauge`: the gauge every sheet body takes its thickness, inner bend radius and K-factor
//!   from (the K-factor when the body's rule is a K-factor).
//! - `bend_table`: `din6935` (the K-factor from the radius-to-thickness ratio, DIN 6935), or a CSV
//!   file in the project with the columns `thickness_mm,radius_mm,angle_deg,allowance_mm`.
//! - `k`: a K-factor every sheet body uses, when there is no bend table.
//!
//! [`resolve`] reads the named tables and writes them into each sheet body's rules before
//! regeneration; a table file is a dependency of the part, so editing it rebuilds the part.
//!
//! Built-in gauge tables:
//! - `steel`: Manufacturers' Standard Gauge for sheet steel, gauges 3 to 28.
//! - `aluminum`: Brown & Sharpe (American Wire Gauge), `d = 0.005 in × 92^((36 − n) / 39)`
//!   (ASTM B258), gauges 8 to 24, as aluminum sheet is ordered by gauge.
//!
//! A built-in row's inner bend radius equals its thickness, and its K-factor is DIN 6935's at
//! that ratio ([`din6935_k`]: 0.325 at `r/t = 1`).

use super::{BendRule, BendTable, BendTableRow, GaugeRow, GaugeTable, SheetMetalFeature};
use crate::model::{FeatureKind, PartDocument};
use serde::{Deserialize, Serialize};

/// `sheet_metal = { … }` on a part.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SheetMetalDefaults {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gauges: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gauge: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bend_table: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub k: Option<f64>,
}

/// The built-in gauge tables by name, with what they are.
pub const BUILTIN_GAUGES: [(&str, &str); 2] = [
    ("steel", "Manufacturers' Standard Gauge, sheet steel"),
    ("aluminum", "Brown & Sharpe (ASTM B258), aluminum sheet"),
];

/// The built-in bend rule `din6935`.
pub const DIN6935: &str = "din6935";

const INCH: f64 = 0.0254;

/// Manufacturers' Standard Gauge for sheet steel: gauge and thickness in inches.
const MSG_STEEL: [(u32, f64); 26] = [
    (3, 0.2391),
    (4, 0.2242),
    (5, 0.2092),
    (6, 0.1943),
    (7, 0.1793),
    (8, 0.1644),
    (9, 0.1495),
    (10, 0.1345),
    (11, 0.1196),
    (12, 0.1046),
    (13, 0.0897),
    (14, 0.0747),
    (15, 0.0673),
    (16, 0.0598),
    (17, 0.0538),
    (18, 0.0478),
    (19, 0.0418),
    (20, 0.0359),
    (21, 0.0329),
    (22, 0.0299),
    (23, 0.0269),
    (24, 0.0239),
    (25, 0.0209),
    (26, 0.0179),
    (27, 0.0164),
    (28, 0.0149),
];

/// DIN 6935's K-factor for an inner radius `r` and thickness `t`: the correction factor
/// `k = 0.65 + 0.5 log10(r / t)` (1 above `r/t = 5`), halved because DIN 6935 measures the
/// neutral axis from the inner face in half-thicknesses. Clamped to `[0, 0.5]`.
pub fn din6935_k(r: f64, t: f64) -> f64 {
    if t <= 0.0 {
        return 0.5;
    }
    let ratio = r / t;
    if ratio > 5.0 {
        return 0.5;
    }
    if ratio <= 0.0 {
        return 0.0;
    }
    ((0.65 + 0.5 * ratio.log10()) / 2.0).clamp(0.0, 0.5)
}

/// A built-in gauge table, SI.
pub fn builtin_gauges(name: &str) -> Option<GaugeTable> {
    let row = |gauge: u32, inches: f64| {
        let t = inches * INCH;
        GaugeRow {
            gauge: gauge.to_string(),
            thickness: t,
            bend_radius: t,
            k_factor: din6935_k(t, t),
        }
    };
    let rows = match name {
        "steel" => MSG_STEEL.iter().map(|&(g, t)| row(g, t)).collect(),
        "aluminum" => (8..=24)
            .map(|n: u32| {
                let d = 0.005 * 92f64.powf((36.0 - n as f64) / 39.0);
                // Tabulated to four decimal places of an inch.
                row(n, (d * 1e4).round() / 1e4)
            })
            .collect(),
        _ => return None,
    };
    Some(GaugeTable {
        name: name.into(),
        rows,
    })
}

/// The cells of a CSV line, trimmed and unquoted.
fn cells(line: &str) -> Vec<String> {
    line.split(',')
        .map(|c| c.trim().trim_matches('"').trim().to_owned())
        .collect()
}

/// The data lines of a CSV with the header `want`: blank lines and `#` comments skipped.
fn data(name: &str, text: &str, want: &[&str]) -> Result<Vec<(usize, Vec<String>)>, String> {
    let mut lines = text
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty() && !l.trim_start().starts_with('#'));
    let header = lines
        .next()
        .map(|(_, l)| cells(l))
        .ok_or_else(|| format!("`{name}` is empty; its first line is `{}`", want.join(",")))?;
    if header.iter().map(String::as_str).ne(want.iter().copied()) {
        return Err(format!(
            "`{name}` starts `{}`; a table's first line is `{}`",
            header.join(","),
            want.join(",")
        ));
    }
    Ok(lines.map(|(i, l)| (i + 1, cells(l))).collect())
}

fn number(name: &str, line: usize, cell: &str) -> Result<f64, String> {
    cell.parse::<f64>()
        .ok()
        .filter(|v| v.is_finite())
        .ok_or_else(|| format!("`{name}` line {line}: `{cell}` isn't a number"))
}

/// A gauge table CSV: `gauge,thickness_mm,bend_radius_mm,k_factor`.
pub fn parse_gauge_csv(name: &str, text: &str) -> Result<GaugeTable, String> {
    let want = ["gauge", "thickness_mm", "bend_radius_mm", "k_factor"];
    let mut rows = Vec::new();
    for (line, c) in data(name, text, &want)? {
        if c.len() != 4 {
            return Err(format!(
                "`{name}` line {line} has {} cells; a row has 4",
                c.len()
            ));
        }
        rows.push(GaugeRow {
            gauge: c[0].clone(),
            thickness: number(name, line, &c[1])? / 1000.0,
            bend_radius: number(name, line, &c[2])? / 1000.0,
            k_factor: number(name, line, &c[3])?,
        });
    }
    Ok(GaugeTable {
        name: name.into(),
        rows,
    })
}

/// A bend table CSV: `thickness_mm,radius_mm,angle_deg,allowance_mm`.
pub fn parse_bend_csv(name: &str, text: &str) -> Result<BendTable, String> {
    let want = ["thickness_mm", "radius_mm", "angle_deg", "allowance_mm"];
    let mut rows = Vec::new();
    for (line, c) in data(name, text, &want)? {
        if c.len() != 4 {
            return Err(format!(
                "`{name}` line {line} has {} cells; a row has 4",
                c.len()
            ));
        }
        rows.push(BendTableRow {
            thickness: number(name, line, &c[0])? / 1000.0,
            radius: number(name, line, &c[1])? / 1000.0,
            angle: number(name, line, &c[2])?.to_radians(),
            allowance: number(name, line, &c[3])? / 1000.0,
        });
    }
    if rows.is_empty() {
        return Err(format!("`{name}` has no rows"));
    }
    Ok(BendTable {
        name: name.into(),
        rows,
    })
}

/// A gauge as it is matched: `16`, `16 ga` and `16 gauge` are gauge 16.
pub fn gauge_key(gauge: &str) -> String {
    let g = gauge.trim().to_lowercase();
    let stem = g
        .strip_suffix("gauge")
        .or_else(|| g.strip_suffix("ga"))
        .unwrap_or(&g);
    stem.trim().to_owned()
}

/// The gauge table `gauges` names: built-in, or a project file `read` returns.
pub fn gauge_table(
    gauges: &str,
    read: &dyn Fn(&str) -> Option<String>,
) -> Result<GaugeTable, String> {
    if let Some(t) = builtin_gauges(gauges) {
        return Ok(t);
    }
    let text = read(gauges).ok_or_else(|| {
        format!(
            "the gauge table `{gauges}` isn't a built-in table ({}) or a file in the project",
            BUILTIN_GAUGES.map(|(n, _)| n).join(", ")
        )
    })?;
    parse_gauge_csv(gauges, &text)
}

/// The rule `bend_table` names: DIN 6935, or a project bend table `read` returns.
pub fn bend_rule(
    bend_table: &str,
    read: &dyn Fn(&str) -> Option<String>,
) -> Result<BendRule, String> {
    if bend_table == DIN6935 {
        return Ok(BendRule::Din6935);
    }
    let text = read(bend_table).ok_or_else(|| {
        format!("the bend table `{bend_table}` isn't `{DIN6935}` or a file in the project")
    })?;
    Ok(BendRule::Table {
        table: parse_bend_csv(bend_table, &text)?,
    })
}

/// The project files a part's sheet-metal defaults read.
pub fn files(defaults: &SheetMetalDefaults) -> Vec<String> {
    let mut out = Vec::new();
    if let Some(g) = defaults
        .gauges
        .as_deref()
        .filter(|g| builtin_gauges(g).is_none())
    {
        out.push(g.to_owned());
    }
    if let Some(b) = defaults.bend_table.as_deref().filter(|b| *b != DIN6935) {
        out.push(b.to_owned());
    }
    out
}

/// The part with its sheet-metal defaults written into every sheet body's rules, reading table
/// files through `read` (project paths as written). A table that can't be read leaves the rules
/// as written; its problem is returned.
pub fn resolve(
    document: &PartDocument,
    read: &dyn Fn(&str) -> Option<String>,
) -> (PartDocument, Vec<String>) {
    let Some(defaults) = &document.sheet_metal else {
        return (document.clone(), Vec::new());
    };
    let mut problems = Vec::new();
    let gauge = match &defaults.gauge {
        Some(g) => {
            let table = gauge_table(defaults.gauges.as_deref().unwrap_or("steel"), read);
            match table {
                Ok(table) => {
                    let key = gauge_key(g);
                    match table.rows.iter().find(|r| gauge_key(&r.gauge) == key) {
                        Some(row) => Some((table.clone(), row.gauge.clone())),
                        None => {
                            problems.push(format!(
                                "the gauge table `{}` has no gauge `{g}`; it has {}",
                                table.name,
                                table
                                    .rows
                                    .iter()
                                    .map(|r| r.gauge.as_str())
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            ));
                            None
                        }
                    }
                }
                Err(e) => {
                    problems.push(e);
                    None
                }
            }
        }
        None => None,
    };
    let rule = match (&defaults.bend_table, defaults.k) {
        (Some(b), _) => match bend_rule(b, read) {
            Ok(rule) => Some(rule),
            Err(e) => {
                problems.push(e);
                None
            }
        },
        (None, Some(k)) => Some(BendRule::KFactor {
            k: crate::value::ValueExpr::number(k),
        }),
        (None, None) => None,
    };
    let mut out = document.clone();
    for f in out.features.iter_mut() {
        let FeatureKind::SheetMetal(sm) = &mut f.kind else {
            continue;
        };
        let rules = match sm {
            SheetMetalFeature::BaseFlange { rules, .. }
            | SheetMetalFeature::LoftedBend { rules, .. }
            | SheetMetalFeature::ConvertToSheetMetal { rules, .. } => rules,
            _ => continue,
        };
        if let Some(rule) = &rule {
            rules.rule = rule.clone();
        }
        if let Some(g) = &gauge {
            rules.gauge = Some(g.clone());
        }
    }
    (out, problems)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn built_in_gauges_are_the_published_thicknesses() {
        let steel = builtin_gauges("steel").unwrap();
        let g16 = steel.rows.iter().find(|r| r.gauge == "16").unwrap();
        assert!((g16.thickness - 0.0598 * INCH).abs() < 1e-12);
        assert_eq!(g16.bend_radius, g16.thickness);
        assert!((g16.k_factor - 0.325).abs() < 1e-12);
        let al = builtin_gauges("aluminum").unwrap();
        let t = |g: &str| al.rows.iter().find(|r| r.gauge == g).unwrap().thickness / INCH;
        // ASTM B258's tabulated values: 10 AWG 0.1019 in, 16 AWG 0.0508 in, 20 AWG 0.0320 in.
        assert!((t("10") - 0.1019).abs() < 1e-9, "{}", t("10"));
        assert!((t("16") - 0.0508).abs() < 1e-9, "{}", t("16"));
        assert!((t("20") - 0.0320).abs() < 1e-9, "{}", t("20"));
        assert!(builtin_gauges("brass").is_none());
    }

    #[test]
    fn din6935_k_follows_the_ratio() {
        assert!((din6935_k(1.0, 1.0) - 0.325).abs() < 1e-12);
        assert!((din6935_k(2.0, 1.0) - (0.65 + 0.5 * 2f64.log10()) / 2.0).abs() < 1e-12);
        assert_eq!(din6935_k(6.0, 1.0), 0.5);
        assert_eq!(din6935_k(0.01, 1.0), 0.0);
    }

    #[test]
    fn project_tables_parse_and_say_what_is_wrong() {
        let g = parse_gauge_csv(
            "t/g.csv",
            "# shop table\ngauge,thickness_mm,bend_radius_mm,k_factor\n16 ga,1.5,1.5,0.4\n",
        )
        .unwrap();
        assert_eq!(g.rows[0].gauge, "16 ga");
        assert!((g.rows[0].thickness - 0.0015).abs() < 1e-15);
        let b = parse_bend_csv(
            "t/b.csv",
            "thickness_mm,radius_mm,angle_deg,allowance_mm\n1.5,1.5,90,2.9\n",
        )
        .unwrap();
        assert!((b.rows[0].angle - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
        let e = parse_bend_csv("t/b.csv", "t,r,a,ba\n").unwrap_err();
        assert!(
            e.contains("first line is `thickness_mm,radius_mm,angle_deg,allowance_mm`"),
            "{e}"
        );
        let e = parse_gauge_csv(
            "t/g.csv",
            "gauge,thickness_mm,bend_radius_mm,k_factor\n16,x,1,0.4",
        )
        .unwrap_err();
        assert_eq!(e, "`t/g.csv` line 2: `x` isn't a number");
        assert_eq!(gauge_key("16 ga"), "16");
        assert_eq!(gauge_key(" 16 Gauge"), "16");
    }
}
