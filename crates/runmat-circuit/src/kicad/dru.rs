//! `.kicad_dru` custom design rules ⇄ [`RuleSpec`].
//!
//! KiCad gives later rules precedence and has no specificity, so imported selectors are
//! wrapped in `:where(…)` (zero specificity) and keep their file order: the shared precedence
//! `(specificity, priority, source order)` then reproduces KiCad's behaviour exactly.
//!
//! Supported conditions are conjunctions (`&&`) of `A.`/`B.` tests: `NetClass`, `NetName`
//! (`==` and `!=`), `Type` (`'Via'`, `'Track'`, `'Pad'`, `'Zone'`, `'Footprint'`) and
//! `inDiffPair('…')`; the `(layer outer|inner|<name>)` clause adds `:layer(…)`. Rules with
//! other conditions are kept in the tree (so writing back is lossless) and reported.

use super::sexpr::{self, SExpr};
use crate::board_rules::{props, RuleSpec, RuleValue};
use crate::geom::Coord;
use runmat_rules::{SourceKey, SourceOrder};

#[derive(Clone, Debug, Default)]
pub struct DruImport {
    pub tree: Vec<SExpr>,
    pub rules: Vec<RuleSpec>,
    /// `(rule name, reason)` for rules that could not be translated.
    pub unsupported: Vec<(String, String)>,
}

fn len_of(s: &str) -> Option<Coord> {
    let t = s.trim();
    let q = if t.chars().last().is_some_and(|c| c.is_ascii_digit()) {
        runmat_rules::parse_quantity(&format!("{t}mm"), &runmat_rules::BuiltinUnits)
    } else {
        runmat_rules::parse_quantity(t, &runmat_rules::BuiltinUnits)
    }
    .ok()?;
    Some((q.value * 1e9).round() as Coord)
}

fn side(cond: &str, who: char) -> Result<String, String> {
    let mut parts: Vec<String> = Vec::new();
    for term in cond.split("&&").map(|t| t.trim()).filter(|t| !t.is_empty()) {
        let Some(rest) = term.strip_prefix(&format!("{who}.")) else {
            if term.starts_with("A.") || term.starts_with("B.") {
                continue;
            }
            return Err(format!("unsupported term `{term}`"));
        };
        let (field, op, val) = if let Some((f, v)) = rest.split_once("!=") {
            (f.trim(), "!=", v.trim())
        } else if let Some((f, v)) = rest.split_once("==") {
            (f.trim(), "==", v.trim())
        } else if let Some(arg) = rest.strip_prefix("inDiffPair(") {
            let _ = arg;
            parts.push(":diff-pair".into());
            continue;
        } else {
            return Err(format!("unsupported term `{term}`"));
        };
        let v = val.trim_matches(|c| c == '\'' || c == '"');
        let sel = match field {
            "NetClass" => format!(":net-class(\"{v}\")"),
            "NetName" => format!(":net(\"{v}\")"),
            "Type" => match v {
                "Via" => "via".into(),
                "Track" => "track".into(),
                "Pad" => "pad".into(),
                "Zone" => "zone".into(),
                "Footprint" => "footprint".into(),
                other => return Err(format!("unsupported type `{other}`")),
            },
            other => return Err(format!("unsupported field `{other}`")),
        };
        parts.push(if op == "!=" {
            format!(":not({sel})")
        } else {
            sel
        });
    }
    // Type selectors must come first in a compound.
    parts.sort_by_key(|p| p.starts_with(':'));
    let s: String = parts.concat();
    Ok(if s.is_empty() { "*".into() } else { s })
}

/// Reads a `.kicad_dru` file. `file_rank` orders it among other rule sources.
pub fn read(text: &str, file: &str, file_rank: u32) -> Result<DruImport, String> {
    let tree = sexpr::parse_all(text).map_err(|e| e.to_string())?;
    let mut out = DruImport {
        tree: tree.clone(),
        ..Default::default()
    };
    for (i, r) in tree.iter().filter(|r| r.head() == Some("rule")).enumerate() {
        let name = r.arg(0).unwrap_or("").to_string();
        let cond = r.child_arg("condition").unwrap_or("");
        let layer = r
            .child_arg("layer")
            .map(|l| format!(":layer({l})"))
            .unwrap_or_default();
        let (a, b) = match (side(cond, 'A'), side(cond, 'B')) {
            (Ok(a), Ok(b)) => (a, b),
            (Err(e), _) | (_, Err(e)) => {
                out.unsupported.push((name, e));
                continue;
            }
        };
        let wrap = |s: &str| {
            format!("copper:where({s}{layer})")
                .replace("copper:where(*", "*:where(")
                .replace(":where()", "")
        };
        for c in r.children("constraint") {
            let kind = c.arg(0).unwrap_or("");
            let min = c.child_arg("min").and_then(len_of);
            let opt = c.child_arg("opt").and_then(len_of);
            let max = c.child_arg("max").and_then(len_of);
            let order = SourceOrder {
                file_rank,
                offset: i as u32,
            };
            let src = SourceKey::new(file, i as u32, i as u32 + 1);
            let single = |p: &str, v: RuleValue| {
                RuleSpec::single(&name, &wrap(&a))
                    .at(src.clone(), order)
                    .set(p, v)
            };
            let pair = |p: &str, v: RuleValue, other: &str| {
                RuleSpec::pair(&name, &wrap(&a), other)
                    .at(src.clone(), order)
                    .set(p, v)
            };
            let spec = match (kind, min, opt, max) {
                ("clearance", Some(m), _, _) => {
                    Some(pair(props::CLEARANCE, RuleValue::Len(m), &wrap(&b)))
                }
                ("hole_to_hole", Some(m), _, _) => {
                    Some(pair(props::HOLE_TO_HOLE, RuleValue::Len(m), &wrap(&b)))
                }
                ("courtyard_clearance", Some(m), _, _) => Some(pair(
                    props::COURTYARD_CLEARANCE,
                    RuleValue::Len(m),
                    "footprint",
                )),
                ("silk_clearance", Some(m), _, _) => {
                    Some(pair(props::SILK_CLEARANCE, RuleValue::Len(m), &wrap(&b)))
                }
                ("edge_clearance", Some(m), _, _) => {
                    Some(pair(props::EDGE_CLEARANCE, RuleValue::Len(m), "edge"))
                }
                ("track_width", _, _, _) => {
                    let mut s = RuleSpec::single(&name, &wrap(&a)).at(src.clone(), order);
                    if let Some(m) = min {
                        s = s.set(props::TRACK_WIDTH_MIN, RuleValue::Len(m));
                    }
                    if let Some(o) = opt {
                        s = s.set(props::TRACK_WIDTH, RuleValue::Len(o));
                    }
                    if let Some(m) = max {
                        s = s.set(props::TRACK_WIDTH_MAX, RuleValue::Len(m));
                    }
                    Some(s)
                }
                ("annular_width", Some(m), _, _) => {
                    Some(single(props::ANNULAR_MIN, RuleValue::Len(m)))
                }
                ("hole_size", Some(m), _, _) => Some(single(props::HOLE_MIN, RuleValue::Len(m))),
                ("diff_pair_gap", _, o, _) if o.or(min).is_some() => {
                    Some(single(props::DIFF_GAP, RuleValue::Len(o.or(min).unwrap())))
                }
                ("diff_pair_uncoupled", _, _, Some(m)) => {
                    Some(single(props::DIFF_UNCOUPLED_MAX, RuleValue::Len(m)))
                }
                ("skew", _, _, Some(m)) => Some(single(props::SKEW_MAX, RuleValue::Len(m))),
                ("length", _, _, _) if min.or(max).is_some() => {
                    let mut s = RuleSpec::single(&name, &wrap(&a)).at(src.clone(), order);
                    if let Some(m) = min {
                        s = s.set(props::LENGTH_MIN, RuleValue::Len(m));
                    }
                    if let Some(m) = max {
                        s = s.set(props::LENGTH_MAX, RuleValue::Len(m));
                    }
                    Some(s)
                }
                ("thermal_relief_gap", Some(m), _, _) => {
                    Some(single(props::THERMAL_GAP, RuleValue::Len(m)))
                }
                ("thermal_spoke_width", Some(m), _, _) | ("thermal_spoke_width", _, Some(m), _) => {
                    Some(single(props::SPOKE_WIDTH, RuleValue::Len(m)))
                }
                ("min_resolved_spokes", _, _, _) => c
                    .arg(1)
                    .and_then(|v| v.parse().ok())
                    .map(|n: i64| single(props::SPOKE_MIN, RuleValue::Int(n))),
                ("zone_connection", _, _, _) => c.arg(1).map(|m| {
                    let t = match m {
                        "solid" => "solid",
                        "none" => "none",
                        _ => "thermal",
                    };
                    single(props::ZONE_CONNECT, RuleValue::Text(t.into()))
                }),
                _ => {
                    out.unsupported
                        .push((name.clone(), format!("constraint `{kind}`")));
                    None
                }
            };
            out.rules.extend(spec);
        }
    }
    Ok(out)
}

/// Writes the original rule file back, token for token.
pub fn write(imp: &DruImport) -> String {
    imp.tree.iter().map(sexpr::write).collect()
}
