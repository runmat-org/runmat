//! A part's settings, written as data in the part:
//!
//! - `manage.document-settings`: `settings = { units, decimals, standard, tolerance }`
//!   ([`runmat_cad::assembly::settings`]): readouts in the part's units and decimals, its 3D
//!   annotations checked against its drafting standard, Measure's general tolerance.
//! - `sheet-metal.gauge`: `sheet_metal = { gauges, gauge }`, the gauge table (built-in or a CSV in
//!   the project) and the gauge every sheet body takes its thickness, radius and K-factor from.
//! - `sheet-metal.bend-table`: `sheet_metal = { bend_table }` or `{ k }`, the bend rule every sheet
//!   body follows ([`runmat_cad::model::sheet_metal::tables`]).
//!
//! Each checks what it names before writing: a gauge the table lacks, or a table file that isn't
//! in the project or doesn't parse, is refused with the reason.

use super::super::{model, text_value, EditRequest};
use crate::runtime::{resolve_path, Files};
use runmat_cad::assembly::settings::{DocumentSettings, GeneralTolerance, UnitSystem};
use runmat_cad::drawing::standard::DrawingStandard;
use runmat_cad::model::sheet_metal::tables::{self, SheetMetalDefaults};
use runmat_cad::model::PartDocument;
use runmat_cad::syntax::edit::{edit_part, EditFailure, SourceChange};
use serde_json::{Map, Value};

type Values = Map<String, Value>;

/// The commands this module answers.
pub(in crate::edit) fn edit(
    path: &str,
    text: &str,
    files: &Files,
    request: &EditRequest,
) -> Option<Result<(SourceChange, String), EditFailure>> {
    Some(match request.command.as_str() {
        "manage.document-settings" => document_settings(path, text, &request.values),
        "sheet-metal.gauge" => gauge(path, text, files, &request.values),
        "sheet-metal.bend-table" => bend_table(path, text, files, &request.values),
        _ => return None,
    })
}

fn number(values: &Values, field: &str) -> Result<Option<f64>, EditFailure> {
    match values.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(n)) => Ok(n.as_f64()),
        Some(Value::String(s)) if s.trim().is_empty() => Ok(None),
        Some(Value::String(s)) => s
            .trim()
            .parse::<f64>()
            .map(Some)
            .map_err(|_| model(format!("`{s}` isn't a number"))),
        Some(Value::Object(o)) => match o.get("value") {
            Some(Value::Object(v)) => Ok(v.get("value").and_then(Value::as_f64)),
            Some(v) => Ok(v.as_f64()),
            None => Ok(None),
        },
        Some(other) => Err(model(format!("`{field}` isn't a number: {other}"))),
    }
}

fn document_settings(
    path: &str,
    text: &str,
    values: &Values,
) -> Result<(SourceChange, String), EditFailure> {
    let units = match text_value(values, "units") {
        "mmgs" | "" => UnitSystem::Mmgs,
        "mks" => UnitSystem::Mks,
        "ips" => UnitSystem::Ips,
        other => {
            return Err(model(format!(
                "`{other}` isn't a unit system: choose mmgs, mks or ips"
            )))
        }
    };
    let standard = match text_value(values, "standard") {
        "iso" | "" => None,
        "asme" => Some(DrawingStandard::Asme),
        other => {
            return Err(model(format!(
                "`{other}` isn't a standard: choose iso or asme"
            )))
        }
    };
    let tolerance = match text_value(values, "tolerance") {
        "none" | "" => None,
        "fine" => Some(GeneralTolerance::Fine),
        "medium" => Some(GeneralTolerance::Medium),
        "coarse" => Some(GeneralTolerance::Coarse),
        "very_coarse" => Some(GeneralTolerance::VeryCoarse),
        other => {
            return Err(model(format!(
                "`{other}` isn't an ISO 2768-1 class: choose fine, medium, coarse or very_coarse"
            )))
        }
    };
    let decimals = match number(values, "decimals")? {
        None => None,
        Some(d) if (0.0..=8.0).contains(&d) && d.fract() == 0.0 => Some(d as u32),
        Some(d) => {
            return Err(model(format!(
                "decimals are a whole number from 0 to 8, not {d}"
            )))
        }
    };
    let settings = DocumentSettings {
        units,
        standard,
        tolerance,
        decimals,
    };
    let mut same = false;
    let change = edit_part(text, path, |document: &mut PartDocument| {
        same = document.settings.as_ref() == Some(&settings);
        document.settings = Some(settings.clone());
        Ok(())
    })?;
    if same {
        return Err(model("the part already has these settings"));
    }
    let mut summary = format!("Units {}", units.key());
    if let Some(d) = decimals {
        summary += &format!(", {d} decimals");
    }
    summary += match standard {
        Some(DrawingStandard::Asme) => ", ASME Y14.5",
        _ => ", ISO",
    };
    if let Some(t) = tolerance {
        summary += &format!(", ISO 2768-{}", t.letter());
    }
    Ok((change, summary))
}

/// The part's sheet-metal defaults changed by `f`, refused when nothing changes.
fn set_defaults(
    path: &str,
    text: &str,
    f: impl Fn(&mut SheetMetalDefaults),
) -> Result<SourceChange, EditFailure> {
    let mut same = false;
    let change = edit_part(text, path, |document: &mut PartDocument| {
        let before = document.sheet_metal.clone();
        let mut defaults = before.clone().unwrap_or_default();
        f(&mut defaults);
        let after = (defaults != SheetMetalDefaults::default()).then_some(defaults);
        same = after == before;
        document.sheet_metal = after;
        Ok(())
    })?;
    if same {
        return Err(model("the part already has this sheet metal setting"));
    }
    Ok(change)
}

fn reader<'a>(path: &'a str, files: &'a Files) -> impl Fn(&str) -> Option<String> + 'a {
    move |reference: &str| files.get(&resolve_path(path, reference, files)).cloned()
}

fn gauge(
    path: &str,
    text: &str,
    files: &Files,
    values: &Values,
) -> Result<(SourceChange, String), EditFailure> {
    let gauges = match text_value(values, "gauges") {
        "file" => {
            let file = text_value(values, "file").trim();
            if file.is_empty() {
                return Err(model(
                    "name the gauge table file, such as tables/steel-gauges.csv",
                ));
            }
            file.to_owned()
        }
        "" => "steel".to_owned(),
        builtin => builtin.to_owned(),
    };
    let gauge = text_value(values, "gauge").trim().to_owned();
    if gauge.is_empty() {
        let change = set_defaults(path, text, |d| {
            d.gauge = None;
            d.gauges = None;
        })?;
        return Ok((
            change,
            "Sheet bodies use their own thickness and radius".into(),
        ));
    }
    let table = tables::gauge_table(&gauges, &reader(path, files)).map_err(model)?;
    let key = tables::gauge_key(&gauge);
    let row = table
        .rows
        .iter()
        .find(|r| tables::gauge_key(&r.gauge) == key)
        .ok_or_else(|| {
            model(format!(
                "the gauge table `{gauges}` has no gauge `{gauge}`; it has {}",
                table
                    .rows
                    .iter()
                    .map(|r| r.gauge.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        })?;
    let written = row.gauge.clone();
    let summary = format!(
        "Gauge {written} ({gauges}): {} mm thick, {} mm bend radius, K {}",
        runmat_cad::assembly::settings::significant(row.thickness * 1000.0, 4),
        runmat_cad::assembly::settings::significant(row.bend_radius * 1000.0, 4),
        runmat_cad::assembly::settings::significant(row.k_factor, 3)
    );
    let change = set_defaults(path, text, |d| {
        d.gauges = Some(gauges.clone());
        d.gauge = Some(written.clone());
    })?;
    Ok((change, summary))
}

fn bend_table(
    path: &str,
    text: &str,
    files: &Files,
    values: &Values,
) -> Result<(SourceChange, String), EditFailure> {
    match text_value(values, "rule") {
        "k_factor" | "" => {
            let k = number(values, "k")?.ok_or_else(|| model("enter the K-factor"))?;
            if !(0.0..=1.0).contains(&k) {
                return Err(model(format!("a K-factor is between 0 and 1, not {k}")));
            }
            let change = set_defaults(path, text, |d| {
                d.bend_table = None;
                d.k = Some(k);
            })?;
            Ok((change, format!("Every bend uses K-factor {k}")))
        }
        "din6935" => {
            let change = set_defaults(path, text, |d| {
                d.bend_table = Some(tables::DIN6935.into());
                d.k = None;
            })?;
            Ok((change, "Every bend uses DIN 6935's K-factor".into()))
        }
        "table" => {
            let file = text_value(values, "table").trim().to_owned();
            if file.is_empty() {
                return Err(model(
                    "name the bend table file, such as tables/aluminum-bends.csv",
                ));
            }
            let rule = tables::bend_rule(&file, &reader(path, files)).map_err(model)?;
            let rows = match &rule {
                runmat_cad::model::sheet_metal::BendRule::Table { table } => table.rows.len(),
                _ => 0,
            };
            let change = set_defaults(path, text, |d| {
                d.bend_table = Some(file.clone());
                d.k = None;
            })?;
            Ok((change, format!("Every bend reads {file} ({rows} rows)")))
        }
        "feature" => {
            let change = set_defaults(path, text, |d| {
                d.bend_table = None;
                d.k = None;
            })?;
            Ok((change, "Each sheet feature uses its own bend rule".into()))
        }
        other => Err(model(format!(
            "`{other}` isn't a bend rule: choose k_factor, din6935, table or feature"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::edit_json;
    use crate::runtime::Files;
    use serde_json::{json, Value};

    const SHEET: &str = "runmat 0.7\nstd 1\n\npart bracket {\n    sketch s1 on origin.zx {\n        up = line(start = [0 mm, 30 mm], end = [0 mm, 0 mm])\n        base = line(start = [0 mm, 0 mm], end = [50 mm, 0 mm])\n    }\n    feature base = sheet_metal.base_flange(\n        profile = sketch.s1,\n        depth = 40 mm,\n        rules = { thickness = 2 mm, bend_radius = 2 mm, rule = k_factor(k = 0.44) }\n    )\n}\n";

    fn run(text: &str, files: &Files, command: &str, values: Value) -> Value {
        let request = json!({ "command": command, "values": values });
        serde_json::from_str(&edit_json(
            "/p/bracket.part",
            text,
            files,
            &request.to_string(),
        ))
        .unwrap()
    }

    fn applied(out: &Value) -> String {
        assert_eq!(out["ok"], true, "{out}");
        out["files"][0]["text"].as_str().unwrap().to_owned()
    }

    fn compiled(text: &str, files: &Files) -> Value {
        let mut all = files.clone();
        all.insert("/p/bracket.part".into(), text.into());
        let host = crate::runtime::values_host::ValueHost {
            files: all,
            runner: Box::new(
                runmat_cad::value::m_table::TableRunner::new(
                    std::collections::BTreeMap::new(),
                    "t",
                )
                .0,
            ),
        };
        crate::runtime::values_host::with_value_host(host, || {
            serde_json::from_str(&crate::runtime::compile_json(
                "/p/bracket.part",
                text,
                files,
            ))
            .unwrap()
        })
    }

    /// The compiled part's diagnostic messages.
    fn messages(c: &Value) -> Vec<String> {
        c["diagnostics"]
            .as_array()
            .map(|d| {
                d.iter()
                    .map(|d| d["message"].as_str().unwrap_or("").to_owned())
                    .collect()
            })
            .unwrap_or_default()
    }

    #[test]
    fn document_settings_write_one_line_and_refuse_a_repeat() {
        let files = Files::new();
        let out = run(
            SHEET,
            &files,
            "manage.document-settings",
            json!({ "units": "ips", "decimals": 3, "standard": "asme", "tolerance": "medium" }),
        );
        let text = applied(&out);
        assert!(
            text.contains(
                "settings = { units = ips, standard = asme, tolerance = medium, decimals = 3 }"
            ),
            "{text}"
        );
        assert_eq!(
            out["summary"],
            "Units ips, 3 decimals, ASME Y14.5, ISO 2768-m"
        );
        let again = run(
            &text,
            &files,
            "manage.document-settings",
            json!({ "units": "ips", "decimals": 3, "standard": "asme", "tolerance": "medium" }),
        );
        assert_eq!(again["ok"], false);
        assert_eq!(again["message"], "the part already has these settings");
        // Mass properties read in the part's units and decimals.
        let readout = crate::analyze::document_readout("/p/bracket.part", &text, &files).unwrap();
        assert_eq!(readout.mass(0.0785), "0.173 lb");
    }

    #[test]
    fn an_asme_part_reports_a_withdrawn_characteristic() {
        let files = Files::new();
        let text = SHEET.replace(
            "    feature base",
            "    tolerance conc = { characteristic = concentricity, on = [feature.base.body], tolerance = 0.05 mm }\n    feature base",
        );
        let iso = compiled(&text, &files);
        assert!(
            !messages(&iso).iter().any(|m| m.contains("withdrew")),
            "{iso}"
        );
        let asme = applied(&run(
            &text,
            &files,
            "manage.document-settings",
            json!({ "units": "mmgs", "standard": "asme", "tolerance": "none" }),
        ));
        let c = compiled(&asme, &files);
        assert!(
            messages(&c).iter().any(|m| m.contains(
                "tolerance `conc`: ASME Y14.5-2018 withdrew the concentricity tolerance"
            )),
            "{:?}",
            messages(&c)
        );
    }

    #[test]
    fn a_gauge_sets_every_sheet_body_and_a_table_file_rebuilds_it() {
        let mut files = Files::new();
        let out = run(
            SHEET,
            &files,
            "sheet-metal.gauge",
            json!({ "gauges": "steel", "gauge": "16 ga" }),
        );
        let text = applied(&out);
        assert!(
            text.contains("sheet_metal = { gauges = steel, gauge = \"16\" }"),
            "{text}"
        );
        assert_eq!(
            out["summary"],
            "Gauge 16 (steel): 1.519 mm thick, 1.519 mm bend radius, K 0.325"
        );
        // A gauge the table lacks names the ones it has.
        let missing = run(
            SHEET,
            &files,
            "sheet-metal.gauge",
            json!({ "gauges": "aluminum", "gauge": "30" }),
        );
        assert_eq!(missing["ok"], false);
        assert!(
            missing["message"]
                .as_str()
                .unwrap()
                .contains("has no gauge `30`; it has 8, 9, 10"),
            "{missing}"
        );
        // A project table: a file that isn't there is refused; once written it is read.
        let refused = run(
            SHEET,
            &files,
            "sheet-metal.gauge",
            json!({ "gauges": "file", "file": "tables/g.csv", "gauge": "A" }),
        );
        assert_eq!(refused["ok"], false, "{refused}");
        files.insert(
            "/p/tables/g.csv".into(),
            "gauge,thickness_mm,bend_radius_mm,k_factor\nA,3,4,0.4\n".into(),
        );
        let text = applied(&run(
            SHEET,
            &files,
            "sheet-metal.gauge",
            json!({ "gauges": "file", "file": "tables/g.csv", "gauge": "A" }),
        ));
        let c = compiled(&text, &files);
        assert!(messages(&c).is_empty(), "{:?}", messages(&c));
        // The table file is a dependency: editing it changes what the part reads.
        assert!(crate::runtime::dependencies("/p/bracket.part", &text)
            .contains(&"/p/tables/g.csv".to_owned()));
        files.insert(
            "/p/tables/g.csv".into(),
            "gauge,thickness_mm,bend_radius_mm,k_factor\nB,3,4,0.4\n".into(),
        );
        let c = compiled(&text, &files);
        assert!(
            messages(&c).iter().any(|m| m.contains("has no gauge `A`")),
            "{:?}",
            messages(&c)
        );
        // An empty gauge clears the part's gauge.
        let cleared = applied(&run(
            &text,
            &files,
            "sheet-metal.gauge",
            json!({ "gauges": "steel", "gauge": "" }),
        ));
        assert!(!cleared.contains("sheet_metal ="), "{cleared}");
    }

    #[test]
    fn the_bend_table_sets_the_rule_every_bend_follows() {
        let mut files = Files::new();
        let din = applied(&run(
            SHEET,
            &files,
            "sheet-metal.bend-table",
            json!({ "rule": "din6935" }),
        ));
        assert!(
            din.contains("sheet_metal = { bend_table = din6935 }"),
            "{din}"
        );
        let k = applied(&run(
            &din,
            &files,
            "sheet-metal.bend-table",
            json!({ "rule": "k_factor", "k": 0.38 }),
        ));
        assert!(k.contains("sheet_metal = { k = 0.38 }"), "{k}");
        let bad = run(
            SHEET,
            &files,
            "sheet-metal.bend-table",
            json!({ "rule": "k_factor", "k": 1.5 }),
        );
        assert_eq!(bad["message"], "a K-factor is between 0 and 1, not 1.5");
        files.insert(
            "/p/tables/al.csv".into(),
            "thickness_mm,radius_mm,angle_deg\n".into(),
        );
        let wrong = run(
            SHEET,
            &files,
            "sheet-metal.bend-table",
            json!({ "rule": "table", "table": "tables/al.csv" }),
        );
        assert!(
            wrong["message"].as_str().unwrap().contains("first line is"),
            "{wrong}"
        );
        files.insert(
            "/p/tables/al.csv".into(),
            "thickness_mm,radius_mm,angle_deg,allowance_mm\n2,2,90,4.5\n".into(),
        );
        let out = run(
            SHEET,
            &files,
            "sheet-metal.bend-table",
            json!({ "rule": "table", "table": "tables/al.csv" }),
        );
        assert_eq!(out["summary"], "Every bend reads tables/al.csv (1 rows)");
        let table = applied(&out);
        assert!(messages(&compiled(&table, &files)).is_empty());
        // The folded body needs no allowance; the flat pattern develops each bend from the
        // table, and a table without a row for the sheet's thickness and radius says so.
        files.insert(
            "/p/tables/al.csv".into(),
            "thickness_mm,radius_mm,angle_deg,allowance_mm\n3,3,90,6\n".into(),
        );
        let doc = runmat_cad::syntax::parse_part("/p/bracket.part", &table)
            .document
            .unwrap();
        let read = |r: &str| files.get(&format!("/p/{r}")).cloned();
        let (resolved, problems) = runmat_cad::model::sheet_metal::tables::resolve(&doc, &read);
        assert!(problems.is_empty(), "{problems:?}");
        let Some(runmat_cad::model::FeatureKind::SheetMetal(
            runmat_cad::model::sheet_metal::SheetMetalFeature::BaseFlange { rules, .. },
        )) = resolved.features.iter().map(|f| f.kind.clone()).next_back()
        else {
            panic!("no base flange")
        };
        let runmat_cad::model::sheet_metal::BendRule::Table { table: bends } = rules.rule else {
            panic!("the base flange doesn't read the table: {:?}", rules.rule)
        };
        let rule = runmat_cad::model::sheet_metal::BendRuleValue::Table { table: bends };
        let e = runmat_cad::model::sheet_metal::bend_allowance(
            &rule,
            0.002,
            0.002,
            std::f64::consts::FRAC_PI_2,
        )
        .unwrap_err();
        assert!(e.contains("has no row for thickness"), "{e}");
        let back = applied(&run(
            &table,
            &files,
            "sheet-metal.bend-table",
            json!({ "rule": "feature" }),
        ));
        assert!(!back.contains("sheet_metal ="), "{back}");
    }
}
