//! Tables placed by hand: a BOM from an assembly view, a hole table from an origin, the
//! revision table (each run adds a revision row), a flat pattern's bend table and a weldment's
//! cut list. A table lands at the `at` point pick, else in its standard place (a BOM on the
//! title block, the revision table top right).

use super::*;
use runmat_cad::drawing::bom_table::{BomStructure, BomTableSpec};
use runmat_cad::drawing::hole_table::{HoleTableSpec, HoleTagging};
use runmat_cad::drawing::revision_table::RevisionRow;
use runmat_cad::drawing::{DrawingTable, TableKind, ViewKind};

/// The feature a face reference names (`feature.h.face.wall(a)` → `h`).
fn feature_of(text: &str) -> Option<&str> {
    let rest = &text[text.find("feature.")? + "feature.".len()..];
    let end = rest
        .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .unwrap_or(rest.len());
    (end > 0).then(|| &rest[..end])
}

pub fn edit(
    path: &str,
    text_: &str,
    files: &Files,
    command: &str,
    values: &Values,
) -> Option<Result<(SourceChange, String), EditFailure>> {
    if ![
        "table.bom",
        "table.hole",
        "table.revision",
        "table.bend",
        "table.cut-list",
    ]
    .contains(&command)
    {
        return None;
    }
    Some((|| {
        let mut d = Drawing::open(path, text_, files)?;
        let at = points(values, "at")?
            .into_iter()
            .next()
            .unwrap_or([0.0, 0.0]);
        let picked_view = |d: &Drawing, field: &str| -> Result<String, EditFailure> {
            let v =
                view_pick(values, field)?.ok_or_else(|| model("pick the view the table lists"))?;
            d.view(&v)?;
            Ok(v)
        };
        let summary = match command {
            "table.bom" => {
                let view = picked_view(&d, "view")?;
                if !d.view(&view)?.model.file.as_str().ends_with(".assembly") {
                    return Err(model("a BOM lists an assembly; pick an assembly view"));
                }
                let structure = match text(values, "type", "top") {
                    "top" => BomStructure::TopLevel,
                    "parts" => BomStructure::PartsOnly,
                    "indented" => BomStructure::Indented,
                    other => return Err(model(format!("there is no `{other}` BOM type"))),
                };
                let sheet = d.view(&view)?.sheet.clone();
                let name = fresh(&d.document, "bom");
                let spec = BomTableSpec {
                    structure,
                    ..BomTableSpec::default()
                };
                let kind = TableKind::Bom {
                    view,
                    include_fasteners: flag(values, "fasteners", true),
                    spec,
                };
                d.document.tables.push(DrawingTable {
                    name: name.clone(),
                    sheet,
                    kind,
                    at,
                });
                format!("Added BOM table {name}")
            }
            "table.hole" => {
                let (view, origin, _) = edges(values, "origin")?
                    .into_iter()
                    .next()
                    .ok_or_else(|| model("pick the hole table's origin (an edge or a corner)"))?;
                let view = match view {
                    Some(v) => v,
                    None => picked_view(&d, "view")?,
                };
                let mut features: Vec<runmat_cad::ids::FeatureId> = Vec::new();
                for (_, _, text) in edges(values, "faces")? {
                    let f = feature_of(&text)
                        .ok_or_else(|| model(format!("`{text}` isn't a hole's face")))?;
                    let id = runmat_cad::ids::FeatureId(f.to_owned());
                    if !features.contains(&id) {
                        features.push(id);
                    }
                }
                let sheet = d.view(&view)?.sheet.clone();
                let name = fresh(&d.document, "holes");
                let spec = HoleTableSpec {
                    features,
                    tagging: if flag(values, "combine", true) {
                        HoleTagging::BySize
                    } else {
                        HoleTagging::Sequential
                    },
                    ..HoleTableSpec::default()
                };
                d.document.tables.push(DrawingTable {
                    name: name.clone(),
                    sheet,
                    kind: TableKind::Hole { view, origin, spec },
                    at,
                });
                format!("Added hole table {name}")
            }
            "table.revision" => {
                let row = RevisionRow {
                    rev: text(values, "rev", "A").trim().to_owned(),
                    description: text(values, "description", "Initial release")
                        .trim()
                        .to_owned(),
                    date: text(values, "date", "").trim().to_owned(),
                    by: text(values, "by", "").trim().to_owned(),
                };
                let existing = d
                    .document
                    .tables
                    .iter_mut()
                    .find_map(|t| match &mut t.kind {
                        TableKind::Revision { rows } => Some((t.name.clone(), rows)),
                        _ => None,
                    });
                match existing {
                    Some((name, rows)) => {
                        if rows.iter().any(|r| r.rev == row.rev) {
                            return Err(model(format!(
                                "revision {} is in the table already",
                                row.rev
                            )));
                        }
                        let rev = row.rev.clone();
                        rows.push(row);
                        format!("Added revision {rev} to {name}")
                    }
                    None => {
                        let sheet = d.sheet()?;
                        let name = fresh(&d.document, "revisions");
                        let rev = row.rev.clone();
                        d.document.tables.push(DrawingTable {
                            name: name.clone(),
                            sheet,
                            kind: TableKind::Revision { rows: vec![row] },
                            at,
                        });
                        format!("Added revision table {name} at revision {rev}")
                    }
                }
            }
            "table.bend" => {
                let view = picked_view(&d, "view")?;
                let v = d
                    .document
                    .views
                    .iter_mut()
                    .find(|v| v.name == view)
                    .expect("checked");
                match &mut v.kind {
                    ViewKind::FlatPattern { bend_table, .. } => {
                        if *bend_table {
                            return Err(model(format!("{view} shows its bend table already")));
                        }
                        *bend_table = true;
                    }
                    _ => {
                        return Err(model(
                            "a bend table lists a flat pattern; pick a flat-pattern view",
                        ))
                    }
                }
                format!("Added the bend table of {view}")
            }
            "table.cut-list" => {
                let view = picked_view(&d, "view")?;
                let sheet = d.view(&view)?.sheet.clone();
                let name = fresh(&d.document, "cut_list");
                d.document.tables.push(DrawingTable {
                    name: name.clone(),
                    sheet,
                    kind: TableKind::CutList { view },
                    at,
                });
                format!("Added cut list {name}")
            }
            _ => unreachable!(),
        };
        Ok((d.change()?, summary))
    })())
}
