//! Three-way merge of drawings by stable ids.
//!
//! Formats, sheets, views and tables merge by name; annotations by their stable id. Each item
//! merges field by field (its top-level fields: an annotation's `view`, `kind`, `at`), so one
//! side moving a dimension while the other re-picks its references merges cleanly. The rules:
//!
//! - changed on one side → that side;
//! - changed on both, the same way → taken once;
//! - changed on both, differently, in the same field → **both changed** conflict (ours kept);
//! - deleted on one side, modified on the other → **delete vs modify** conflict (the modified
//!   item is kept, so no work is lost silently);
//! - added on both under one id with different content → **both added** conflict (ours kept).
//!
//! Order follows ours, with their additions placed after their nearest preceding item.
//! After the merge, references re-resolve: structurally (annotations on views, views on sheets
//! and parent views, tables on views, balloons on BOM tables that still exist) and, given the
//! model, against it: an annotation whose entities both sides agreed on but the merge resolves
//! differently is **redirected**, one that a side resolved and the merge does not is
//! **broken** (`cad.merge.redirected`, `cad.merge.broken`), exactly as part merges report.

use crate::diagnostics::{Cause, DiagCode, DiagnosticReport, Severity, Site, Subject};
use crate::drawing::{check_drawing, AnnotationKind, DrawingDocument, ModelReferences, TableKind};
use crate::refs::name::EntityName;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DrawingCollection {
    Header,
    Format,
    Sheet,
    View,
    Annotation,
    Table,
}

impl DrawingCollection {
    fn word(self) -> &'static str {
        match self {
            DrawingCollection::Header => "header",
            DrawingCollection::Format => "format",
            DrawingCollection::Sheet => "sheet",
            DrawingCollection::View => "view",
            DrawingCollection::Annotation => "annotation",
            DrawingCollection::Table => "table",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MergeSide {
    Ours,
    Theirs,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "conflict")]
pub enum DrawingConflictKind {
    BothChanged { fields: Vec<String> },
    DeleteModify { deleted_by: MergeSide },
    BothAdded { fields: Vec<String> },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DrawingConflict {
    pub collection: DrawingCollection,
    pub id: String,
    pub kind: DrawingConflictKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ours: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub theirs: Option<Value>,
}

#[derive(Clone, Debug)]
pub struct DrawingMerge {
    pub merged: DrawingDocument,
    pub conflicts: Vec<DrawingConflict>,
    /// Conflicts (warnings) and re-resolution results.
    pub diagnostics: DiagnosticReport,
}

fn to_value<T: Serialize>(t: &T) -> Value {
    serde_json::to_value(t).expect("drawing items serialise")
}

/// Field-wise three-way merge of two JSON objects; `Err` lists the conflicting fields (the
/// merged value then keeps ours in those fields).
fn merge_fields(base: Option<&Value>, ours: &Value, theirs: &Value) -> (Value, Vec<String>) {
    let empty = Map::new();
    let b = base.and_then(|v| v.as_object()).unwrap_or(&empty);
    let (Some(o), Some(t)) = (ours.as_object(), theirs.as_object()) else {
        return if ours == theirs {
            (ours.clone(), Vec::new())
        } else {
            (ours.clone(), vec!["value".into()])
        };
    };
    let keys: BTreeSet<&String> = b.keys().chain(o.keys()).chain(t.keys()).collect();
    let mut out = Map::new();
    let mut conflicts = Vec::new();
    // Keep ours' field order where it has the field, then the rest.
    let mut ordered: Vec<&String> = o.keys().collect();
    ordered.extend(keys.iter().filter(|k| !o.contains_key(**k)).copied());
    for k in ordered {
        let (bv, ov, tv) = (b.get(k), o.get(k), t.get(k));
        let pick = if ov == tv {
            ov
        } else if ov == bv {
            tv
        } else if tv == bv {
            ov
        } else {
            conflicts.push(k.clone());
            ov
        };
        if let Some(v) = pick {
            out.insert(k.clone(), v.clone());
        }
    }
    (Value::Object(out), conflicts)
}

fn merge_collection<T: Serialize + DeserializeOwned + Clone>(
    collection: DrawingCollection,
    base: &[T],
    ours: &[T],
    theirs: &[T],
    id: impl Fn(&T) -> String,
    conflicts: &mut Vec<DrawingConflict>,
) -> Vec<T> {
    let index =
        |v: &[T]| -> BTreeMap<String, Value> { v.iter().map(|x| (id(x), to_value(x))).collect() };
    let (b, o, t) = (index(base), index(ours), index(theirs));
    let mut result: Vec<(String, Value)> = Vec::new();
    let mut seen = BTreeSet::new();
    let decide = |key: &String, conflicts: &mut Vec<DrawingConflict>| -> Option<Value> {
        let (bv, ov, tv) = (b.get(key), o.get(key), t.get(key));
        let conflict = |kind: DrawingConflictKind| DrawingConflict {
            collection,
            id: key.clone(),
            kind,
            base: bv.cloned(),
            ours: ov.cloned(),
            theirs: tv.cloned(),
        };
        match (bv, ov, tv) {
            (_, Some(o), Some(t)) => {
                let (v, fields) = merge_fields(bv, o, t);
                if !fields.is_empty() {
                    conflicts.push(conflict(if bv.is_some() {
                        DrawingConflictKind::BothChanged { fields }
                    } else {
                        DrawingConflictKind::BothAdded { fields }
                    }));
                }
                Some(v)
            }
            (Some(b), Some(o), None) => {
                if o == b {
                    None
                } else {
                    conflicts.push(conflict(DrawingConflictKind::DeleteModify {
                        deleted_by: MergeSide::Theirs,
                    }));
                    Some(o.clone())
                }
            }
            (Some(b), None, Some(t)) => {
                if t == b {
                    None
                } else {
                    conflicts.push(conflict(DrawingConflictKind::DeleteModify {
                        deleted_by: MergeSide::Ours,
                    }));
                    Some(t.clone())
                }
            }
            (None, Some(o), None) => Some(o.clone()),
            (None, None, Some(t)) => Some(t.clone()),
            (_, None, None) => None,
        }
    };
    for x in ours {
        let k = id(x);
        if seen.insert(k.clone()) {
            if let Some(v) = decide(&k, conflicts) {
                result.push((k, v));
            }
        }
    }
    // Theirs' items not in ours: after their nearest preceding item that is in the result.
    let mut prev: Option<String> = None;
    for x in theirs {
        let k = id(x);
        if seen.insert(k.clone()) {
            if let Some(v) = decide(&k, conflicts) {
                let at = prev
                    .as_ref()
                    .and_then(|p| result.iter().position(|(rk, _)| rk == p))
                    .map(|i| i + 1)
                    .unwrap_or(if prev.is_none() { 0 } else { result.len() });
                result.insert(at, (k.clone(), v));
            }
        }
        if result.iter().any(|(rk, _)| *rk == k) {
            prev = Some(k);
        }
    }
    // Items only in the base were deleted on both sides.
    result
        .into_iter()
        .map(|(_, v)| serde_json::from_value(v).expect("a merge of valid items is valid"))
        .collect()
}

fn msite(collection: DrawingCollection, id: &str, param: &str) -> Site {
    Site::with_param(
        Subject::Value {
            name: format!("{}.{}", collection.word(), id),
        },
        param,
    )
}

/// Merge `ours` and `theirs` against their common `base`. With a model, references
/// re-resolve against it and changes in what they name are reported.
pub fn merge_drawings(
    base: &DrawingDocument,
    ours: &DrawingDocument,
    theirs: &DrawingDocument,
    model: Option<&dyn ModelReferences>,
) -> DrawingMerge {
    let mut conflicts = Vec::new();
    let (header, hf) = merge_fields(
        Some(&to_value(&base.header)),
        &to_value(&ours.header),
        &to_value(&theirs.header),
    );
    if !hf.is_empty() {
        conflicts.push(DrawingConflict {
            collection: DrawingCollection::Header,
            id: "header".into(),
            kind: DrawingConflictKind::BothChanged { fields: hf },
            base: Some(to_value(&base.header)),
            ours: Some(to_value(&ours.header)),
            theirs: Some(to_value(&theirs.header)),
        });
    }
    let formats = merge_collection(
        DrawingCollection::Format,
        &base.formats,
        &ours.formats,
        &theirs.formats,
        |f| f.name.clone(),
        &mut conflicts,
    );
    let sheets = merge_collection(
        DrawingCollection::Sheet,
        &base.sheets,
        &ours.sheets,
        &theirs.sheets,
        |s| s.name.clone(),
        &mut conflicts,
    );
    let views = merge_collection(
        DrawingCollection::View,
        &base.views,
        &ours.views,
        &theirs.views,
        |v| v.name.clone(),
        &mut conflicts,
    );
    let annotations = merge_collection(
        DrawingCollection::Annotation,
        &base.annotations,
        &ours.annotations,
        &theirs.annotations,
        |a| a.id.clone(),
        &mut conflicts,
    );
    let tables = merge_collection(
        DrawingCollection::Table,
        &base.tables,
        &ours.tables,
        &theirs.tables,
        |t| t.name.clone(),
        &mut conflicts,
    );
    let merged = DrawingDocument {
        path: ours.path.clone(),
        header: serde_json::from_value(header).expect("a merged header is valid"),
        formats,
        sheets,
        views,
        annotations,
        tables,
    };
    let mut diagnostics = DiagnosticReport::default();
    for c in &conflicts {
        let what = match &c.kind {
            DrawingConflictKind::BothChanged { fields } => format!(
                "both sides changed {} of {} `{}`; ours is kept",
                fields.join(", "),
                c.collection.word(),
                c.id
            ),
            DrawingConflictKind::BothAdded { fields } => format!(
                "both sides added {} `{}` with different {}; ours is kept",
                c.collection.word(),
                c.id,
                fields.join(", ")
            ),
            DrawingConflictKind::DeleteModify { deleted_by } => format!(
                "{} deleted {} `{}`, which {} changed; the changed {} is kept",
                match deleted_by {
                    MergeSide::Ours => "we",
                    MergeSide::Theirs => "they",
                },
                c.collection.word(),
                c.id,
                match deleted_by {
                    MergeSide::Ours => "they",
                    MergeSide::Theirs => "we",
                },
                c.collection.word()
            ),
        };
        diagnostics.push(
            DiagCode::DrawingMergeConflict,
            Severity::Warning,
            msite(c.collection, &c.id, "merge"),
            what,
            Cause::Root,
            Vec::new(),
        );
    }
    structural(&merged, &mut diagnostics);
    if let Some(model) = model {
        reresolve(ours, theirs, &merged, model, &mut diagnostics);
    }
    DrawingMerge {
        merged,
        conflicts,
        diagnostics,
    }
}

/// Cross-references inside the merged drawing that a merge can break.
fn structural(d: &DrawingDocument, rep: &mut DiagnosticReport) {
    let sheets: BTreeSet<&str> = d.sheets.iter().map(|s| s.name.as_str()).collect();
    let views: BTreeSet<&str> = d.views.iter().map(|v| v.name.as_str()).collect();
    let formats: BTreeSet<&str> = d.formats.iter().map(|f| f.name.as_str()).collect();
    let boms: BTreeSet<&str> = d
        .tables
        .iter()
        .filter(|t| matches!(t.kind, TableKind::Bom { .. }))
        .map(|t| t.name.as_str())
        .collect();
    let mut broken = |c: DrawingCollection, id: &str, param: &str, m: String| {
        rep.push(
            DiagCode::MergeBroken,
            Severity::Error,
            msite(c, id, param),
            m,
            Cause::Root,
            Vec::new(),
        );
    };
    for s in &d.sheets {
        if !formats.contains(s.format.as_str()) {
            broken(
                DrawingCollection::Sheet,
                &s.name,
                "format",
                format!(
                    "after the merge, sheet `{}` uses format `{}`, which was deleted",
                    s.name, s.format
                ),
            );
        }
    }
    for v in &d.views {
        if !sheets.contains(v.sheet.as_str()) {
            broken(
                DrawingCollection::View,
                &v.name,
                "sheet",
                format!(
                    "after the merge, view `{}` is on sheet `{}`, which was deleted",
                    v.name, v.sheet
                ),
            );
        }
        if let Some(p) = v.kind.parent() {
            if !views.contains(p) {
                broken(
                    DrawingCollection::View,
                    &v.name,
                    "from",
                    format!(
                        "after the merge, view `{}` derives from view `{}`, which was deleted",
                        v.name, p
                    ),
                );
            }
        }
    }
    for a in &d.annotations {
        if !views.contains(a.view.as_str()) {
            broken(
                DrawingCollection::Annotation,
                &a.id,
                "view",
                format!(
                    "after the merge, annotation `{}` is on view `{}`, which was deleted",
                    a.id, a.view
                ),
            );
        }
        if let AnnotationKind::Balloon { table, .. } = &a.kind {
            if !boms.contains(table.as_str()) {
                broken(
                    DrawingCollection::Annotation,
                    &a.id,
                    "table",
                    format!(
                        "after the merge, balloon `{}` numbers by BOM table `{}`, which was deleted",
                        a.id, table
                    ),
                );
            }
        }
    }
    for t in &d.tables {
        if !sheets.contains(t.sheet.as_str()) {
            broken(
                DrawingCollection::Table,
                &t.name,
                "sheet",
                format!(
                    "after the merge, table `{}` is on sheet `{}`, which was deleted",
                    t.name, t.sheet
                ),
            );
        }
        let view = match &t.kind {
            TableKind::Bom { view, .. }
            | TableKind::Hole { view, .. }
            | TableKind::Bend { view }
            | TableKind::Weld { view }
            | TableKind::CutList { view } => Some(view),
            TableKind::Revision { .. } | TableKind::Notes { .. } => None,
        };
        if let Some(v) = view {
            if !views.contains(v.as_str()) {
                broken(
                    DrawingCollection::Table,
                    &t.name,
                    "view",
                    format!(
                        "after the merge, table `{}` reads view `{}`, which was deleted",
                        t.name, v
                    ),
                );
            }
        }
    }
}

fn sorted(mut v: Vec<Vec<EntityName>>) -> Vec<Vec<EntityName>> {
    for e in &mut v {
        e.sort();
    }
    v.sort();
    v
}

/// Annotation references re-resolved against the model: redirected and broken.
fn reresolve(
    ours: &DrawingDocument,
    theirs: &DrawingDocument,
    merged: &DrawingDocument,
    model: &dyn ModelReferences,
    rep: &mut DiagnosticReport,
) {
    let o = check_drawing(ours, model).resolved;
    let t = check_drawing(theirs, model).resolved;
    let m = check_drawing(merged, model).resolved;
    for a in &merged.annotations {
        let (ov, tv) = (
            o.get(&a.id).cloned().map(sorted),
            t.get(&a.id).cloned().map(sorted),
        );
        let mv = m.get(&a.id).cloned().map(sorted);
        if ov.is_none() && tv.is_none() {
            continue;
        }
        let site = msite(DrawingCollection::Annotation, &a.id, "references");
        match &mv {
            None => {
                rep.push(
                    DiagCode::MergeBroken,
                    Severity::Error,
                    site,
                    format!(
                        "annotation `{}` resolved on {} but no longer resolves after the merge",
                        a.id,
                        match (&ov, &tv) {
                            (Some(_), Some(_)) => "both sides",
                            (Some(_), None) => "our side",
                            _ => "their side",
                        }
                    ),
                    Cause::Root,
                    Vec::new(),
                );
            }
            Some(mv) => {
                let agreed = match (&ov, &tv) {
                    (Some(x), Some(y)) if x == y => Some(x),
                    _ => None,
                };
                if let Some(x) = agreed {
                    if x != mv {
                        rep.push(
                            DiagCode::MergeRedirected,
                            Severity::Warning,
                            site,
                            format!(
                                "annotation `{}` now names different model entities than both sides did",
                                a.id
                            ),
                            Cause::Root,
                            Vec::new(),
                        );
                    }
                }
            }
        }
    }
}
