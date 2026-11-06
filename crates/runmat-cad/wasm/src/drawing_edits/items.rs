//! Edits to the views, annotations and tables selected on the sheet (`view:<name>`,
//! `annotation:<id>`, `table:<name>`):
//!
//! - `drawing.delete` removes them, and with a view whatever is drawn from it (its projected,
//!   section and detail views, its annotations and the tables that list it);
//! - `drawing.move` puts the first at the `to` point pick (the others keep their offsets from
//!   it), or moves each by `dx`, `dy`;
//! - `drawing.align` lines them up on the first one's x (vertical) or y (horizontal);
//! - `drawing.view-label` relabels a section, detail or auxiliary view.

use super::*;
use runmat_cad::drawing::{TableKind, ViewKind};

#[derive(Clone, Copy, PartialEq)]
enum Item<'s> {
    View(&'s str),
    Annotation(&'s str),
    Table(&'s str),
}

fn items(selection: &[String]) -> Vec<Item<'_>> {
    selection
        .iter()
        .filter_map(|s| {
            if let Some(v) = s.strip_prefix("view:") {
                Some(Item::View(v))
            } else if let Some(a) = s.strip_prefix("annotation:") {
                Some(Item::Annotation(a))
            } else {
                s.strip_prefix("table:").map(Item::Table)
            }
        })
        .collect()
}

fn at_mut<'d>(d: &'d mut Drawing<'_>, item: Item<'_>) -> Result<&'d mut [f64; 2], EditFailure> {
    let doc = &mut d.document;
    match item {
        Item::View(n) => doc.views.iter_mut().find(|v| v.name == n).map(|v| &mut v.at).ok_or_else(|| model(format!("the drawing has no view `{n}`"))),
        Item::Annotation(n) => doc.annotations.iter_mut().find(|a| a.id == n).map(|a| &mut a.at).ok_or_else(|| model(format!("the drawing has no annotation `{n}`"))),
        Item::Table(n) => doc.tables.iter_mut().find(|t| t.name == n).map(|t| &mut t.at).ok_or_else(|| model(format!("the drawing has no table `{n}`"))),
    }
}

fn name(item: Item<'_>) -> &str {
    match item {
        Item::View(n) | Item::Annotation(n) | Item::Table(n) => n,
    }
}

/// The view a table lists.
fn lists(k: &TableKind) -> Option<&str> {
    match k {
        TableKind::Bom { view, .. } | TableKind::Hole { view, .. } | TableKind::Bend { view } | TableKind::Weld { view } | TableKind::CutList { view } => Some(view.as_str()),
        TableKind::Revision { .. } => None,
    }
}

/// Deletes `views` and everything drawn from them, and the annotations and tables named.
fn delete(d: &mut Drawing, selected: &[Item]) -> Result<String, EditFailure> {
    let mut views: Vec<String> = Vec::new();
    for item in selected {
        match item {
            Item::View(v) => {
                d.view(v)?;
                views.push((*v).to_owned());
            }
            Item::Annotation(a) if !d.document.annotations.iter().any(|x| x.id == *a) => return Err(model(format!("the drawing has no annotation `{a}`"))),
            Item::Table(t) if !d.document.tables.iter().any(|x| x.name == *t) => return Err(model(format!("the drawing has no table `{t}`"))),
            _ => {}
        }
    }
    // Views drawn from a deleted view go with it.
    loop {
        let more: Vec<String> = d
            .document
            .views
            .iter()
            .filter(|v| !views.contains(&v.name) && v.kind.parent().is_some_and(|p| views.iter().any(|x| x == p)))
            .map(|v| v.name.clone())
            .collect();
        if more.is_empty() {
            break;
        }
        views.extend(more);
    }
    let before = (d.document.views.len(), d.document.annotations.len(), d.document.tables.len());
    d.document.views.retain(|v| !views.contains(&v.name));
    d.document.annotations.retain(|a| !views.contains(&a.view) && !selected.contains(&Item::Annotation(&a.id)));
    d.document.tables.retain(|t| !lists(&t.kind).is_some_and(|v| views.iter().any(|x| x == v)) && !selected.contains(&Item::Table(&t.name)));
    // A balloon numbered by a deleted BOM table goes too.
    let tables: Vec<String> = d.document.tables.iter().map(|t| t.name.clone()).collect();
    d.document.annotations.retain(|a| !matches!(&a.kind, runmat_cad::drawing::AnnotationKind::Balloon { table, .. } if !tables.contains(table)));
    let (v, a, t) = (before.0 - d.document.views.len(), before.1 - d.document.annotations.len(), before.2 - d.document.tables.len());
    let mut parts = Vec::new();
    for (n, what) in [(v, "view"), (a, "annotation"), (t, "table")] {
        if n > 0 {
            parts.push(format!("{n} {what}{}", if n == 1 { "" } else { "s" }));
        }
    }
    Ok(format!("Deleted {}", parts.join(", ")))
}

pub fn edit(path: &str, text_: &str, files: &Files, command: &str, values: &Values, selection: &[String]) -> Option<Result<(SourceChange, String), EditFailure>> {
    if !["drawing.delete", "drawing.move", "drawing.align", "drawing.view-label"].contains(&command) {
        return None;
    }
    Some((|| {
        let mut d = Drawing::open(path, text_, files)?;
        let selected = items(selection);
        if selected.is_empty() {
            return Err(model("select the views, annotations or tables first"));
        }
        let summary = match command {
            "drawing.delete" => delete(&mut d, &selected)?,
            "drawing.move" => {
                let to = points(values, "to")?.into_iter().next();
                let delta = match to {
                    Some(to) => {
                        let first = *at_mut(&mut d, selected[0])?;
                        [to[0] - first[0], to[1] - first[1]]
                    }
                    None => [number(values, "dx", 0.0), number(values, "dy", 0.0)],
                };
                if delta == [0.0, 0.0] {
                    return Err(model("pick where to move it, or give a distance"));
                }
                for item in &selected {
                    let at = at_mut(&mut d, *item)?;
                    // An annotation the sheet places itself has no place yet; it takes the
                    // place it is moved to.
                    let base = if *at == [0.0, 0.0] && matches!(item, Item::Annotation(_)) && to.is_some() { [0.0, 0.0] } else { *at };
                    let moved = [base[0] + delta[0], base[1] + delta[1]];
                    *at = moved.map(|v| (v * 10.0).round() / 10.0);
                }
                format!("Moved {}", selected.iter().map(|i| name(*i)).collect::<Vec<_>>().join(", "))
            }
            "drawing.align" => {
                if selected.len() < 2 {
                    return Err(model("select two or more items to align"));
                }
                let horizontal = text(values, "align", "horizontal") == "horizontal";
                let k = if horizontal { 1 } else { 0 };
                let first = *at_mut(&mut d, selected[0])?;
                for item in &selected[1..] {
                    at_mut(&mut d, *item)?[k] = first[k];
                }
                format!("Aligned {} {}", selected.iter().map(|i| name(*i)).collect::<Vec<_>>().join(", "), if horizontal { "horizontally" } else { "vertically" })
            }
            "drawing.view-label" => {
                let label = text(values, "label", "").trim().to_uppercase();
                if label.is_empty() {
                    return Err(model("write the view's label"));
                }
                let mut named = Vec::new();
                for item in &selected {
                    let Item::View(n) = item else { continue };
                    let view = d.document.views.iter_mut().find(|v| v.name == *n).ok_or_else(|| model(format!("the drawing has no view `{n}`")))?;
                    match &mut view.kind {
                        ViewKind::Section { label: l, .. } | ViewKind::Detail { label: l, .. } | ViewKind::Auxiliary { label: l, .. } => *l = label.clone(),
                        _ => return Err(model(format!("{n} has no label; sections, details and auxiliary views do"))),
                    }
                    named.push(*n);
                }
                if named.is_empty() {
                    return Err(model("select the views to label"));
                }
                format!("Labelled {} {label}", named.join(", "))
            }
            _ => unreachable!(),
        };
        Ok((d.change()?, summary))
    })())
}
