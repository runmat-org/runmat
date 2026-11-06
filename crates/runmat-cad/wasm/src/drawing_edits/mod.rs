//! Drawing commands from the sheet: views, annotations and tables placed by hand, and the
//! edits that move, align, relabel and delete them. Each is written as the smallest change to
//! the drawing's text (and, for tolerances, datums and finishes, to the drawn part's PMI, so
//! the part stays the one place a tolerance is declared).
//!
//! Sheet picks arrive as the Desktop names them:
//!
//! - `view:<name>`, `annotation:<id>`, `table:<name>`: Outline nodes and sheet items;
//! - `edge:<view>|<reference>`: a model edge drawn in a view, by its reference text
//!   (`edges_between(feature.body.face.side(s1.l4), feature.body.face.end)`);
//! - `point:<x>,<y>`: a point on the sheet, in sheet millimetres (y up).
//!
//! A bare reference text is taken as an edge of the command's view.

mod annotations;
mod items;
mod pmi;
mod sheets;
mod tables;
mod views;

#[cfg(test)]
mod tests;

use crate::edit::model;
use crate::runtime::{compile_part, module_scope, resolve_path, Files};
use runmat_cad::drawing::DrawingDocument;
use runmat_cad::refs::reference::Reference;
use runmat_cad::syntax::edit::{diff_source_in, EditFailure, SourceChange};
use runmat_cad::syntax::{parse_drawing_in, DRAWING};
use serde_json::{Map, Value};

pub type Values = Map<String, Value>;

/// One file a drawing command changes: its path, its text before, and the change.
pub type FileEdit = (String, String, SourceChange);

/// A pick on the sheet.
#[derive(Clone, Debug, PartialEq)]
pub enum SheetPick {
    View(String),
    Annotation(String),
    Table(String),
    /// A model edge seen in a view, and its reference.
    Edge { view: Option<String>, reference: Reference, text: String },
    Point([f64; 2]),
}

pub fn parse_pick(pick: &str) -> Result<SheetPick, EditFailure> {
    if let Some(name) = pick.strip_prefix("view:") {
        return Ok(SheetPick::View(name.to_owned()));
    }
    if let Some(id) = pick.strip_prefix("annotation:") {
        return Ok(SheetPick::Annotation(id.to_owned()));
    }
    if let Some(name) = pick.strip_prefix("table:") {
        return Ok(SheetPick::Table(name.to_owned()));
    }
    if let Some(xy) = pick.strip_prefix("point:") {
        let (x, y) = xy.split_once(',').ok_or_else(|| model(format!("`{pick}` isn't a sheet point")))?;
        let (x, y) = (x.trim().parse::<f64>(), y.trim().parse::<f64>());
        return match (x, y) {
            (Ok(x), Ok(y)) => Ok(SheetPick::Point([x, y])),
            _ => Err(model(format!("`{pick}` isn't a sheet point"))),
        };
    }
    let (view, text) = match pick.strip_prefix("edge:").and_then(|r| r.split_once('|')) {
        Some((view, text)) => (Some(view.to_owned()), text),
        None => (None, pick),
    };
    let reference = reference(text)?;
    Ok(SheetPick::Edge { view, reference, text: text.to_owned() })
}

/// A reference text as the model reference it lowers to.
pub fn reference(text: &str) -> Result<Reference, EditFailure> {
    let value = crate::picks::lower_text(text)?;
    serde_json::from_value(value).map_err(|_| model(format!("`{text}` isn't a model reference")))
}

/// A pick field's picks.
pub fn picks(values: &Values, field: &str) -> Result<Vec<SheetPick>, EditFailure> {
    crate::features::picks_of(values, field).iter().map(|p| parse_pick(p)).collect()
}

/// A field's edge picks: their views and references.
pub fn edges(values: &Values, field: &str) -> Result<Vec<(Option<String>, Reference, String)>, EditFailure> {
    picks(values, field)?
        .into_iter()
        .map(|p| match p {
            SheetPick::Edge { view, reference, text } => Ok((view, reference, text)),
            _ => Err(model(format!("pick model edges for `{field}`"))),
        })
        .collect()
}

/// A field's one view pick.
pub fn view_pick(values: &Values, field: &str) -> Result<Option<String>, EditFailure> {
    Ok(picks(values, field)?.into_iter().find_map(|p| match p {
        SheetPick::View(v) => Some(v),
        SheetPick::Edge { view, .. } => view,
        _ => None,
    }))
}

/// A field's sheet point picks.
pub fn points(values: &Values, field: &str) -> Result<Vec<[f64; 2]>, EditFailure> {
    Ok(picks(values, field)?
        .into_iter()
        .filter_map(|p| match p {
            SheetPick::Point(p) => Some(p),
            _ => None,
        })
        .collect())
}

pub fn text<'a>(values: &'a Values, id: &str, default: &'a str) -> &'a str {
    values.get(id).and_then(Value::as_str).filter(|s| !s.trim().is_empty()).unwrap_or(default)
}

pub fn flag(values: &Values, id: &str, default: bool) -> bool {
    values.get(id).and_then(Value::as_bool).unwrap_or(default)
}

/// A number field in its unit (a number, or a number as text).
pub fn number(values: &Values, id: &str, default: f64) -> f64 {
    match values.get(id) {
        Some(Value::Number(n)) => n.as_f64().unwrap_or(default),
        Some(Value::String(s)) => s.trim().trim_end_matches("mm").trim().parse().unwrap_or(default),
        Some(Value::Object(o)) => o.get("value").and_then(Value::as_f64).unwrap_or(default),
        _ => default,
    }
}

/// `1:2` as `(1, 2)`; empty or `sheet` for none.
pub fn scale(word: &str) -> Result<Option<(u32, u32)>, EditFailure> {
    let word = word.trim();
    if word.is_empty() || word == "sheet" {
        return Ok(None);
    }
    let bad = || model(format!("`{word}` isn't a scale (write it as 1:2)"));
    let (a, b) = word.split_once(':').ok_or_else(bad)?;
    match (a.trim().parse::<u32>(), b.trim().parse::<u32>()) {
        (Ok(a), Ok(b)) if a > 0 && b > 0 => Ok(Some((a, b))),
        _ => Err(bad()),
    }
}

/// A name for a new item: `stem`, or `stem2`, `stem3` … when it is taken by a view,
/// annotation or table.
pub fn fresh(document: &DrawingDocument, stem: &str) -> String {
    let taken = |n: &str| {
        document.views.iter().any(|v| v.name == n)
            || document.annotations.iter().any(|a| a.id == n)
            || document.tables.iter().any(|t| t.name == n)
    };
    let stem: String = stem
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '_' })
        .collect();
    let stem = if stem.chars().next().is_some_and(|c| c.is_ascii_alphabetic()) { stem } else { format!("v_{stem}") };
    if !taken(&stem) {
        return stem;
    }
    (2..).map(|n| format!("{stem}{n}")).find(|n| !taken(n)).expect("a free name")
}

/// The open drawing and what it is read in.
pub struct Drawing<'a> {
    pub path: &'a str,
    pub text: &'a str,
    pub files: &'a Files,
    pub document: DrawingDocument,
    uses: Vec<runmat_cad::scope::UseDecl>,
    scope: runmat_cad::modules::ModuleScope,
}

impl<'a> Drawing<'a> {
    pub fn open(path: &'a str, text: &'a str, files: &'a Files) -> Result<Self, EditFailure> {
        let scope = module_scope(path, text, files);
        let parsed = parse_drawing_in(path, text, &scope);
        let document = parsed.document.ok_or_else(|| EditFailure::Invalid(parsed.diagnostics.clone()))?;
        Ok(Drawing { path, text, files, document, uses: parsed.uses, scope })
    }

    /// The change from the text to the edited document.
    pub fn change(&self) -> Result<SourceChange, EditFailure> {
        diff_source_in(&DRAWING, self.path, self.text, &self.document, &self.uses, &[], &self.scope)
    }

    pub fn view(&self, name: &str) -> Result<&runmat_cad::drawing::View, EditFailure> {
        self.document.view(name).ok_or_else(|| model(format!("the drawing has no view `{name}`")))
    }

    /// The first sheet's name.
    pub fn sheet(&self) -> Result<String, EditFailure> {
        self.document
            .sheets
            .first()
            .map(|s| s.name.clone())
            .ok_or_else(|| model("add a sheet first (Sheet format)"))
    }

    /// The compiled display of a view's model, as the drawing compile reads it.
    pub fn display(&self, view: &str) -> Option<Value> {
        let file = self.view(view).ok()?.model.file.as_str().to_owned();
        display_of(&file, self.files)
    }
}

/// A part file's display (`{ kind: "part", data }`).
pub fn display_of(file: &str, files: &Files) -> Option<Value> {
    if !file.ends_with(".part") {
        return None;
    }
    let source = files.get(file)?;
    let part = compile_part(file, source).ok()?;
    Some(serde_json::json!({ "kind": "part", "data": part.display }))
}

/// The model a new view shows: the `model` field resolved from the drawing, else what the
/// drawing's first view shows.
pub fn model_ref(d: &Drawing, values: &Values) -> Result<runmat_cad::drawing::ModelRef, EditFailure> {
    let word = text(values, "model", "");
    if word.is_empty() {
        return d
            .document
            .views
            .first()
            .map(|v| v.model.clone())
            .ok_or_else(|| model("name the model to draw (a .part or .assembly file)"));
    }
    let file = resolve_path(d.path, word, d.files);
    if !d.files.contains_key(&file) {
        return Err(model(format!("{word} isn't in the workspace")));
    }
    if !(file.ends_with(".part") || file.ends_with(".assembly")) {
        return Err(model(format!("{word} isn't a part or an assembly")));
    }
    let configuration = text(values, "configuration", "");
    Ok(runmat_cad::drawing::ModelRef {
        file: runmat_cad::ids::FilePath::new(file),
        configuration: (!configuration.is_empty() && configuration != "default")
            .then(|| runmat_cad::ids::ConfigurationName::new(configuration)),
        variant: Default::default(),
    })
}

/// Runs a drawing command on the drawing `path`; `None` when `command` isn't one.
pub fn drawing_edit(
    path: &str,
    text: &str,
    files: &Files,
    command: &str,
    values: &Values,
    selection: &[String],
) -> Option<Result<(Vec<FileEdit>, String), EditFailure>> {
    let one = |r: Result<(SourceChange, String), EditFailure>| r.map(|(c, s)| (vec![(path.to_owned(), text.to_owned(), c)], s));
    if let Some(r) = views::edit(path, text, files, command, values) {
        return Some(one(r));
    }
    if let Some(r) = tables::edit(path, text, files, command, values) {
        return Some(one(r));
    }
    if let Some(r) = items::edit(path, text, files, command, values, selection) {
        return Some(one(r));
    }
    if let Some(r) = sheets::edit(path, text, files, command, values) {
        return Some(one(r));
    }
    annotations::edit(path, text, files, command, values)
}
