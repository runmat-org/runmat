//! A drawing compiled in stages, and its views' projections kept between compiles.
//!
//! Opening a drawing costs mostly hidden-line projection: each view projects its model's edges
//! (exact on the kernel for a part, from the display meshes for an assembly), a second or more
//! per view. The Desktop therefore compiles a drawing it hasn't shown yet in stages:
//!
//! 1. `{"drawing": {"stage": "sheet"}}`: the sheets, frames and title blocks only. No model is
//!    regenerated and no view projected, so the sheet paints at once. The projection says
//!    `"pending": true`, and each view draws as its label until its lines arrive.
//! 2. `{"drawing": {"views": ["front", "top"]}}`: those views projected, the rest pending. The
//!    Desktop adds one view per compile, so each view fills in when its projection is ready.
//! 3. No `drawing` option: the whole drawing, as before.
//!
//! Each view's projection is kept by [`view_key`] in the part cache (`runtime::part_cache`), so
//! an unchanged view is never projected twice on one host, and the page stores the entries it
//! makes (IndexedDB) for the next visit. The key holds everything the projection reads: the
//! model's input (a part's path, text and configuration; an assembly's path and every file the
//! compile was given), the view's definition and those of the views it is drawn from, the
//! sheet's scale and [`PROJECTION_VERSION`], but not where the view sits on the sheet. The page keeps entries per host build, so another
//! kernel or host never reads them.
//!
//! The parts the views show are regenerated once for the stages: the last few regenerated parts
//! are kept ([`regenerated_part`]) by path, text and configuration.

use crate::runtime::{regenerate_part_in, Diagnostic, Files, RegeneratedPart};
use serde_json::Value;
use std::cell::RefCell;
use std::collections::BTreeSet;
use std::rc::Rc;

/// Changes when a projection's output changes for the same input (the line format, thinning).
pub const PROJECTION_VERSION: &str = "view-lines-1";

/// Regenerated parts kept for the next stage's compile.
const KEPT_PARTS: usize = 4;

/// Which part of a drawing a compile makes.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum Stage {
    /// Everything.
    #[default]
    Full,
    /// The sheets only: no model regenerated, no view projected.
    Sheet,
    /// The named views projected, the others pending.
    Views(BTreeSet<String>),
}

impl Stage {
    /// The stage `options` (the compile's options JSON) asks for.
    pub fn from_options(options: &str) -> Stage {
        let value: Value = serde_json::from_str(options).unwrap_or_default();
        let drawing = &value["drawing"];
        if drawing["stage"] == "sheet" {
            return Stage::Sheet;
        }
        match drawing["views"].as_array() {
            Some(names) => Stage::Views(names.iter().filter_map(Value::as_str).map(str::to_owned).collect()),
            None => Stage::Full,
        }
    }

    /// Whether view `name` is projected in this stage.
    pub fn projects(&self, name: &str) -> bool {
        match self {
            Stage::Full => true,
            Stage::Sheet => false,
            Stage::Views(names) => names.contains(name),
        }
    }

    /// Whether some view is left pending.
    pub fn partial(&self) -> bool {
        *self != Stage::Full
    }
}

thread_local! {
    static STAGE: RefCell<Stage> = RefCell::new(Stage::Full);
    static PARTS: RefCell<Vec<(String, Rc<RegeneratedPart>)>> = const { RefCell::new(Vec::new()) };
}

/// Runs `run` (a drawing compile) in `stage`.
pub fn with_stage<T>(stage: Stage, run: impl FnOnce() -> T) -> T {
    let previous = STAGE.with(|s| std::mem::replace(&mut *s.borrow_mut(), stage));
    let out = run();
    STAGE.with(|s| *s.borrow_mut() = previous);
    out
}

/// The stage the compile in progress makes.
pub fn stage() -> Stage {
    STAGE.with(|s| s.borrow().clone())
}

/// The input a model's projections read: a part's path, text and configuration; an assembly's
/// path and every file the compile reads.
pub fn model_key(path: &str, text: &str, configuration: Option<&str>, files: Option<&Files>) -> String {
    let mut key = format!("{path}\0{text}\0{}", configuration.unwrap_or(""));
    for (file, content) in files.into_iter().flatten() {
        key.push('\0');
        key.push_str(file);
        key.push('\0');
        key.push_str(content);
    }
    key
}

/// The key of view `name`'s projection: [`model_key`], the view's definition and those of the
/// views it is drawn from (a section's and a detail's parent), the sheet's scale.
pub fn view_key(model: &str, views: &Value, name: &str, sheet_scale: f64) -> String {
    let mut chain = Vec::new();
    let mut at = name.to_owned();
    while let Some(view) = views.as_array().and_then(|vs| vs.iter().find(|v| v["name"] == at.as_str())) {
        if chain.len() > views.as_array().map_or(0, Vec::len) {
            break;
        }
        // Where the view sits on the sheet doesn't change its lines.
        let mut definition = view.clone();
        if let Some(o) = definition.as_object_mut() {
            o.remove("at");
        }
        chain.push(definition);
        match view["kind"]["from"].as_str() {
            Some(parent) => at = parent.to_owned(),
            None => break,
        }
    }
    let chain = serde_json::to_string(&chain).unwrap_or_default();
    format!("view\0{PROJECTION_VERSION}\0{:x}\0{chain}\0{model}", sheet_scale.to_bits())
}

/// Part `path` regenerated in `configuration`, from the parts kept for the stages when its text
/// is unchanged. A part that read a dynamic value isn't kept.
pub fn regenerated_part(path: &str, text: &str, configuration: Option<&str>) -> Result<Rc<RegeneratedPart>, Vec<Diagnostic>> {
    let key = model_key(path, text, configuration, None);
    if let Some(part) = PARTS.with(|p| p.borrow().iter().find(|(k, _)| *k == key).map(|(_, part)| Rc::clone(part))) {
        return Ok(part);
    }
    let part = Rc::new(regenerate_part_in(path, text, configuration, &Default::default())?);
    if !part.dynamic {
        PARTS.with(|p| {
            let mut p = p.borrow_mut();
            p.retain(|(k, _)| !k.starts_with(&format!("{path}\0")));
            p.insert(0, (key, Rc::clone(&part)));
            p.truncate(KEPT_PARTS);
        });
    }
    Ok(part)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn the_stage_comes_from_the_options() {
        assert_eq!(Stage::from_options(""), Stage::Full);
        assert_eq!(Stage::from_options(r#"{"configuration":"long"}"#), Stage::Full);
        assert_eq!(Stage::from_options(r#"{"drawing":{"stage":"sheet"}}"#), Stage::Sheet);
        let views = Stage::from_options(r#"{"drawing":{"views":["front"]}}"#);
        assert!(views.projects("front") && !views.projects("top") && views.partial());
        assert!(!Stage::Sheet.projects("front") && Stage::Full.projects("front") && !Stage::Full.partial());
    }

    const PART: &str = "runmat 0.7\nstd 1\n\npart bracket {\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [40 mm, 0 mm])\n        l2 = line(start = [40 mm, 0 mm], end = [40 mm, 20 mm])\n        l3 = line(start = [40 mm, 20 mm], end = [0 mm, 20 mm])\n        l4 = line(start = [0 mm, 20 mm], end = [0 mm, 0 mm])\n    }\n    feature body = extrude(profile = sketch.s1, distance = 10 mm)\n}\n";
    const DRAWING: &str = "runmat 0.7\nstd 1\n\ndrawing bracket {\n    format a3 = { size = { size = iso, name = A3 }, title_block = [] }\n    sheet \"1\" = { format = a3, scale = [1, 1], projection = third_angle }\n    view front = standard(orientation = front, sheet = \"1\", model = { part = bracket }, at = [100, 100], display = visible_lines)\n    view top = projected(from = front, direction = up, sheet = \"1\", model = { part = bracket }, at = [100, 200], display = visible_lines)\n}\n";

    fn compile(stage: Stage, text: &str) -> Value {
        let files: Files = [("/p/bracket.part".to_string(), PART.to_string())].into();
        let out = with_stage(stage, || crate::drawing::compile_json("/p/bracket.drawing", text, &files));
        serde_json::from_str(&out).unwrap()
    }

    fn drawn(out: &Value) -> Vec<String> {
        out["projection"]["data"]["hlr"].as_object().map(|o| o.keys().cloned().collect()).unwrap_or_default()
    }

    #[test]
    fn a_drawing_compiles_in_stages_and_keeps_each_view_it_projected() {
        crate::runtime::part_cache::clear();
        let sheet = compile(Stage::Sheet, DRAWING);
        assert_eq!(sheet["projection"]["data"]["pending"], true);
        assert!(drawn(&sheet).is_empty(), "the sheet stage projects nothing");
        assert!(sheet["projection"]["data"]["layouts"]["1"]["primitives"].as_array().is_some_and(|p| !p.is_empty()), "the sheet is laid out");
        assert!(sheet.get("models").is_none(), "a stage sends no models");

        let first = compile(Stage::Views(["front".to_string()].into()), DRAWING);
        assert_eq!(drawn(&first), ["front"]);
        assert_eq!(first["projection"]["data"]["pending"], true);

        // The full compile reuses the front view's projection and projects the top view.
        let before = crate::runtime::part_cache::stats();
        let full = compile(Stage::Full, DRAWING);
        let after = crate::runtime::part_cache::stats();
        assert_eq!(drawn(&full), ["front", "top"]);
        assert!(full["projection"]["data"].get("pending").is_none());
        assert!(full.get("models").is_some());
        assert_eq!(after.hits - before.hits, 1, "the front view comes from the cache");
        assert_eq!(full["projection"]["data"]["hlr"]["front"], first["projection"]["data"]["hlr"]["front"]);

        // A later stage brings every view the cache holds, and moving a view keeps its lines.
        let moved = DRAWING.replace("at = [100, 200]", "at = [120, 210]");
        let again = compile(Stage::Views(BTreeSet::new()), &moved);
        assert_eq!(drawn(&again), ["front", "top"]);
    }

    #[test]
    fn a_view_key_reads_its_parents_and_nothing_else() {
        let views = json!([
            { "name": "front", "kind": { "view": "standard", "orientation": "front" }, "at": [100, 100] },
            { "name": "top", "kind": { "view": "projected", "from": "front", "direction": "up" }, "at": [100, 200] },
            { "name": "iso", "kind": { "view": "standard", "orientation": "isometric" }, "at": [300, 200] }
        ]);
        let model = model_key("/p/a.part", "part a {}", None, None);
        let top = view_key(&model, &views, "top", 1.0);
        // Turning the parent changes the child's key; moving a view on the sheet, or changing
        // an unrelated view, doesn't.
        let mut turned = views.clone();
        turned[0]["kind"]["orientation"] = json!("back");
        assert_ne!(view_key(&model, &turned, "top", 1.0), top);
        let mut moved = views.clone();
        moved[0]["at"] = json!([120, 100]);
        moved[1]["at"] = json!([120, 200]);
        moved[2]["kind"]["orientation"] = json!("top");
        assert_eq!(view_key(&model, &moved, "top", 1.0), top);
        // The sheet's scale and the model's text are in it.
        assert_ne!(view_key(&model, &views, "top", 0.5), top);
        assert_ne!(view_key(&model_key("/p/a.part", "part a { }", None, None), &views, "top", 1.0), top);
        assert_ne!(view_key(&model_key("/p/a.part", "part a {}", Some("long"), None), &views, "top", 1.0), top);
    }

    #[test]
    fn a_view_digest_says_it_is_a_view() {
        use crate::runtime::part_cache::{assembly_key, digest, VIEW_DIGEST_PREFIX};
        let views = json!([{ "name": "front", "kind": { "view": "standard", "orientation": "front" } }]);
        let view = digest(&view_key(&model_key("/p/a.part", "part a {}", None, None), &views, "front", 1.0));
        assert!(view.starts_with(VIEW_DIGEST_PREFIX), "{view}");
        let part = digest(&assembly_key("/p/a.part", "part a {}", &Default::default()));
        assert!(!part.starts_with(VIEW_DIGEST_PREFIX), "{part}");
    }
}
