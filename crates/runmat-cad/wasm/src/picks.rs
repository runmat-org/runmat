//! Canvas and Outline picks as the model values command fields take.
//!
//! A pick arrives as the text the Desktop holds for it:
//!
//! - a face, edge or vertex the canvas picked: its reference text
//!   (`feature.body.face.end`, `edges_between(…, …)`, `vertex_of(…)`);
//! - an Outline node: `feature:<id>`, `sketch:<id>`, `body:<index>`, `origin:top|front|right|point`;
//! - a sketch entity in the sketch canvas: `sketch.<sketch>.<entity>`.
//!
//! Each is lowered through the part grammar's own value forms ([`CadValues`]), so a pick becomes
//! exactly the model value its source text would.

use crate::runtime::compile_part;
use runmat_cad::model::{FeatureKind, PartDocument};
use runmat_cad::syntax::edit::EditFailure;
use runmat_cad::syntax::value::CadValues;
use runmat_design_syntax::ast::Statement;
use runmat_design_syntax::json::ValueSyntax;
use runmat_design_syntax::{parse_source, Dialect};
use serde_json::{json, Value};
use std::cell::OnceCell;

fn model(message: impl Into<String>) -> EditFailure {
    EditFailure::Model(message.into())
}

/// One expression's model value, as the part grammar lowers it.
pub fn lower_text(text: &str) -> Result<Value, EditFailure> {
    let source = format!("runmat 0.7\nstd 1\n\npart probe {{\n    let probe = {text}\n}}\n");
    let syntax = parse_source(Dialect::Part, source);
    if !syntax.diagnostics.is_empty() {
        return Err(model(format!("`{text}` isn't a reference")));
    }
    let expr = syntax
        .syntax()
        .descendants()
        .into_iter()
        .filter_map(Statement::cast)
        .find_map(|s| match s {
            Statement::Value(v) if v.keyword() == "let" => v.value(),
            _ => None,
        })
        .ok_or_else(|| model(format!("`{text}` isn't a reference")))?;
    match CadValues.lower(&expr) {
        Some(Ok(json)) => Ok(json.to_value()),
        Some(Err(e)) => Err(model(format!("`{text}`: {}", e.message))),
        None => Err(model(format!("`{text}` isn't a reference"))),
    }
}

/// What a part's picks resolve against: its document, and (when a pick names a body by its
/// index) its display.
pub struct Picks<'a> {
    pub document: &'a PartDocument,
    path: &'a str,
    text: &'a str,
    bodies: OnceCell<Vec<Value>>,
}

impl<'a> Picks<'a> {
    pub fn new(document: &'a PartDocument, path: &'a str, text: &'a str) -> Self {
        Self {
            document,
            path,
            text,
            bodies: OnceCell::new(),
        }
    }

    /// The regenerated part's display bodies (faces with their reference texts).
    pub fn bodies(&self) -> &[Value] {
        self.bodies.get_or_init(|| {
            compile_part(self.path, self.text)
                .ok()
                .and_then(|c| serde_json::to_value(&c.display).ok())
                .and_then(|d| d["bodies"].as_array().cloned())
                .unwrap_or_default()
        })
    }

    fn kind_of(&self, feature: &str) -> Option<&'static str> {
        let f = self.document.features.iter().find(|f| f.id.0 == feature)?;
        Some(match &f.kind {
            FeatureKind::Sketch(_) => "sketch",
            FeatureKind::Plane(_) => "plane",
            FeatureKind::Axis(_) => "axis",
            FeatureKind::Point(_) => "point",
            FeatureKind::Frame(_) => "frame",
            _ => "feature",
        })
    }

    /// A pick's source text.
    fn text_of(&self, pick: &str) -> Result<String, EditFailure> {
        if let Some(which) = pick.strip_prefix("origin:") {
            return Ok(match which {
                "top" => "origin.xy",
                "front" => "origin.zx",
                "right" => "origin.yz",
                "point" => "origin.point",
                "x" | "y" | "z" => return Ok(format!("origin.{which}")),
                other => return Err(model(format!("`origin:{other}` isn't a datum"))),
            }
            .to_owned());
        }
        if let Some(at) = pick.strip_prefix("body:") {
            let i: usize = at
                .parse()
                .map_err(|_| model(format!("`{pick}` isn't a body")))?;
            return self
                .bodies()
                .iter()
                .find(|b| b["index"].as_u64() == Some(i as u64))
                .and_then(|b| b["reference"].as_str())
                .map(str::to_owned)
                .ok_or_else(|| model(format!("body {i} has no name to refer to")));
        }
        if let Some(id) = pick
            .strip_prefix("feature:")
            .or_else(|| pick.strip_prefix("sketch:"))
        {
            return Ok(match self.kind_of(id) {
                Some("plane") => format!("plane.{id}"),
                Some("axis") => format!("axis.{id}"),
                Some("point") => format!("point.{id}"),
                Some("sketch") => format!("sketch.{id}"),
                _ => format!("feature.{id}.body"),
            });
        }
        Ok(pick.to_owned())
    }

    /// A face, edge, vertex or body reference.
    pub fn reference(&self, pick: &str) -> Result<Value, EditFailure> {
        let v = lower_text(&self.text_of(pick)?)?;
        if v.get("ref").is_some() {
            return Ok(v);
        }
        Err(model(format!(
            "`{pick}` isn't a face, edge, vertex or body"
        )))
    }

    pub fn references(&self, picks: &[String]) -> Result<Vec<Value>, EditFailure> {
        picks.iter().map(|p| self.reference(p)).collect()
    }

    /// A plane: a datum plane, or a planar face.
    pub fn plane(&self, pick: &str) -> Result<Value, EditFailure> {
        let v = lower_text(&self.text_of(pick)?)?;
        if v.get("on").is_some() {
            return Ok(v);
        }
        if v.get("ref").is_some() {
            return Ok(json!({ "on": "face", "face": v }));
        }
        Err(model(format!("`{pick}` isn't a plane or a face")))
    }

    /// An axis: a datum axis, or a linear edge.
    pub fn axis(&self, pick: &str) -> Result<Value, EditFailure> {
        let v = lower_text(&self.text_of(pick)?)?;
        if v.get("along").is_some() {
            return Ok(v);
        }
        if v.get("ref").is_some() {
            return Ok(json!({ "along": "edge", "edge": v }));
        }
        Err(model(format!("`{pick}` isn't an axis or an edge")))
    }

    /// A point: a datum point, a vertex, or a sketch point (a point entity, an arc's or circle's
    /// centre, a line's start).
    pub fn point(&self, pick: &str) -> Result<Value, EditFailure> {
        if let Some(rest) = pick.strip_prefix("sketch.") {
            let parts: Vec<&str> = rest.split('.').collect();
            if parts.len() == 2 {
                let role = self.point_role(parts[0], parts[1]);
                return lower_text(&format!("sketch.{}.{}.{role}", parts[0], parts[1]));
            }
        }
        let v = lower_text(&self.text_of(pick)?)?;
        if v.get("point").is_some() {
            return Ok(v);
        }
        if v.get("ref").is_some() {
            return Ok(json!({ "point": "vertex", "vertex": v }));
        }
        Err(model(format!("`{pick}` isn't a point or a vertex")))
    }

    fn point_role(&self, sketch: &str, entity: &str) -> &'static str {
        let kind = self
            .document
            .features
            .iter()
            .find(|f| f.id.0 == sketch)
            .and_then(|f| serde_json::to_value(&f.kind).ok())
            .and_then(|k| {
                k["entities"].as_array().and_then(|es| {
                    es.iter()
                        .find(|e| e["id"] == entity)
                        .and_then(|e| e["geometry"]["entity"].as_str().map(str::to_owned))
                })
            });
        match kind.as_deref() {
            Some("point") => "at",
            Some("circle") | Some("arc") => "center",
            _ => "start",
        }
    }

    /// The sketch a profile pick names.
    pub fn sketch(&self, pick: &str) -> Result<String, EditFailure> {
        let id = pick
            .strip_prefix("feature:")
            .or_else(|| pick.strip_prefix("sketch:"))
            .or_else(|| pick.strip_prefix("sketch."))
            .map(|s| s.split('.').next().unwrap_or(s))
            .ok_or_else(|| model(format!("`{pick}` isn't a sketch")))?;
        match self.kind_of(id) {
            Some("sketch") => Ok(id.to_owned()),
            _ => Err(model(format!("`{id}` isn't a sketch"))),
        }
    }

    /// A sketch entity pick (`sketch.s1.l1`) as (sketch, entity).
    pub fn sketch_entity(&self, pick: &str) -> Result<(String, String), EditFailure> {
        let rest = pick
            .strip_prefix("sketch.")
            .ok_or_else(|| model(format!("`{pick}` isn't a sketch entity")))?;
        let (s, e) = rest
            .split_once('.')
            .ok_or_else(|| model(format!("`{pick}` isn't a sketch entity")))?;
        Ok((s.to_owned(), e.to_owned()))
    }

    /// The body (reference) a face belongs to, from the display.
    pub fn body_of_face(&self, face: &str) -> Option<Value> {
        let body = self.bodies().iter().find(|b| {
            b["faces"].as_array().is_some_and(|fs| {
                fs.iter().any(|f| {
                    f["reference"] == face
                        || f["names"]
                            .as_array()
                            .is_some_and(|n| n.iter().any(|n| n == face))
                })
            })
        })?;
        lower_text(body["reference"].as_str()?).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_lower_to_the_values_their_text_writes() {
        let face = lower_text("feature.body.face.end").unwrap();
        assert_eq!(face["ref"], "name");
        let edge =
            lower_text("edges_between(feature.body.face.side(s1.l1), feature.body.face.end)")
                .unwrap();
        assert_eq!(edge["ref"], "edges_between");
        let plane = lower_text("origin.xy").unwrap();
        assert_eq!(plane["on"], "datum");
        assert!(lower_text("3 mm").is_err() || lower_text("3 mm").unwrap().get("ref").is_none());
    }
}
