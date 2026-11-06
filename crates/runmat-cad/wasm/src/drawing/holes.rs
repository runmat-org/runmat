//! Hole tables' rows ([`runmat_cad::drawing::hole_table`]) from the parts the drawing shows:
//! each hole feature's recorded holes, the view's axes, and the origin a table names. The
//! compile puts them in the projection as `hole_tables`.

use crate::runtime::RegeneratedPart;
use runmat_cad::drawing::hole_table::{hole_table, HoleGeometry, HoleInstance};
use runmat_cad::drawing::{DrawingDocument, ModelRef, TableKind, View};
use runmat_cad::model::datum::Frame;
use runmat_cad::refs::reference::Reference;
use runmat_cad::regen::records::FeatureRecord;
use serde_json::{Map, Value};
use std::collections::BTreeMap;

struct Geometry<'a> {
    parts: &'a BTreeMap<String, RegeneratedPart>,
    displays: &'a Map<String, Value>,
    views: &'a Value,
}

impl HoleGeometry for Geometry<'_> {
    fn holes(&self, model: &ModelRef) -> Result<Vec<HoleInstance>, String> {
        let part = self.parts.get(model.file.as_str()).ok_or_else(|| format!("{} isn't a part that compiles", model.file.as_str()))?;
        Ok(part
            .document
            .features
            .iter()
            .filter_map(|f| match part.result.state.records.get(&f.id) {
                Some(FeatureRecord::Holes { holes }) => Some(holes.clone()),
                _ => None,
            })
            .flatten()
            .collect())
    }

    fn view_frame(&self, view: &View) -> Result<Frame, String> {
        let a = crate::hlr::axes(crate::hlr::orientation(self.views, &view.name, 0));
        Ok(Frame { origin: [0.0; 3], x: a.u, y: a.v, z: [-a.depth[0], -a.depth[1], -a.depth[2]] })
    }

    fn point(&self, model: &ModelRef, reference: &Reference) -> Result<[f64; 3], String> {
        let display = self.displays.get(model.file.as_str()).ok_or("has no display")?;
        // A planar face's middle by its extent (a fillet's many vertices don't pull it aside).
        let boxed = super::extent_of(display, reference);
        let mid = (super::layout::floats(&boxed[0]).len() == 3).then(|| {
            let (lo, hi) = (super::layout::floats(&boxed[0]), super::layout::floats(&boxed[1]));
            [0, 1, 2].map(|i| (lo[i] + hi[i]) / 2.0)
        });
        let p = mid.or_else(|| super::point_of(display, reference)).ok_or("names no point in the model")?;
        Ok(p.map(|v| v / 1000.0))
    }
}

/// Table name → its rows, for each hole table of the drawing.
pub fn hole_tables(document: &DrawingDocument, parts: &BTreeMap<String, RegeneratedPart>, displays: &Map<String, Value>) -> Value {
    let views = serde_json::to_value(&document.views).unwrap_or(Value::Null);
    let geometry = Geometry { parts, displays, views: &views };
    let mut out = Map::new();
    for t in &document.tables {
        let TableKind::Hole { view, origin, spec } = &t.kind else { continue };
        let Some(v) = document.views.iter().find(|v| &v.name == view) else { continue };
        let data = hole_table(&t.name, spec, v, origin, &geometry);
        out.insert(t.name.clone(), serde_json::to_value(&data).unwrap_or(Value::Null));
    }
    Value::Object(out)
}
