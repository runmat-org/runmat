//! An assembly's saved exploded views as the Desktop shows them. Each view of the top assembly
//! is evaluated over the resolved tree (`assembly::explode::evaluate_exploded`), and the
//! compile's projection carries it under `exploded`:
//!
//! ```json
//! [{ "name": "service", "base": null,
//!    "steps": [{ "id": "lift_top", "motion": "translate", "occurrences": ["body_top"] }],
//!    "lines": [{ "id": "l1", "occurrence": "battery", "points": [[x, y, z], …] }],
//!    "occurrences": [{ "uuid": "…", "path": "body_top", "transform": [r00 … r22, tx, ty, tz] }],
//!    "diagnostics": ["…"] }]
//! ```
//!
//! `occurrences` lists every moved occurrence (and its subtree) with its exploded world
//! transform in the display's layout: the rotation row by row, then the translation, in metres.
//! Line points are in metres too. The canvas swaps each listed occurrence's transform for the
//! exploded one while the view is shown.

use runmat_cad::assembly::explode::{evaluate_exploded, exploded_index, ExplodeContext, ExplodeMotion};
use runmat_cad::assembly::resolve::ResolvedAssembly;
use runmat_cad::assembly::source::{DesignSource, MemorySource};
use runmat_cad::regen::kernel::Transform;
use runmat_cad::units::StubUnits;
use runmat_cad::value::NoDynamicValues;
use serde_json::{json, Value};

fn transform(w: &Transform) -> Vec<f64> {
    let mut t = Vec::with_capacity(12);
    for r in 0..3 {
        for c in 0..3 {
            t.push(w.rotation[r][c]);
        }
    }
    t.extend(w.translation);
    t
}

/// The top assembly's exploded views, evaluated; an empty array when it has none.
pub fn exploded_views(resolved: &ResolvedAssembly, source: &MemorySource) -> Value {
    let Ok(doc) = source.assembly(&resolved.top, None) else {
        return json!([]);
    };
    let ctx = ExplodeContext {
        units: &StubUnits,
        dynamic: &NoDynamicValues,
        lock: None,
        source: Some(source),
    };
    let views: Vec<Value> = doc
        .exploded
        .iter()
        .map(|view| {
            let state = evaluate_exploded(&doc, resolved, &view.name, &ctx);
            let index = exploded_index(resolved, &state);
            let steps: Vec<Value> = view
                .steps
                .iter()
                .map(|s| {
                    let motion = match s.motion {
                        ExplodeMotion::Translate { .. } => "translate",
                        ExplodeMotion::Rotate { .. } => "rotate",
                        ExplodeMotion::Radial { .. } => "radial",
                    };
                    json!({
                        "id": s.id,
                        "motion": motion,
                        "occurrences": s.occurrences.iter().map(|o| o.to_string()).collect::<Vec<_>>(),
                    })
                })
                .collect();
            json!({
                "name": view.name,
                "base": view.base,
                "steps": steps,
                "lines": index.lines.iter().map(|l| json!({ "id": l.id, "occurrence": l.occurrence, "points": l.points })).collect::<Vec<_>>(),
                "occurrences": index.occurrences.iter().map(|o| json!({ "uuid": o.uuid.to_string(), "path": o.path, "transform": transform(&o.world) })).collect::<Vec<_>>(),
                "diagnostics": index.diagnostics.iter().map(|d| d.message.clone()).collect::<Vec<_>>(),
            })
        })
        .collect();
    Value::Array(views)
}
