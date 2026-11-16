//! An assembly's motion studies as the Desktop plays them. Each study of the top assembly is
//! solved frame by frame over the resolved tree ([`runmat_cad::assembly::motion::kinematics`]),
//! and the compile's projection carries it under `motion`:
//!
//! ```json
//! [{ "name": "open_close", "duration": 4, "profile": "linear", "track": "lid",
//!    "drives": [{ "mate": "hinge", "quantity": "angle", "unit": "deg", "from": 0, "to": 90,
//!                 "values": [0, 1.9, …] }],
//!    "times": [0, 0.083, …],
//!    "frames": [[{ "uuid": "…", "transform": [r00 … r22, tx, ty, tz] }, …], …],
//!    "plot": { "occurrence": "lid", "speed": [mm/s …], "acceleration": [mm/s² …],
//!              "angular_speed": [deg/s …] },
//!    "stopped": { "time": 3.2, "message": "…" }, "problems": ["…"] }]
//! ```
//!
//! Each frame lists the moving occurrences (and their subtrees) with their solved world
//! transforms, in the display's layout: the rotation row by row, then the translation, in metres.
//! The canvas swaps each listed occurrence's transform for the frame's while the study plays.
//! The projection's `positions` lists the named mate positions.

use runmat_cad::assembly::motion::drag::{top_node, Dragged};
use runmat_cad::assembly::motion::kinematics::{
    run_study, DriveQuantity, MotionContext, MotionRun,
};
use runmat_cad::assembly::resolve::{ResolveContext, ResolveRequest, ResolvedAssembly};
use runmat_cad::assembly::source::{DesignSource, MemorySource};
use runmat_cad::regen::kernel::Transform;
use runmat_cad::units::Units;
use runmat_cad::value::{NoDynamicValues, ValueExpr};
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

/// Rounded for the projection (it is sent on every compile).
fn round(v: f64, places: i32) -> f64 {
    let f = 10f64.powi(places);
    (v * f).round() / f
}

/// The resolve context and request the studies and drags solve with.
fn with_context<T>(
    resolved: &ResolvedAssembly,
    source: &MemorySource,
    f: impl FnOnce(&MotionContext<'_>) -> T,
) -> T {
    let ctx = ResolveContext {
        source,
        units: &Units,
        dynamic: &NoDynamicValues,
        lock: None,
        catalog: None,
        tables: None,
        gauge: None,
    };
    let request = ResolveRequest {
        configuration: resolved.node(0).configuration.name.clone(),
        ..ResolveRequest::default()
    };
    f(&MotionContext {
        resolve: &ctx,
        request: &request,
    })
}

/// Top-level component `occurrence` dragged by `delta` (m) from its box centre, through the
/// freedom its mates leave ([`runmat_cad::assembly::motion::drag`]).
pub fn drag(
    resolved: &ResolvedAssembly,
    source: &MemorySource,
    occurrence: &str,
    delta: [f64; 3],
) -> Result<Dragged, String> {
    let doc = source
        .assembly(&resolved.top, None)
        .map_err(|_| "the assembly doesn't compile".to_owned())?;
    let node = top_node(resolved, occurrence)
        .ok_or_else(|| format!("the assembly has no component `{occurrence}`"))?;
    let n = resolved.node(node);
    let grab = source
        .configured(&n.definition, n.revision.as_ref(), &n.configuration)
        .ok()
        .and_then(|c| c.bbox)
        .filter(|b| !b.is_empty())
        .map(|b| std::array::from_fn(|i| (b.min[i] + b.max[i]) / 2.0))
        .unwrap_or([0.0; 3]);
    with_context(resolved, source, |ctx| {
        runmat_cad::assembly::motion::drag::drag(&doc, resolved, occurrence, grab, delta, ctx)
    })
}

/// Study `name` of the top assembly, solved.
pub fn run(resolved: &ResolvedAssembly, source: &MemorySource, name: &str) -> Option<MotionRun> {
    let doc = source.assembly(&resolved.top, None).ok()?;
    Some(with_context(resolved, source, |ctx| run_study(&doc, resolved, name, ctx)))
}

/// Every node under (and including) the moving nodes.
fn moving_nodes(resolved: &ResolvedAssembly, run: &MotionRun) -> Vec<usize> {
    let mut out = Vec::new();
    let mut queue: Vec<u32> = run.moving.clone();
    while let Some(n) = queue.pop() {
        out.push(n as usize);
        queue.extend(resolved.node(n).children.iter().copied());
    }
    out.sort_unstable();
    out
}

fn study_json(resolved: &ResolvedAssembly, run: &MotionRun, profile: &str, track: Option<&str>) -> Value {
    let nodes = moving_nodes(resolved, run);
    let frames: Vec<Value> = run
        .frames
        .iter()
        .map(|f| {
            Value::Array(
                nodes
                    .iter()
                    .map(|&n| {
                        json!({
                            "uuid": resolved.nodes[n].uuid.to_string(),
                            "transform": transform(&f[n]).into_iter().map(|v| round(v, 9)).collect::<Vec<_>>(),
                        })
                    })
                    .collect(),
            )
        })
        .collect();
    let drives: Vec<Value> = run
        .drives
        .iter()
        .map(|d| {
            let (unit, scale) = match d.quantity {
                DriveQuantity::Angle => ("deg", 180.0 / std::f64::consts::PI),
                DriveQuantity::Length => ("mm", 1000.0),
            };
            let values: Vec<f64> = d.values.iter().map(|v| round(v * scale, 4)).collect();
            json!({
                "mate": d.mate,
                "quantity": d.quantity,
                "unit": unit,
                "from": values.first(),
                "to": values.last(),
                "values": values,
            })
        })
        .collect();
    let plot = run.track.as_ref().map(|t| {
        let norm = |v: &[f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        json!({
            "occurrence": t.occurrence,
            "speed": t.velocity.iter().map(|v| round(norm(v) * 1000.0, 4)).collect::<Vec<_>>(),
            "acceleration": t.acceleration.iter().map(|a| round(norm(a) * 1000.0, 3)).collect::<Vec<_>>(),
            "angular_speed": t.angular_speed.iter().map(|w| round(w.to_degrees(), 4)).collect::<Vec<_>>(),
            "position": t.position.iter().map(|p| p.map(|c| round(c, 9))).collect::<Vec<_>>(),
        })
    });
    json!({
        "name": run.name,
        "duration": run.duration,
        "profile": profile,
        "track": track,
        "drives": drives,
        "times": run.times.iter().map(|t| round(*t, 6)).collect::<Vec<_>>(),
        "frames": frames,
        "plot": plot,
        "stopped": run.stopped.as_ref().map(|(t, m)| json!({ "time": round(*t, 6), "message": m })),
        "problems": run.problems,
    })
}

/// The top assembly's motion studies, solved; an empty array when it has none.
pub fn motion_studies(resolved: &ResolvedAssembly, source: &MemorySource) -> Value {
    let Ok(doc) = source.assembly(&resolved.top, None) else {
        return json!([]);
    };
    Value::Array(
        doc.motions
            .iter()
            .filter_map(|s| {
                let run = run(resolved, source, &s.name)?;
                Some(study_json(resolved, &run, s.profile.key(), s.track.as_deref()))
            })
            .collect(),
    )
}

/// A value as the Outline shows it (`90 deg`, `25 mm`), from its expression.
fn shown(e: &ValueExpr) -> String {
    match e {
        ValueExpr::Quantity { value, unit } => format!("{} {unit}", round(*value, 6)),
        other => runmat_cad::syntax::value::print_value(other).unwrap_or_default(),
    }
}

/// The top assembly's named mate positions: `[{ name, values: [{ mate, value }] }]`.
pub fn mate_positions(resolved: &ResolvedAssembly, source: &MemorySource) -> Value {
    let Ok(doc) = source.assembly(&resolved.top, None) else {
        return json!([]);
    };
    Value::Array(
        doc.positions
            .iter()
            .map(|p| {
                let problem = p
                    .values
                    .iter()
                    .find(|v| !doc.mates.iter().any(|m| m.name == v.mate))
                    .map(|v| format!("no mate `{}` in the assembly", v.mate));
                json!({
                    "name": p.name,
                    "values": p.values.iter().map(|v| json!({ "mate": v.mate, "value": shown(&v.value) })).collect::<Vec<_>>(),
                    "problem": problem,
                })
            })
            .collect(),
    )
}
