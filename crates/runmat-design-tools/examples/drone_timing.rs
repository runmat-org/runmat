//! Where a design compile's time goes, natively (release): each part's regeneration by feature,
//! its display and display levels, its volume and bounds; then the whole assembly compiled cold,
//! again unchanged, and after an edit to one part (the per-part cache, `runtime::part_cache`).
//!
//! `cargo run --release -p runmat-design-tools --features occt --example drone_timing -- [project dir] [top file] [edited part]`
//! (defaults: runmat-cad-wasm's tests/drone, /drone.assembly, /mechanical/body/battery.part). `FOCUS=a,b` limits the
//! per-part pass to parts whose path contains one of the words. Prints one JSON object.

use runmat_cad::metering::{MeterClock, UsageEvent, UsageMeter};
use runmat_cad::regen::{regenerate_metered, RegenCache, RegenContext, RegenRequest};
use runmat_cad::sketch_solver::ConstraintSketchSolver;
use runmat_cad::units::StubUnits;
use runmat_cad::value::NoDynamicValues;
use runmat_cad_wasm::runtime::{compile_json, part_cache, Files, ASSEMBLY_TESSELLATION};
use serde_json::{json, Value};
use std::cell::RefCell;
use std::path::Path;
use std::time::Instant;

struct Clock(Instant);
impl MeterClock for Clock {
    fn now_nanos(&self) -> u64 {
        self.0.elapsed().as_nanos() as u64
    }
}
struct Meter {
    clock: Clock,
    features: RefCell<Vec<(String, String, f64)>>,
}
impl UsageMeter for Meter {
    fn record(&self, event: &UsageEvent) {
        if let UsageEvent::FeatureRegeneration { feature, kind, duration_nanos, .. } = event {
            self.features.borrow_mut().push((format!("{feature:?}"), kind.clone(), duration_nanos.unwrap_or(0) as f64 / 1e6));
        }
    }
    fn clock(&self) -> Option<&dyn MeterClock> {
        Some(&self.clock)
    }
}

fn walk(root: &Path, dir: &Path, out: &mut Files) {
    for entry in std::fs::read_dir(dir).unwrap().flatten() {
        let p = entry.path();
        if p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with('.')) {
            continue;
        }
        if p.is_dir() {
            walk(root, &p, out);
        } else if let Ok(text) = std::fs::read_to_string(&p) {
            out.insert(format!("/{}", p.strip_prefix(root).unwrap().display()), text);
        }
    }
}

fn ms(t: Instant) -> f64 {
    (t.elapsed().as_secs_f64() * 1e4).round() / 10.0
}

fn part_pass(path: &str, text: &str) -> Value {
    let parsed = runmat_cad::syntax::parse_part(path, text);
    let Some(document) = parsed.document else { return json!({ "path": path, "error": "syntax" }) };
    let kernel = runmat_cad_wasm::kernel::make();
    let meter = Meter { clock: Clock(Instant::now()), features: RefCell::default() };
    let ctx = RegenContext {
        kernel: kernel.as_ref(),
        solver: &ConstraintSketchSolver,
        units: &StubUnits,
        dynamic: &NoDynamicValues,
        lock: None,
        catalog: None,
        policy: Default::default(),
        slots: None,
    };
    let t = Instant::now();
    let result = regenerate_metered(&document, &ctx, &RegenRequest::default(), &mut RegenCache::default(), &meter);
    let regen_ms = ms(t);
    let t = Instant::now();
    let display = runmat_cad::display::part_display(&result, kernel.as_ref(), &ASSEMBLY_TESSELLATION);
    let display_ms = ms(t);
    let t = Instant::now();
    let levels = runmat_cad::display::display_levels(&result, kernel.as_ref(), &ASSEMBLY_TESSELLATION);
    let levels_ms = ms(t);
    let mut features = meter.features.into_inner();
    features.sort_by(|a, b| b.2.total_cmp(&a.2));
    let triangles: usize = display.bodies.iter().flat_map(|b| &b.faces).map(|f| f.triangles.len() / 3).sum();
    let level_triangles: Vec<usize> = levels.iter().map(|l| l.triangles.len() / 3).collect();
    let mass = runmat_cad_wasm::mass::part_mass(path, text, None).ok();
    json!({
        "path": path,
        "regen_ms": regen_ms,
        "display_ms": display_ms,
        "levels_ms": levels_ms,
        "slowest": features.iter().take(6).map(|(id, kind, ms)| json!([id, kind, (ms * 10.0).round() / 10.0])).collect::<Vec<_>>(),
        "triangles": triangles,
        "level_triangles": level_triangles,
        "volume_m3": mass.as_ref().map(|m| m.total.volume),
        "bounds": display.bounds.as_ref().map(|b| json!({ "min": b.min, "max": b.max })),
        "errors": result.diagnostics.diagnostics.iter().filter(|d| d.severity == runmat_cad::diagnostics::Severity::Error).map(|d| d.message.clone()).collect::<Vec<_>>(),
    })
}

fn main() {
    // OCCT, the kernel the Desktop and the browser regenerate on.
    runmat_design_tools::kernel::install();
    assert_eq!(runmat_cad_wasm::kernel::name(), "occt");
    let mut args = std::env::args().skip(1);
    let dir = args.next().unwrap_or_else(|| concat!(env!("CARGO_MANIFEST_DIR"), "/../runmat-cad/wasm/tests/drone").to_owned());
    let top = args.next().unwrap_or_else(|| "/drone.assembly".to_owned());
    let edited = args.next().unwrap_or_else(|| "/mechanical/body/battery.part".to_owned());
    let root = Path::new(&dir);
    let mut files = Files::new();
    walk(root, root, &mut files);
    let focus: Vec<String> = std::env::var("FOCUS").map(|f| f.split(',').map(str::to_owned).collect()).unwrap_or_default();

    let mut parts = Vec::new();
    if std::env::var("SKIP_PARTS").is_err() {
        for (path, text) in &files {
            if !path.ends_with(".part") || (!focus.is_empty() && !focus.iter().any(|f| path.contains(f.as_str()))) {
                continue;
            }
            parts.push(part_pass(path, text));
        }
    }

    let mut compiles = Vec::new();
    if std::env::var("SKIP_ASSEMBLY").is_err() {
        let text = files[&top].clone();
        let mut run = |label: &str, files: &Files| {
            let before = part_cache::stats();
            let t = Instant::now();
            let out = compile_json(&top, &text, files);
            let took = ms(t);
            let after = part_cache::stats();
            let parsed: Value = serde_json::from_str(&out).unwrap_or(Value::Null);
            let errors = parsed["diagnostics"].as_array().map(|d| d.iter().filter(|d| d["severity"] == "error").count()).unwrap_or(0);
            compiles.push(json!({ "label": label, "ms": took, "bytes": out.len(), "errors": errors, "hits": after.hits - before.hits, "misses": after.misses - before.misses, "cache_bytes": after.bytes }));
            out
        };
        part_cache::clear();
        let cold = run("cold", &files);
        let warm = run("unchanged", &files);
        assert_eq!(cold, warm, "an unchanged recompile gives the same output");
        let mut changed = files.clone();
        if let Some(t) = changed.get_mut(&edited) {
            // One dimension changes: the first `mm` value gains 0.5 mm.
            if let Some(at) = t.find(" mm") {
                let start = t[..at].rfind(|c: char| !(c.is_ascii_digit() || c == '.')).map(|i| i + 1).unwrap_or(0);
                if let Ok(v) = t[start..at].parse::<f64>() {
                    t.replace_range(start..at, &format!("{}", v + 0.5));
                }
            }
        }
        run("one part edited", &changed);
    }
    println!("{}", serde_json::to_string_pretty(&json!({ "parts": parts, "compiles": compiles })).unwrap());
}
