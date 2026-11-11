//! The airliner's general arrangement drawing compiles in the browser CAD host's memory.
//!
//! The drawing projects the whole airliner (about 3,200 occurrences) in three views at 1:200.
//! The test compiles it the way the Desktop's drawing worker does: the sheet stage, then one
//! view per compile (`call_framed`, meshes as bytes), and holds each compile's Rust heap peak to
//! three quarters of the host's `-sMAXIMUM_MEMORY` ceiling, as `cad_drawing_memory` does for the
//! drone.

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Counting;

static NOW: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = System.alloc(layout);
        if !p.is_null() {
            let now = NOW.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(now, Ordering::Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        NOW.fetch_sub(layout.size(), Ordering::Relaxed);
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let p = System.realloc(ptr, layout, new_size);
        if !p.is_null() {
            if new_size >= layout.size() {
                let now = NOW.fetch_add(new_size - layout.size(), Ordering::Relaxed) + new_size
                    - layout.size();
                PEAK.fetch_max(now, Ordering::Relaxed);
            } else {
                NOW.fetch_sub(layout.size() - new_size, Ordering::Relaxed);
            }
        }
        p
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

#[path = "../../../runmat-cad/wasm/tests/aircraft_text/mod.rs"]
#[allow(dead_code)]
mod aircraft_text;

/// The browser host's memory ceiling, `-sMAXIMUM_MEMORY` in build-exact-cad-wasm.sh.
fn host_ceiling() -> usize {
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../scripts/runtime/build-exact-cad-wasm.sh");
    let text = std::fs::read_to_string(&script).expect("build-exact-cad-wasm.sh");
    let at = text
        .find("-sMAXIMUM_MEMORY=")
        .expect("the host build sets its memory ceiling")
        + "-sMAXIMUM_MEMORY=".len();
    text[at..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>()
        .parse()
        .expect("a byte count")
}

/// The general arrangement: side elevation, plan and front view of the whole aircraft at 1:200.
const GA: &str = r#"runmat 0.7
std 1

drawing airliner {
    format a2 = {
        size = { size = iso, name = A2 },
        title_block = [
            { label = "Title", source = { source = literal, text = "General arrangement" } },
            { label = "Scale", source = { source = scale } }
        ]
    }

    sheet "1" = { format = a2, scale = [1, 200], projection = third_angle }

    view front = standard(orientation = front, sheet = "1", model = { assembly = airliner, from = ".." }, at = [150, 105], display = visible_lines)
    view top = projected(from = front, direction = up, sheet = "1", model = { assembly = airliner, from = ".." }, at = [150, 280], display = visible_lines)
    view right = projected(from = front, direction = right, sheet = "1", model = { assembly = airliner, from = ".." }, at = [390, 105], display = visible_lines)

    table parts = bom(view = front, sheet = "1", at = [330, 230])
}
"#;

/// Compiles `path` with `options` as the browser worker does; the answer and the heap's peak
/// above what was held before the call.
fn compile(path: &str, files: &BTreeMap<String, String>, options: &str) -> (String, usize) {
    let request = serde_json::json!({
        "fn": "compile",
        "args": [path, files[path], serde_json::to_string(files).unwrap(), options],
    })
    .to_string();
    let base = NOW.load(Ordering::Relaxed);
    PEAK.store(base, Ordering::Relaxed);
    let (json, _bytes) = runmat_geometry_wasm_host::cad::call_framed(&request);
    let peak = PEAK.load(Ordering::Relaxed) - base;
    (json, peak)
}

/// The general arrangement compiles stage by stage with no error, each view has visible lines,
/// and each compile's heap peak stays within its share of the host's ceiling.
#[test]
fn the_airliner_general_arrangement_compiles_within_the_host_memory() {
    let mut files: BTreeMap<String, String> = aircraft_text::files()
        .into_iter()
        .map(|(n, t)| (format!("/p/airliner/{n}"), t))
        .collect();
    let path = "/p/airliner/drawings/airliner.drawing".to_string();
    files.insert(path.clone(), GA.to_string());
    let share = host_ceiling() / 4 * 3;
    let mut over = Vec::new();
    for (stage, options) in [
        ("sheet", r#"{"drawing":{"stage":"sheet"}}"#),
        ("front", r#"{"drawing":{"views":["front"]}}"#),
        ("top", r#"{"drawing":{"views":["front","top"]}}"#),
        ("right", r#"{"drawing":{"views":["front","top","right"]}}"#),
    ] {
        let started = std::time::Instant::now();
        let (json, peak) = compile(&path, &files, options);
        eprintln!(
            "airliner.drawing {stage}: heap peak {} MB, answer {} chars, {:?}",
            peak >> 20,
            json.len(),
            started.elapsed()
        );
        let answer: serde_json::Value = serde_json::from_str(&json).expect("json");
        let errors: Vec<_> = answer["diagnostics"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|d| d["severity"] == "error")
            .collect();
        assert!(errors.is_empty(), "{stage}: {errors:?}");
        if stage != "sheet" {
            let entry = &answer["projection"]["data"]["hlr"][stage];
            eprintln!(
                "  {stage}: visible {} runs ({} chars), picks {} ({} chars), models {} chars",
                entry["visible"].as_array().map(Vec::len).unwrap_or(0),
                entry["visible"].to_string().len(),
                entry["picks"].as_array().map(Vec::len).unwrap_or(0),
                entry["picks"].to_string().len(),
                answer["models"].to_string().len(),
            );
            let lines = &entry["visible"];
            let count = lines.as_array().map(Vec::len).unwrap_or(0);
            assert!(count > 0, "{stage} has no visible lines");
        }
        if peak > share {
            over.push(format!("{stage} peaked at {} MB", peak >> 20));
        }
    }
    assert!(
        over.is_empty(),
        "over the compile's {} MB share of the host's {} MB: {over:?}",
        share >> 20,
        host_ceiling() >> 20
    );
}
