//! Drawing compiles fit the browser CAD host's memory.
//!
//! The browser host links with a 512 MB memory ceiling (`-sMAXIMUM_MEMORY` in
//! scripts/runtime/build-exact-cad-wasm.sh); an allocation past it aborts the CAD worker, and the
//! editor with it. The camera drone's assembly drawing projects the whole drone in three views.
//! These tests compile the drone's drawings the way the browser worker does
//! (`call_framed`, meshes as bytes) and measure the Rust heap's peak during the compile.
//!
//! The Rust heap is what a drawing compile controls: the parts it regenerates into the part
//! cache, the display meshes as JSON values, the hidden-line models and the answer's text. The
//! test holds its peak to three quarters of the ceiling and leaves the last quarter to OCCT's
//! own allocations, which this allocator doesn't see, and to the host allocator's slack. Native
//! pointers are twice the size of wasm32's, so the native peak overstates the host's.

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
                let now = NOW.fetch_add(new_size - layout.size(), Ordering::Relaxed) + new_size - layout.size();
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

/// The browser host's memory ceiling, `-sMAXIMUM_MEMORY` in build-exact-cad-wasm.sh.
fn host_ceiling() -> usize {
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../scripts/runtime/build-exact-cad-wasm.sh");
    let text = std::fs::read_to_string(&script).expect("build-exact-cad-wasm.sh");
    let at = text.find("-sMAXIMUM_MEMORY=").expect("the host build sets its memory ceiling") + "-sMAXIMUM_MEMORY=".len();
    text[at..].chars().take_while(char::is_ascii_digit).collect::<String>().parse().expect("a byte count")
}

fn read_tree(dir: &std::path::Path, at: &str, all: &mut BTreeMap<String, String>) {
    for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') {
            continue;
        }
        if e.path().is_dir() {
            read_tree(&e.path(), &format!("{at}/{name}"), all);
        } else if let Ok(text) = std::fs::read_to_string(e.path()) {
            all.insert(format!("{at}/{name}"), text);
        }
    }
}

fn drone() -> BTreeMap<String, String> {
    let dir = std::env::var("RUNMAT_DRONE_SRC").map(PathBuf::from).unwrap_or_else(|_| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../runmat-cad/wasm/tests/drone")
    });
    let mut all = BTreeMap::new();
    read_tree(&dir, "/p/drone", &mut all);
    all
}

/// Compiles `path` as the browser worker does; the answer, its mesh bytes and the heap's peak
/// above what was held before the call.
fn compile(path: &str, files: &BTreeMap<String, String>) -> (String, Vec<u8>, usize) {
    let request = serde_json::json!({
        "fn": "compile",
        "args": [path, files[path], serde_json::to_string(files).unwrap(), "{}"],
    })
    .to_string();
    let base = NOW.load(Ordering::Relaxed);
    PEAK.store(base, Ordering::Relaxed);
    let (json, bytes) = runmat_geometry_wasm_host::cad::call_framed(&request);
    let peak = PEAK.load(Ordering::Relaxed) - base;
    (json, bytes, peak)
}

/// Every drone drawing compiles with no error and a heap peak within its share of the host's
/// ceiling. `RUNMAT_DRAWING_DUMP=<dir>` writes each answer and its bytes there, so
/// a change to the pipeline can be byte-compared.
#[test]
fn the_drone_drawings_compile_within_the_host_memory() {
    let files = drone();
    let share = host_ceiling() / 4 * 3;
    let dump = std::env::var("RUNMAT_DRAWING_DUMP").ok();
    let mut over = Vec::new();
    for name in ["drone", "arm", "hinge_mount"] {
        let path = format!("/p/drone/drawings/{name}.drawing");
        runmat_cad_wasm::runtime::part_cache::clear();
        let started = std::time::Instant::now();
        let (json, bytes, peak) = compile(&path, &files);
        eprintln!("{name}.drawing: heap peak {} MB, answer {} chars + {} bytes, {:?}", peak >> 20, json.len(), bytes.len(), started.elapsed());
        if let Some(dir) = &dump {
            std::fs::write(format!("{dir}/{name}.json"), &json).unwrap();
            std::fs::write(format!("{dir}/{name}.bin"), &bytes).unwrap();
        }
        let answer: serde_json::Value = serde_json::from_str(&json).expect("json");
        let errors: Vec<_> = answer["diagnostics"].as_array().into_iter().flatten().filter(|d| d["severity"] == "error").collect();
        assert!(errors.is_empty(), "{name}.drawing: {errors:?}");
        if peak > share {
            over.push(format!("{name}.drawing peaked at {} MB", peak >> 20));
        }
    }
    assert!(over.is_empty(), "over the compile's {} MB share of the host's {} MB: {over:?}", share >> 20, host_ceiling() >> 20);
}
