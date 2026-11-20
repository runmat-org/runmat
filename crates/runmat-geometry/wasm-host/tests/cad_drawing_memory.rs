//! Drawing compiles fit the browser CAD host's memory.
//!
//! An allocation past the browser host's memory aborts the CAD worker, and the editor with it.
//! The camera drone's assembly drawing projects the whole drone in three views. These tests
//! compile the drone's drawings the way the browser worker does (`call_framed`, meshes as bytes)
//! and measure the Rust heap's peak during the compile.
//!
//! The Rust heap is what a drawing compile controls: the parts it regenerates into the part
//! cache, the display meshes as JSON values, the hidden-line models and the answer's text. The
//! test holds its peak to [`PEAK_BYTES`], which leaves room for OCCT's own allocations (this
//! allocator doesn't see them) and the host allocator's slack. Native pointers are twice the
//! size of wasm32's, so the native peak overstates the host's.

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

/// The most a drawing compile's Rust heap may reach: 256 MB. The drone's assembly drawing peaks
/// at about 190 MB (natively), so a regression past the margin fails here. The bound was three
/// quarters of the browser host's former 512 MB ceiling; the drone's peak justifies a tighter one.
const PEAK_BYTES: usize = 268_435_456;

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
    let dir = std::env::var("RUNMAT_DRONE_SRC")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
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

/// Every drone drawing compiles with no error and a heap peak within [`PEAK_BYTES`].
/// `RUNMAT_DRAWING_DUMP=<dir>` writes each answer and its bytes there, so
/// a change to the pipeline can be byte-compared.
#[test]
fn the_drone_drawings_compile_within_the_host_memory() {
    let files = drone();
    let dump = std::env::var("RUNMAT_DRAWING_DUMP").ok();
    let mut over = Vec::new();
    for name in ["drone", "arm", "hinge_mount"] {
        let path = format!("/p/drone/drawings/{name}.drawing");
        runmat_cad_wasm::runtime::part_cache::clear();
        let started = std::time::Instant::now();
        let (json, bytes, peak) = compile(&path, &files);
        eprintln!(
            "{name}.drawing: heap peak {} MB, answer {} chars + {} bytes, {:?}",
            peak >> 20,
            json.len(),
            bytes.len(),
            started.elapsed()
        );
        if let Some(dir) = &dump {
            std::fs::write(format!("{dir}/{name}.json"), &json).unwrap();
            std::fs::write(format!("{dir}/{name}.bin"), &bytes).unwrap();
        }
        let answer: serde_json::Value = serde_json::from_str(&json).expect("json");
        let errors: Vec<_> = answer["diagnostics"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|d| d["severity"] == "error")
            .collect();
        assert!(errors.is_empty(), "{name}.drawing: {errors:?}");
        if peak > PEAK_BYTES {
            over.push(format!("{name}.drawing peaked at {} MB", peak >> 20));
        }
    }
    assert!(
        over.is_empty(),
        "over the {} MB peak a compile may reach: {over:?}",
        PEAK_BYTES >> 20
    );
}
