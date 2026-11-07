//! Part regenerations kept across compiles on this thread (the CAD worker keeps one host, so
//! one cache, for the page's life).
//!
//! An assembly compile regenerates every part file it places. A part's regeneration depends on
//! its path (diagnostics name it), its text, the configuration it is regenerated in (a name and
//! the input values an occurrence sets) and the tessellation; a part reads no other file. Each
//! entry is keyed on exactly those, so an unchanged part reuses its document, display and
//! published datums, and any change to them is a new key. A part whose regeneration asked for a
//! dynamic value (an `.m` function, a published value of another file, a snapshot) depends on
//! more than its key says, so it is never kept.
//!
//! The cache holds at most [`BUDGET_BYTES`] of display meshes; the least recently used entries
//! go first.

use super::Diagnostic;
use runmat_cad::assembly::source::{ConfigKey, PublishedGeometry};
use runmat_cad::display::PartDisplay;
use runmat_cad::ids::PublishName;
use runmat_cad::model::PartDocument;
use runmat_cad::regen::kernel::TessellationOptions;
use runmat_cad::value::{
    DynamicCall, DynamicContract, DynamicOutput, DynamicValueEvaluator, Quantity, ValueError,
    ValueXref,
};
use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, HashMap};
use std::rc::Rc;

/// The display meshes the cache keeps, by their array sizes: a quarter of the browser host's
/// 512 MB memory ceiling (`-sMAXIMUM_MEMORY` in build-exact-cad-wasm.sh), leaving the rest to
/// OCCT and the compile in progress. The camera drone's displays take about 10 MB.
pub const BUDGET_BYTES: usize = 128 << 20;

/// A part file regenerated for an assembly: what the assembly reads of it.
pub struct AssemblyPart {
    /// The regeneration's diagnostics ([`super::regen_diagnostics`]).
    pub diagnostics: Vec<Diagnostic>,
    pub document: PartDocument,
    /// The display under the assembly tessellation, with its coarser levels.
    pub display: PartDisplay,
    pub publishes: BTreeMap<PublishName, PublishedGeometry>,
}

/// A configured occurrence's display ([`crate::configured`]).
pub struct ConfiguredPart {
    pub display: PartDisplay,
}

enum Entry {
    Assembly(Rc<AssemblyPart>),
    Configured(Rc<ConfiguredPart>),
}

struct Slot {
    entry: Entry,
    bytes: usize,
    used: u64,
}

#[derive(Default)]
struct Cache {
    slots: HashMap<String, Slot>,
    bytes: usize,
    clock: u64,
    hits: u64,
    misses: u64,
}

thread_local! {
    static CACHE: RefCell<Cache> = RefCell::new(Cache::default());
}

/// The cache's counters: entries, display bytes, hits and misses since the thread started.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheStats {
    pub entries: usize,
    pub bytes: usize,
    pub hits: u64,
    pub misses: u64,
}

pub fn stats() -> CacheStats {
    CACHE.with(|c| {
        let c = c.borrow();
        CacheStats { entries: c.slots.len(), bytes: c.bytes, hits: c.hits, misses: c.misses }
    })
}

/// Empties the cache (tests; a host that wants its memory back).
pub fn clear() {
    CACHE.with(|c| *c.borrow_mut() = Cache::default());
}

fn display_bytes(display: &PartDisplay) -> usize {
    let mut n = 0;
    for body in &display.bodies {
        for f in &body.faces {
            n += 4 * (f.positions.len() + f.triangles.len() + f.normals.len());
        }
        for e in &body.edges {
            n += 4 * e.points.len();
        }
    }
    for level in &display.levels {
        n += 4 * (level.positions.len() + level.triangles.len() + level.normals.len());
    }
    n + 1024
}

fn tessellation_key(t: &TessellationOptions) -> String {
    format!(
        "{:x}/{:x}/{:x}/{:?}",
        t.chord_tolerance.to_bits(),
        t.angle_tolerance.to_bits(),
        t.size_fraction.to_bits(),
        t.chord_max.map(f64::to_bits)
    )
}

/// A short name for the entry under `key`, for a caller that holds its display: equal keys give
/// equal digests, and any change to a key gives another (a 64-bit hash and the key's length).
pub fn digest(key: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    key.hash(&mut h);
    format!("{:016x}-{:x}", h.finish(), key.len())
}

/// The key of part `path` with `text`, regenerated for an assembly.
pub fn assembly_key(path: &str, text: &str, tessellation: &TessellationOptions) -> String {
    format!("assembly\0{path}\0{}\0{text}", tessellation_key(tessellation))
}

/// The key of part `path` (whose document is `document`) regenerated in configuration `key`.
pub fn configured_key(path: &str, document: &PartDocument, key: &ConfigKey) -> String {
    let document = serde_json::to_string(document).unwrap_or_default();
    let config = serde_json::to_string(key).unwrap_or_default();
    format!("configured\0{path}\0{config}\0{document}")
}

fn get(key: &str) -> Option<Entry> {
    CACHE.with(|c| {
        let mut c = c.borrow_mut();
        c.clock += 1;
        let now = c.clock;
        let found = c.slots.get_mut(key).map(|slot| {
            slot.used = now;
            match &slot.entry {
                Entry::Assembly(p) => Entry::Assembly(Rc::clone(p)),
                Entry::Configured(p) => Entry::Configured(Rc::clone(p)),
            }
        });
        if found.is_some() {
            c.hits += 1;
        } else {
            c.misses += 1;
        }
        found
    })
}

fn put(key: String, entry: Entry, bytes: usize) {
    CACHE.with(|c| {
        let mut c = c.borrow_mut();
        if bytes > BUDGET_BYTES {
            return;
        }
        while c.bytes + bytes > BUDGET_BYTES {
            let Some(oldest) = c.slots.iter().min_by_key(|(_, s)| s.used).map(|(k, _)| k.clone()) else { break };
            if let Some(slot) = c.slots.remove(&oldest) {
                c.bytes -= slot.bytes;
            }
        }
        c.clock += 1;
        let used = c.clock;
        if let Some(previous) = c.slots.insert(key, Slot { entry, bytes, used }) {
            c.bytes -= previous.bytes;
        }
        c.bytes += bytes;
    })
}

/// The assembly part under `key`, made by `make` when the cache doesn't hold it. `make` returns
/// the part and whether its regeneration asked for a dynamic value (such a part isn't kept).
pub fn assembly_part<E>(
    key: String,
    make: impl FnOnce() -> Result<(AssemblyPart, bool), E>,
) -> Result<Rc<AssemblyPart>, E> {
    if let Some(Entry::Assembly(p)) = get(&key) {
        return Ok(p);
    }
    let (part, dynamic) = make()?;
    let part = Rc::new(part);
    if !dynamic {
        let bytes = display_bytes(&part.display);
        put(key, Entry::Assembly(Rc::clone(&part)), bytes);
    }
    Ok(part)
}

/// The configured display under `key`, made by `make` when the cache doesn't hold it (see
/// [`assembly_part`]).
pub fn configured_part(key: String, make: impl FnOnce() -> (ConfiguredPart, bool)) -> Rc<ConfiguredPart> {
    if let Some(Entry::Configured(p)) = get(&key) {
        return p;
    }
    let (part, dynamic) = make();
    let part = Rc::new(part);
    if !dynamic {
        let bytes = display_bytes(&part.display);
        put(key, Entry::Configured(Rc::clone(&part)), bytes);
    }
    part
}

/// An evaluator that notes whether the regeneration asked it for anything.
pub struct Watched<'a> {
    inner: &'a dyn DynamicValueEvaluator,
    used: &'a Cell<bool>,
}

impl<'a> Watched<'a> {
    pub fn new(inner: &'a dyn DynamicValueEvaluator, used: &'a Cell<bool>) -> Self {
        Self { inner, used }
    }
}

impl DynamicValueEvaluator for Watched<'_> {
    fn evaluate(&self, call: &DynamicCall, arguments: &[(String, Quantity)]) -> Result<Quantity, String> {
        self.used.set(true);
        self.inner.evaluate(call, arguments)
    }
    fn digest(&self, call: &DynamicCall) -> String {
        self.used.set(true);
        self.inner.digest(call)
    }
    fn contract(&self, call: &DynamicCall) -> DynamicContract {
        self.used.set(true);
        self.inner.contract(call)
    }
    fn evaluate_output(&self, call: &DynamicCall, arguments: &[(String, Quantity)]) -> Result<DynamicOutput, String> {
        self.used.set(true);
        self.inner.evaluate_output(call, arguments)
    }
    fn snapshot(&self, key: &str) -> Option<Quantity> {
        self.used.set(true);
        self.inner.snapshot(key)
    }
    fn reference(&self, path: &[String]) -> Option<Result<Quantity, String>> {
        self.used.set(true);
        self.inner.reference(path)
    }
    fn published(&self, xref: &ValueXref) -> Option<Result<Quantity, ValueError>> {
        self.used.set(true);
        self.inner.published(xref)
    }
}

#[cfg(test)]
mod tests {
    use super::{clear, stats};
    use crate::runtime::{compile_json, Files};
    use serde_json::Value;

    fn block(name: &str, width_mm: u32) -> String {
        format!("runmat 0.7\nstd 1\n\npart {name} {{\n    input width : length = {width_mm} mm\n    sketch s on origin.xy {{\n        l1 = line(start = [0 mm, 0 mm], end = [input.width, 0 mm])\n        l2 = line(start = l1.end, end = [input.width, 10 mm])\n        l3 = line(start = l2.end, end = [0 mm, 10 mm])\n        l4 = line(start = l3.end, end = l1.start)\n    }}\n    feature box = extrude(profile = sketch.s, distance = 5 mm)\n    output {name} = {{ body = feature.box.body }}\n}}\n")
    }

    const RIG: &str = "runmat 0.7\nstd 1\n\nassembly rig {\n    use part a as a\n    use part b as b {\n        placement = { placement = offset, offset = { translation = [0 mm, 50 mm, 0 mm] } }\n    }\n    use part b as wide {\n        configuration = { select = inputs, values = { width = 40 mm } }\n        placement = { placement = offset, offset = { translation = [0 mm, 100 mm, 0 mm] } }\n    }\n}\n";

    fn files(a: &str, b: &str) -> Files {
        [
            ("/p/a.part".to_owned(), a.to_owned()),
            ("/p/b.part".to_owned(), b.to_owned()),
            ("/p/rig.assembly".to_owned(), RIG.to_owned()),
        ]
        .into()
    }

    fn compile(files: &Files) -> Value {
        let out: Value = serde_json::from_str(&compile_json("/p/rig.assembly", RIG, files)).unwrap();
        assert!(out.get("projection").is_some(), "{}", out["diagnostics"]);
        out
    }

    /// The x extent of each prototype's bounds (metres), by prototype order.
    fn widths(out: &Value) -> Vec<f64> {
        out["display"]["data"]["prototypes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| {
                let b = &p["part"]["bounds"];
                b["max"][0].as_f64().unwrap() - b["min"][0].as_f64().unwrap()
            })
            .collect()
    }

    /// An unchanged recompile regenerates nothing and gives the same output.
    #[test]
    fn an_unchanged_recompile_reuses_every_part() {
        clear();
        let files = files(&block("a", 20), &block("b", 30));
        let first = compile(&files);
        let after_first = stats();
        // a, b, and b in the configuration `wide` uses.
        assert_eq!((after_first.entries, after_first.hits, after_first.misses), (3, 0, 3));
        let second = compile(&files);
        let after_second = stats();
        assert_eq!((after_second.entries, after_second.hits, after_second.misses), (3, 3, 3));
        assert_eq!(first, second);
    }

    /// A changed part is regenerated, and only it; the configured occurrence of an unchanged
    /// part is reused, and a changed configuration is a new regeneration.
    #[test]
    fn a_change_regenerates_exactly_what_it_changes() {
        clear();
        let before = compile(&files(&block("a", 20), &block("b", 30)));
        let base = stats();
        let edited = compile(&files(&block("a", 25), &block("b", 30)));
        let now = stats();
        assert_eq!(now.misses - base.misses, 1, "only a regenerates");
        assert_eq!(now.hits - base.hits, 2, "b and b in `wide` are reused");
        let (w0, w1) = (widths(&before), widths(&edited));
        assert!(w0.iter().any(|w| (w - 0.020).abs() < 1e-6), "{w0:?}");
        assert!(w1.iter().any(|w| (w - 0.025).abs() < 1e-6), "a draws its new width: {w1:?}");
        assert!(!w1.iter().any(|w| (w - 0.020).abs() < 1e-6), "no stale a: {w1:?}");

        // b's text changes: b and its configured occurrence both regenerate, a is reused.
        let base = stats();
        let out = compile(&files(&block("a", 25), &block("b", 32)));
        let now = stats();
        assert_eq!((now.misses - base.misses, now.hits - base.hits), (2, 1));
        let w = widths(&out);
        assert!(w.iter().any(|x| (x - 0.032).abs() < 1e-6) && w.iter().any(|x| (x - 0.040).abs() < 1e-6), "{w:?}");
    }

    /// A part whose values call `.m` functions depends on more than its text: it isn't kept.
    #[test]
    fn a_part_with_dynamic_values_is_never_kept() {
        clear();
        let dynamic = "runmat 0.7\nstd 1\n\npart a {\n    let r = radiusFor(size = 3 mm) as m\n    sketch s on origin.xy {\n        c = circle(center = [0 mm, 0 mm], radius = 4 mm)\n    }\n    feature disc = extrude(profile = sketch.s, distance = r)\n    output a = { body = feature.disc.body }\n}\n";
        let files = files(dynamic, &block("b", 30));
        let _ = compile_json("/p/rig.assembly", RIG, &files);
        let _ = compile_json("/p/rig.assembly", RIG, &files);
        let s = stats();
        // b and its configured occurrence are kept; a is regenerated each time.
        assert_eq!(s.entries, 2);
        assert_eq!(s.misses, 4, "{s:?}");
    }
}
