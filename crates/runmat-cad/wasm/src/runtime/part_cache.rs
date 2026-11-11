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
//! Circuit boards' 3D models are kept the same way, keyed on the layout and every circuit file
//! it reads ([`board_key`]).
//!
//! The cache holds at most [`BUDGET_BYTES`] of display meshes; the least recently used entries
//! go first. Entries are held by [`digest`] of their key.
//!
//! Entries also outlive the host. [`take_fresh`] writes each entry regenerated since the last
//! take as one [`entry_frame`] (its JSON with the meshes as bytes, [`runmat_cad::display::wire`]),
//! and [`seed`] reads such frames back in. The browser keeps them by digest (the examples ship
//! theirs as static files, and the page stores what it regenerates), and before a compile it
//! runs a [`probe`]: the compile with every entry the cache doesn't hold left out and listed, so
//! the page can fetch those entries and seed them first. A digest names the key exactly, so an
//! edited file is a new digest and regenerates; the page keeps stored entries apart per host
//! build, since another build may regenerate differently.

use super::Diagnostic;
use runmat_cad::display::wire;
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
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::rc::Rc;

/// The display meshes the cache keeps, by their array sizes: a quarter of the browser host's
/// 512 MB memory ceiling (`-sMAXIMUM_MEMORY` in build-exact-cad-wasm.sh), leaving the rest to
/// OCCT and the compile in progress. The camera drone's displays take about 10 MB.
pub const BUDGET_BYTES: usize = 128 << 20;

/// A part file regenerated for an assembly: what the assembly reads of it.
#[derive(Serialize, Deserialize)]
pub struct AssemblyPart {
    /// The regeneration's diagnostics ([`super::regen_diagnostics`]).
    pub diagnostics: Vec<Diagnostic>,
    pub document: PartDocument,
    /// The display under the assembly tessellation, with its coarser levels.
    pub display: PartDisplay,
    pub publishes: BTreeMap<PublishName, PublishedGeometry>,
}

/// A configured occurrence's display ([`crate::configured`]).
#[derive(Serialize, Deserialize)]
pub struct ConfiguredPart {
    pub display: PartDisplay,
}

enum Entry {
    Assembly(Rc<AssemblyPart>),
    Configured(Rc<ConfiguredPart>),
    Board(Rc<PartDisplay>),
    /// A drawing view's projected lines (`drawing::progress`).
    View(Rc<serde_json::Value>),
}

struct Slot {
    entry: Entry,
    bytes: usize,
    used: u64,
}

#[derive(Default)]
struct Cache {
    /// By [`digest`] of the key.
    slots: HashMap<String, Slot>,
    /// Digests regenerated (not seeded) since the last [`take_fresh`].
    fresh: Vec<String>,
    bytes: usize,
    clock: u64,
    hits: u64,
    misses: u64,
}

/// A [`probe`] in progress: the digests it may regenerate, and those it left out.
struct Probe {
    allow: HashSet<String>,
    misses: Vec<String>,
}

thread_local! {
    static CACHE: RefCell<Cache> = RefCell::new(Cache::default());
    static PROBE: RefCell<Option<Probe>> = const { RefCell::new(None) };
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

/// What a drawing view's digest starts with (`drawing::progress::view_key`). The page fetches a
/// view's entry from its stores like any other, but never asks a helper host to make one: only
/// a compile that projects the view makes it.
pub const VIEW_DIGEST_PREFIX: &str = "view-";

/// A short name for the entry under `key`, for a caller that holds its display: equal keys give
/// equal digests, and any change to a key gives another (a 64-bit hash and the key's length).
/// A view's digest starts with [`VIEW_DIGEST_PREFIX`].
pub fn digest(key: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    key.hash(&mut h);
    let kind = if key.starts_with("view\0") { VIEW_DIGEST_PREFIX } else { "" };
    format!("{kind}{:016x}-{:x}", h.finish(), key.len())
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

/// The key of circuit board `file`'s 3D model: its layout's text and every circuit file the
/// layout reads, by path. A board's compile reads nothing else.
pub fn board_key(file: &str, layout: &str, text: &str, circuit_files: &BTreeMap<String, String>) -> String {
    let mut key = format!("board\0{file}\0{layout}\0{text}");
    for (path, content) in circuit_files {
        key.push('\0');
        key.push_str(path);
        key.push('\0');
        key.push_str(content);
    }
    key
}

fn get(key: &str) -> Option<Entry> {
    let key = digest(key);
    let key = key.as_str();
    CACHE.with(|c| {
        let mut c = c.borrow_mut();
        c.clock += 1;
        let now = c.clock;
        let found = c.slots.get_mut(key).map(|slot| {
            slot.used = now;
            match &slot.entry {
                Entry::Assembly(p) => Entry::Assembly(Rc::clone(p)),
                Entry::Configured(p) => Entry::Configured(Rc::clone(p)),
                Entry::Board(p) => Entry::Board(Rc::clone(p)),
                Entry::View(p) => Entry::View(Rc::clone(p)),
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

/// Keeps a regenerated entry under its key.
fn put(key: String, entry: Entry, bytes: usize) {
    let key = digest(&key);
    if insert(key.clone(), entry, bytes) {
        CACHE.with(|c| c.borrow_mut().fresh.push(key));
    }
}

/// Holds `entry` under `digest`, evicting the least recently used; false when it is too large.
fn insert(key: String, entry: Entry, bytes: usize) -> bool {
    CACHE.with(|c| {
        let mut c = c.borrow_mut();
        if bytes > BUDGET_BYTES {
            return false;
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
        true
    })
}

/// Runs `run` (a compile) as a probe: an entry the cache doesn't hold is regenerated only when
/// its digest is in `allow`; any other is left out ([`skip`]) and listed. Returns `run`'s result
/// and the digests it left out, each once, in the order met.
pub fn probe<T>(allow: HashSet<String>, run: impl FnOnce() -> T) -> (T, Vec<String>) {
    struct Restore(Option<Probe>);
    impl Drop for Restore {
        fn drop(&mut self) {
            let previous = self.0.take();
            PROBE.with(|p| *p.borrow_mut() = previous);
        }
    }
    let restore = Restore(PROBE.with(|p| p.borrow_mut().replace(Probe { allow, misses: Vec::new() })));
    let out = run();
    let misses = PROBE.with(|p| p.borrow_mut().take()).map(|p| p.misses).unwrap_or_default();
    drop(restore);
    (out, misses)
}

/// Whether a compile should leave out the entry under `key`: only during a [`probe`], when the
/// cache doesn't hold it and the probe may not regenerate it (the digest is then listed).
pub fn skip(key: &str) -> bool {
    PROBE.with(|p| {
        let mut p = p.borrow_mut();
        let Some(probe) = p.as_mut() else { return false };
        let d = digest(key);
        if probe.allow.contains(&d) || CACHE.with(|c| c.borrow().slots.contains_key(&d)) {
            return false;
        }
        if !probe.misses.contains(&d) {
            probe.misses.push(d);
        }
        true
    })
}

/// An entry as stored outside the host: the digest it is held by and exactly one of its kinds.
#[derive(Serialize)]
struct StoredRef<'a> {
    digest: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    assembly: Option<&'a AssemblyPart>,
    #[serde(skip_serializing_if = "Option::is_none")]
    configured: Option<&'a ConfiguredPart>,
    #[serde(skip_serializing_if = "Option::is_none")]
    board: Option<&'a PartDisplay>,
    #[serde(skip_serializing_if = "Option::is_none")]
    view: Option<&'a serde_json::Value>,
}

/// [`StoredRef`] read back.
#[derive(Deserialize)]
struct Stored {
    digest: String,
    assembly: Option<AssemblyPart>,
    configured: Option<ConfiguredPart>,
    board: Option<PartDisplay>,
    view: Option<serde_json::Value>,
}

/// One stored entry: the JSON's length and the bytes' length (`u32`, little-endian), the JSON,
/// zero padding to a multiple of 4, then the mesh bytes its references point into. A frame's
/// length is a multiple of 4, so frames concatenate.
pub fn entry_frame(json: &str, bytes: &[u8]) -> Vec<u8> {
    let padded = json.len().div_ceil(4) * 4;
    let mut out = Vec::with_capacity(8 + padded + bytes.len().div_ceil(4) * 4);
    out.extend_from_slice(&(json.len() as u32).to_le_bytes());
    out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(json.as_bytes());
    out.resize(8 + padded, 0);
    out.extend_from_slice(bytes);
    out.resize(out.len().div_ceil(4) * 4, 0);
    out
}

/// Each entry regenerated since the last call that the cache still holds, as `(digest, frame)`;
/// the list starts over.
pub fn take_fresh() -> Vec<(String, Vec<u8>)> {
    let fresh = CACHE.with(|c| std::mem::take(&mut c.borrow_mut().fresh));
    let mut out = Vec::new();
    for d in fresh {
        let Some(frame) = CACHE.with(|c| {
            let c = c.borrow();
            let slot = c.slots.get(&d)?;
            let (json, bytes) = wire::collect(|| {
                let none = StoredRef { digest: &d, assembly: None, configured: None, board: None, view: None };
                let stored = match &slot.entry {
                    Entry::Assembly(p) => StoredRef { assembly: Some(&**p), ..none },
                    Entry::Configured(p) => StoredRef { configured: Some(&**p), ..none },
                    Entry::Board(p) => StoredRef { board: Some(&**p), ..none },
                    Entry::View(p) => StoredRef { view: Some(&**p), ..none },
                };
                serde_json::to_string(&stored)
            });
            Some(entry_frame(&json.ok()?, &bytes))
        }) else {
            continue;
        };
        if !out.iter().any(|(seen, _)| *seen == d) {
            out.push((d, frame));
        }
    }
    out
}

/// Holds every entry in `frames` (concatenated [`entry_frame`]s); returns how many it took. A
/// frame that doesn't read (cut short, or written by another build) is passed over.
pub fn seed(mut frames: &[u8]) -> usize {
    let mut taken = 0;
    while frames.len() >= 8 {
        let word = |at: usize| u32::from_le_bytes([frames[at], frames[at + 1], frames[at + 2], frames[at + 3]]) as usize;
        let (json_len, bytes_len) = (word(0), word(4));
        let start = 8 + json_len.div_ceil(4) * 4;
        let end = start + bytes_len;
        if end > frames.len() {
            break;
        }
        let json = &frames[8..8 + json_len];
        let bytes = &frames[start..end];
        let stored: Option<Stored> =
            wire::revive(bytes, || serde_json::from_slice(json).ok());
        if let Some(stored) = stored {
            let entry = match (stored.assembly, stored.configured, stored.board, stored.view) {
                (Some(p), None, None, None) => Some((display_bytes(&p.display), Entry::Assembly(Rc::new(p)))),
                (None, Some(p), None, None) => Some((display_bytes(&p.display), Entry::Configured(Rc::new(p)))),
                (None, None, Some(d), None) => Some((display_bytes(&d), Entry::Board(Rc::new(d)))),
                (None, None, None, Some(v)) => Some((json_size(&v), Entry::View(Rc::new(v)))),
                _ => None,
            };
            if let Some((size, entry)) = entry {
                if insert(stored.digest, entry, size) {
                    taken += 1;
                }
            }
        }
        frames = &frames[(end.div_ceil(4) * 4).min(frames.len())..];
    }
    taken
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

/// Circuit board `key`'s 3D model, made by `make` when the cache doesn't hold it.
pub fn board_part<E>(key: String, make: impl FnOnce() -> Result<PartDisplay, E>) -> Result<PartDisplay, E> {
    if let Some(Entry::Board(p)) = get(&key) {
        return Ok((*p).clone());
    }
    let display = make()?;
    let bytes = display_bytes(&display);
    put(key, Entry::Board(Rc::new(display.clone())), bytes);
    Ok(display)
}

/// A drawing view's projected lines under `key` (`drawing::progress::view_key`), made by `make`
/// when the cache doesn't hold them. `None` when a [`probe`] leaves them out.
pub fn view_lines(key: String, make: impl FnOnce() -> serde_json::Value) -> Option<serde_json::Value> {
    if let Some(Entry::View(v)) = get(&key) {
        return Some((*v).clone());
    }
    if skip(&key) {
        return None;
    }
    let lines = make();
    let bytes = json_size(&lines);
    put(key, Entry::View(Rc::new(lines.clone())), bytes);
    Some(lines)
}

/// Whether the cache holds the entry under `key`.
pub fn holds(key: &str) -> bool {
    let d = digest(key);
    CACHE.with(|c| c.borrow().slots.contains_key(&d))
}

/// Lists `key` as a miss when a [`probe`] runs and the cache doesn't hold it, so the page fetches
/// the entry with the others (a drawing's views that this stage doesn't project yet).
pub fn want(key: &str) {
    let _ = skip(key);
}

/// A JSON value's size as text, for the cache's budget.
fn json_size(value: &serde_json::Value) -> usize {
    serde_json::to_string(value).map_or(1024, |t| t.len() + 64)
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
    use super::{clear, probe, seed, stats, take_fresh};
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

    /// Stored entries read back in a fresh cache give the same compile, regenerating nothing; a
    /// probe of an empty cache regenerates nothing and lists every entry it left out.
    #[test]
    fn stored_entries_seed_a_fresh_cache_exactly() {
        clear();
        let files = files(&block("a", 20), &block("b", 30));
        let first = compile(&files);
        let fresh = take_fresh();
        assert_eq!(fresh.len(), 3, "a, b and b in `wide`");
        assert!(take_fresh().is_empty(), "the list starts over");

        clear();
        let (_, misses) = probe(Default::default(), || compile_json("/p/rig.assembly", RIG, &files));
        assert_eq!(stats().misses, 0, "a probe regenerates nothing");
        // `wide`'s configured digest needs b's document, so only a and b show before seeding.
        assert_eq!(misses.len(), 2, "{misses:?}");
        assert!(misses.iter().all(|d| fresh.iter().any(|(f, _)| f == d)), "{misses:?}");

        let all: Vec<u8> = fresh.iter().flat_map(|(_, f)| f.iter().copied()).collect();
        assert_eq!(seed(&all), 3);
        assert!(take_fresh().is_empty(), "seeded entries aren't fresh");
        let (out, misses) = probe(Default::default(), || compile(&files));
        assert!(misses.is_empty(), "{misses:?}");
        let s = stats();
        assert_eq!((s.hits, s.misses), (3, 0));
        assert_eq!(out, first);

        // A cut-short or foreign frame is passed over.
        clear();
        assert_eq!(seed(&all[..all.len() - 8]), 2);
        assert_eq!(seed(b"not a frame at all"), 0);
    }
}
