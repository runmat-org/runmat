//! The change from one projection to the next, for a client that holds the last one.
//!
//! After a small edit, most of a 50k-primitive board's projection is unchanged. [`index_delta`]
//! names only what changed, so the worker sends kilobytes instead of the whole projection:
//!
//! - `fields`: each top-level field of [`CircuitIndex`] that changed (other than `board`), whole.
//! - `board.fields`: each board field that changed and has no row patch, whole.
//! - `board.rows.<pads|tracks|vias|zones|footprints>`: either `{ "all": [rows] }`, or
//!   `{ "set": [rows], "remove": [keys] }`. A client applies `set` by key: a row whose key it
//!   holds replaces that row where it is, and a new key goes at the end, in `set` order. Keys
//!   are a pad's `id`, a copper row's `id` and a footprint's `reference`.
//! - `board.rows.<drc|unrouted>` (rows without a key): `{ "all": [rows] }`, or `{ "runs": [op] }`
//!   where each op is `[start, length]` (that many old rows from `start`) or a new row, in order.
//!
//! A patch is sent only when applying it gives the new rows in the new order exactly; otherwise
//! that array goes whole. A board that appears or goes away goes whole as a field, so applying the
//! delta to the old projection always gives the new one ([`apply_index_delta`] does it here; the
//! Desktop's `circuit-delta.ts` does it on the client).

use crate::index::{BoardIndex, CircuitIndex, CopperRow, EcoIndex, FootprintRow, PadRow};
use serde::Serialize;
use serde_json::{json, Map, Value};
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

/// What changed from `old` to `new`, as described in the module docs.
pub fn index_delta(old: &CircuitIndex, new: &CircuitIndex) -> Value {
    // Every field is compared below: a new field fails to compile here until it is.
    let CircuitIndex { format_version: _, design: _, variant: _, variants: _, hierarchy: _, components: _, nets: _, net_classes: _, bom: _, eco: _, back_annotation: _, last_eco: _, erc: _, review: _, schematic: _, board: _, simulations: _ } = new;
    let mut fields = Map::new();
    let mut put = |name: &str, changed: bool, value: Value| {
        if changed {
            fields.insert(name.to_owned(), value);
        }
    };
    macro_rules! field {
        ($($name:ident),*) => {
            $(put(stringify!($name), old.$name != new.$name, to_value(&new.$name));)*
        };
    }
    field!(format_version, design, variant, variants, hierarchy, components, nets, net_classes, bom, eco, back_annotation, last_eco, erc, review, schematic, simulations);
    let mut out = Map::new();
    out.insert("fields".into(), Value::Object(fields));
    match (&old.board, &new.board) {
        (Some(a), Some(b)) => {
            out.insert("board".into(), board_delta(a, b));
        }
        (None, None) => {}
        (_, b) => {
            if let Value::Object(f) = &mut out["fields"] {
                f.insert("board".into(), to_value(b));
            }
        }
    }
    Value::Object(out)
}

fn to_value<T: Serialize>(v: &T) -> Value {
    serde_json::to_value(v).unwrap_or(Value::Null)
}

fn board_delta(old: &BoardIndex, new: &BoardIndex) -> Value {
    let mut out = small_fields(old, new);
    let rows = out["rows"].as_object_mut().expect("rows");
    let mut patch = |name: &str, p: Option<Value>| {
        if let Some(p) = p {
            rows.insert(name.to_owned(), p);
        }
    };
    patch("pads", row_patch(&old.pads, &new.pads, |r| r.id.clone()));
    patch("tracks", row_patch(&old.tracks, &new.tracks, |r| r.id));
    patch("vias", row_patch(&old.vias, &new.vias, |r| r.id));
    patch("zones", row_patch(&old.zones, &new.zones, |r| r.id));
    patch("footprints", row_patch(&old.footprints, &new.footprints, |r| r.reference.clone()));
    out
}

/// The board's delta without its keyed tables: the whole fields and the keyless row runs.
fn small_fields(old: &BoardIndex, new: &BoardIndex) -> Value {
    // Every field is compared here or in a keyed patch: a new field fails to compile until it is.
    let BoardIndex { name: _, layers: _, footprints: _, pads: _, tracks: _, vias: _, zones: _, rules: _, drc: _, rooms: _, outline: _, keepouts: _, texts: _, graphics: _, dimensions: _, fiducials: _, unrouted: _, nets: _, controlled: _, length_groups: _ } = new;
    let mut fields = Map::new();
    let mut put = |name: &str, changed: bool, value: Value| {
        if changed {
            fields.insert(name.to_owned(), value);
        }
    };
    macro_rules! field {
        ($($name:ident),*) => {
            $(put(stringify!($name), old.$name != new.$name, to_value(&new.$name));)*
        };
    }
    field!(name, layers, rules, rooms, outline, keepouts, texts, graphics, dimensions, fiducials, nets, controlled, length_groups);
    let mut rows = Map::new();
    if let Some(p) = runs_patch(&old.drc, &new.drc) {
        rows.insert("drc".into(), p);
    }
    if let Some(p) = runs_patch(&old.unrouted, &new.unrouted) {
        rows.insert("unrouted".into(), p);
    }
    json!({ "fields": fields, "rows": rows })
}

/// What a board patch leaves of the projection before it, for [`patched_delta`].
///
/// A patch rebuilds the rows of the items it changed and moves every other row over from the
/// old projection unchanged. So the delta needs only each keyed table's old keys, the old rows
/// of the rebuilt items, and the old values of what the patch rebuilds whole (the board's other
/// fields, DRC, unrouted links and both ECO directions); nothing else is copied or compared.
pub struct PatchBase {
    pub keys: OldKeys,
    /// Old rows of the rebuilt items, by key.
    pub pads: HashMap<String, PadRow>,
    pub tracks: HashMap<u64, CopperRow>,
    pub vias: HashMap<u64, CopperRow>,
    pub zones: HashMap<u64, CopperRow>,
    pub footprints: HashMap<String, FootprintRow>,
    /// The old board with its keyed tables emptied.
    pub board: BoardIndex,
    pub eco: EcoIndex,
    pub back_annotation: EcoIndex,
}

/// Each keyed board table's keys, in order.
#[derive(Default)]
pub struct OldKeys {
    pub pads: Vec<String>,
    pub tracks: Vec<u64>,
    pub vias: Vec<u64>,
    pub zones: Vec<u64>,
    pub footprints: Vec<String>,
}

impl PatchBase {
    /// Takes `old` apart before a patch rebuilds it: `rebuilt` says whether a track, via or
    /// pour id is rebuilt, `rebuilt_ref` whether a footprint (and its pads) is. The keyed
    /// tables stay in `old` for the patch to reuse; the rest of the board moves here.
    pub fn take(index: &mut CircuitIndex, rebuilt: impl Fn(u64) -> bool, rebuilt_ref: impl Fn(&str) -> bool) -> Option<PatchBase> {
        let old = index.board.as_mut()?;
        let rows = |rows: &[CopperRow]| -> HashMap<u64, CopperRow> { rows.iter().filter(|r| rebuilt(r.id)).map(|r| (r.id, r.clone())).collect() };
        let base = PatchBase {
            keys: OldKeys {
                pads: old.pads.iter().map(|r| r.id.clone()).collect(),
                tracks: old.tracks.iter().map(|r| r.id).collect(),
                vias: old.vias.iter().map(|r| r.id).collect(),
                zones: old.zones.iter().map(|r| r.id).collect(),
                footprints: old.footprints.iter().map(|r| r.reference.clone()).collect(),
            },
            pads: old.pads.iter().filter(|r| rebuilt_ref(&r.component)).map(|r| (r.id.clone(), r.clone())).collect(),
            tracks: rows(&old.tracks),
            vias: rows(&old.vias),
            zones: rows(&old.zones),
            footprints: old.footprints.iter().filter(|r| rebuilt_ref(&r.reference)).map(|r| (r.reference.clone(), r.clone())).collect(),
            board: BoardIndex {
                name: std::mem::take(&mut old.name),
                layers: std::mem::take(&mut old.layers),
                footprints: Vec::new(),
                pads: Vec::new(),
                tracks: Vec::new(),
                vias: Vec::new(),
                zones: Vec::new(),
                rules: std::mem::take(&mut old.rules),
                drc: std::mem::take(&mut old.drc),
                rooms: std::mem::take(&mut old.rooms),
                outline: std::mem::take(&mut old.outline),
                keepouts: std::mem::take(&mut old.keepouts),
                texts: std::mem::take(&mut old.texts),
                graphics: std::mem::take(&mut old.graphics),
                dimensions: std::mem::take(&mut old.dimensions),
                fiducials: std::mem::take(&mut old.fiducials),
                // The patch reuses the old links of the nets it didn't touch.
                unrouted: old.unrouted.clone(),
                nets: std::mem::take(&mut old.nets),
                controlled: std::mem::take(&mut old.controlled),
                length_groups: std::mem::take(&mut old.length_groups),
            },
            eco: std::mem::take(&mut index.eco),
            back_annotation: std::mem::take(&mut index.back_annotation),
        };
        Some(base)
    }
}

/// [`index_delta`] from the projection `base` was taken from to `new`, the projection the patch
/// made: only the rebuilt rows and the rebuilt fields are compared.
pub fn patched_delta(base: &PatchBase, new: &CircuitIndex) -> Value {
    let mut fields = Map::new();
    if base.eco != new.eco {
        fields.insert("eco".into(), to_value(&new.eco));
    }
    if base.back_annotation != new.back_annotation {
        fields.insert("back_annotation".into(), to_value(&new.back_annotation));
    }
    let Some(b) = &new.board else {
        fields.insert("board".into(), Value::Null);
        return json!({ "fields": fields });
    };
    let mut board = small_fields(&base.board, b);
    let rows = board["rows"].as_object_mut().expect("rows");
    let mut patch = |name: &str, p: Option<Value>| {
        if let Some(p) = p {
            rows.insert(name.to_owned(), p);
        }
    };
    fn same<'a, K: Eq + Hash, T: PartialEq>(keys: &'a [K], old: &'a HashMap<K, T>) -> impl Fn(usize, &T) -> bool + 'a {
        move |i, r| old.get(&keys[i]).is_none_or(|o| o == r)
    }
    let k = &base.keys;
    patch("pads", keyed_patch(&k.pads, &b.pads, |r| r.id.clone(), same(&k.pads, &base.pads)));
    patch("tracks", keyed_patch(&k.tracks, &b.tracks, |r| r.id, same(&k.tracks, &base.tracks)));
    patch("vias", keyed_patch(&k.vias, &b.vias, |r| r.id, same(&k.vias, &base.vias)));
    patch("zones", keyed_patch(&k.zones, &b.zones, |r| r.id, same(&k.zones, &base.zones)));
    patch("footprints", keyed_patch(&k.footprints, &b.footprints, |r| r.reference.clone(), same(&k.footprints, &base.footprints)));
    json!({ "fields": fields, "board": board })
}

/// The patch from `old` rows to `new` rows by `key`, `None` when they are equal.
fn row_patch<T, K>(old: &[T], new: &[T], key: impl Fn(&T) -> K) -> Option<Value>
where
    T: PartialEq + Serialize,
    K: Eq + Hash + Clone + Serialize,
{
    if old == new {
        return None;
    }
    let keys: Vec<K> = old.iter().map(&key).collect();
    keyed_patch(&keys, new, key, |i, r| old[i] == *r)
}

/// The patch from rows with keys `old` to `new` rows, where `same(i, row)` says whether `row`
/// equals old row `i` (which has `row`'s key); `None` when nothing changed.
fn keyed_patch<T, K>(old: &[K], new: &[T], key: impl Fn(&T) -> K, same: impl Fn(usize, &T) -> bool) -> Option<Value>
where
    T: Serialize,
    K: Eq + Hash + Clone + Serialize,
{
    let whole = || Some(json!({ "all": to_value(&new) }));
    // The usual case after an edit: the same keys in the same places, a few rows changed.
    if old.len() == new.len() && old.iter().zip(new).all(|(a, b)| *a == key(b)) {
        let set: Vec<&T> = new.iter().enumerate().filter(|(i, b)| !same(*i, b)).map(|(_, b)| b).collect();
        if set.is_empty() {
            return None;
        }
        if set.len() * 2 > new.len().max(1) {
            return whole();
        }
        return Some(json!({ "set": set, "remove": Vec::<K>::new() }));
    }
    let old_at: HashMap<K, usize> = old.iter().enumerate().map(|(i, k)| (k.clone(), i)).collect();
    let new_keys: HashSet<K> = new.iter().map(&key).collect();
    if old_at.len() != old.len() || new_keys.len() != new.len() {
        return whole();
    }
    // The order the client would hold after the patch: the old keys still present, then the
    // new keys in their new order. It must be the new order exactly.
    let kept = old.iter().filter(|k| new_keys.contains(*k)).cloned();
    let added = new.iter().map(&key).filter(|k| !old_at.contains_key(k));
    if !kept.chain(added).eq(new.iter().map(&key)) {
        return whole();
    }
    let set: Vec<&T> = new
        .iter()
        .filter(|r| !matches!(old_at.get(&key(r)), Some(&i) if same(i, r)))
        .collect();
    let remove: Vec<K> = old.iter().filter(|k| !new_keys.contains(*k)).cloned().collect();
    // A patch larger than half the rows is sent whole: it is simpler to apply and no larger.
    if set.len() * 2 > new.len().max(1) {
        return whole();
    }
    Some(json!({ "set": set, "remove": remove }))
}

/// The patch from keyless `old` rows to `new` rows as copied runs and new rows, `None` when equal.
fn runs_patch<T: PartialEq + Serialize>(old: &[T], new: &[T]) -> Option<Value> {
    if old == new {
        return None;
    }
    // The rows both ends share are copied as they are; only the middle is matched by content.
    let prefix = old.iter().zip(new).take_while(|(a, b)| a == b).count();
    let suffix = old[prefix..].iter().rev().zip(new[prefix..].iter().rev()).take_while(|(a, b)| a == b).count();
    let (old_mid, new_mid) = (prefix..old.len() - suffix, &new[prefix..new.len() - suffix]);
    let mut first: HashMap<String, usize> = HashMap::new();
    for i in old_mid.clone().rev() {
        first.insert(serde_json::to_string(&old[i]).unwrap_or_default(), i);
    }
    let mut ops: Vec<Value> = Vec::new();
    let mut run: Option<(usize, usize)> = (prefix > 0).then_some((0, prefix));
    let mut literal = 0;
    let flush = |run: &mut Option<(usize, usize)>, ops: &mut Vec<Value>| {
        if let Some((start, len)) = run.take() {
            ops.push(json!([start, len]));
        }
    };
    for r in new_mid {
        if let Some((start, len)) = &mut run {
            if *start + *len < old.len() && old[*start + *len] == *r {
                *len += 1;
                continue;
            }
        }
        flush(&mut run, &mut ops);
        match first.get(&serde_json::to_string(r).unwrap_or_default()) {
            Some(&j) => run = Some((j, 1)),
            None => {
                literal += 1;
                ops.push(to_value(r));
            }
        }
    }
    flush(&mut run, &mut ops);
    if suffix > 0 {
        ops.push(json!([old.len() - suffix, suffix]));
    }
    if literal * 2 > new.len().max(1) {
        return Some(json!({ "all": to_value(&new) }));
    }
    Some(json!({ "runs": ops }))
}

/// `old` with `delta` applied, as a client applies it: the projection `index_delta` was made
/// from. For tests and for hosts that keep the projection in Rust.
pub fn apply_index_delta(old: &CircuitIndex, delta: &Value) -> Result<CircuitIndex, String> {
    let mut v = serde_json::to_value(old).map_err(|e| e.to_string())?;
    let obj = v.as_object_mut().ok_or("a projection is an object")?;
    if let Some(Value::Object(fields)) = delta.get("fields") {
        for (k, x) in fields {
            obj.insert(k.clone(), x.clone());
        }
    }
    if let Some(board) = delta.get("board") {
        let b = obj.get_mut("board").and_then(Value::as_object_mut).ok_or("the delta patches a board the projection doesn't have")?;
        if let Some(Value::Object(fields)) = board.get("fields") {
            for (k, x) in fields {
                b.insert(k.clone(), x.clone());
            }
        }
        if let Some(Value::Object(rows)) = board.get("rows") {
            for (name, p) in rows {
                let key = if name == "footprints" { "reference" } else { "id" };
                let list = b.entry(name.clone()).or_insert(Value::Array(Vec::new()));
                apply_rows(list, p, key)?;
            }
        }
    }
    serde_json::from_value(v).map_err(|e| e.to_string())
}

fn apply_rows(list: &mut Value, p: &Value, key: &str) -> Result<(), String> {
    if let Some(all) = p.get("all") {
        *list = all.clone();
        return Ok(());
    }
    if let Some(Value::Array(ops)) = p.get("runs") {
        let old = list.as_array().cloned().unwrap_or_default();
        let mut out = Vec::new();
        for op in ops {
            match op {
                Value::Array(r) => {
                    let (start, len) = (r[0].as_u64().unwrap_or(0) as usize, r[1].as_u64().unwrap_or(0) as usize);
                    out.extend_from_slice(old.get(start..start + len).ok_or("a run past the old rows")?);
                }
                row => out.push(row.clone()),
            }
        }
        *list = Value::Array(out);
        return Ok(());
    }
    let rows = list.as_array_mut().ok_or("rows are a list")?;
    let gone: HashSet<String> = p["remove"].as_array().into_iter().flatten().map(|k| k.to_string()).collect();
    rows.retain(|r| !gone.contains(&r[key].to_string()));
    let at: HashMap<String, usize> = rows.iter().enumerate().map(|(i, r)| (r[key].to_string(), i)).collect();
    for r in p["set"].as_array().into_iter().flatten() {
        match at.get(&r[key].to_string()) {
            Some(&i) => rows[i] = r.clone(),
            None => rows.push(r.clone()),
        }
    }
    Ok(())
}
