//! The framed compile (`runmat_cad_call_framed`), the one the browser worker calls: the display's
//! mesh arrays come back as bytes beside the JSON, and each reference reads back exactly the
//! values the JSON compile writes.

use runmat_geometry_wasm_host::cad;
use serde_json::Value;
use std::ffi::CString;

const BRACKET: &str = include_str!("../../../runmat-cad/tests/corpus/documents/bracket.part");

/// Replaces each `{"$f32": [offset, count]}` / `{"$u32": …}` in `v` with the numbers it names.
fn inline(v: &mut Value, bytes: &[u8]) {
    match v {
        Value::Object(map) => {
            let reference = if map.len() == 1 { map.iter().next().map(|(k, at)| (k.clone(), at.clone())) } else { None };
            if let Some((tag, at)) = reference.filter(|(k, _)| k == "$f32" || k == "$u32") {
                let (offset, count) = (at[0].as_u64().unwrap() as usize, at[1].as_u64().unwrap() as usize);
                let word = |i: usize| <[u8; 4]>::try_from(&bytes[offset + 4 * i..offset + 4 * i + 4]).unwrap();
                *v = Value::Array(
                    (0..count)
                        .map(|i| if tag == "$f32" { serde_json::to_value(f32::from_le_bytes(word(i))).unwrap() } else { Value::from(u32::from_le_bytes(word(i))) })
                        .collect(),
                );
                return;
            }
            for child in map.values_mut() {
                inline(child, bytes);
            }
        }
        Value::Array(items) => items.iter_mut().for_each(|c| inline(c, bytes)),
        _ => {}
    }
}

/// `a` and `b` agree, numbers compared as the `f32` the display stores.
fn same(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => x.as_f64().map(|v| v as f32) == y.as_f64().map(|v| v as f32),
        (Value::Array(x), Value::Array(y)) => x.len() == y.len() && x.iter().zip(y).all(|(p, q)| same(p, q)),
        (Value::Object(x), Value::Object(y)) => x.len() == y.len() && x.iter().all(|(k, v)| y.get(k).is_some_and(|w| same(v, w))),
        _ => a == b,
    }
}

#[test]
fn the_framed_compile_sends_mesh_arrays_as_bytes_that_read_back_exactly() {
    let request = serde_json::json!({ "fn": "compile", "args": ["/p/bracket.part", BRACKET, "{}", "{}"] }).to_string();
    let plain: Value = serde_json::from_str(&cad::call(&request)).unwrap();
    let (json, bytes) = cad::call_framed(&request);
    assert!(json.contains("\"$f32\"") && json.contains("\"$u32\""), "the meshes travel as references");
    assert!(json.len() < serde_json::to_string(&plain).unwrap().len(), "the JSON is smaller: {} bytes", json.len());
    let mut framed: Value = serde_json::from_str(&json).unwrap();
    inline(&mut framed, &bytes);
    assert!(same(&framed, &plain), "the framed answer reads back as the JSON one");

    // The C entry: one frame, [json length][bytes length][json][padding][bytes], freed once.
    let c = CString::new(request).unwrap();
    unsafe {
        let frame = cad::runmat_cad_call_framed(c.as_ptr());
        let header = std::slice::from_raw_parts(frame, 8);
        let json_len = u32::from_le_bytes(header[0..4].try_into().unwrap()) as usize;
        let bytes_len = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;
        assert_eq!((json_len, bytes_len), (json.len(), bytes.len()));
        let text = std::slice::from_raw_parts(frame.add(8), json_len);
        assert_eq!(std::str::from_utf8(text).unwrap(), json);
        let at = 8 + json_len.div_ceil(4) * 4;
        assert_eq!(std::slice::from_raw_parts(frame.add(at), bytes_len), &bytes[..]);
        cad::runmat_cad_free_framed(frame);
    }
}

/// The cache's travel through the host: an assembly's regenerated entries come out of
/// `cacheFresh`, a probe of an emptied cache lists them and regenerates nothing, and once they
/// are seeded through `runmat_cad_seed` the probe leaves nothing out and answers as the first
/// compile did.
#[test]
fn stored_part_entries_seed_the_host_and_its_probe_finds_nothing_missing() {
    use runmat_cad_wasm::runtime::part_cache;
    let rig = "runmat 0.7\nstd 1\n\nassembly rig {\n    use part bracket as a\n}\n";
    let files = serde_json::json!({ "/p/bracket.part": BRACKET, "/p/rig.assembly": rig }).to_string();
    let compile = serde_json::json!({ "fn": "compile", "args": ["/p/rig.assembly", rig, files, "{}"] }).to_string();
    let probe = serde_json::json!({ "fn": "probe", "args": ["/p/rig.assembly", rig, files, "{}", "[]"] }).to_string();
    let misses = || -> Vec<String> { serde_json::from_str(&cad::call(r#"{"fn":"cacheMisses","args":[]}"#)).unwrap() };

    part_cache::clear();
    let first = cad::call(&compile);
    let (index, frames) = cad::call_framed(r#"{"fn":"cacheFresh","args":[]}"#);
    let index: Value = serde_json::from_str(&index).unwrap();
    let digests: Vec<&str> = index.as_array().unwrap().iter().map(|e| e["digest"].as_str().unwrap()).collect();
    assert_eq!(digests.len(), 1, "{index}");

    part_cache::clear();
    cad::call(&probe);
    assert_eq!(misses(), digests);
    assert_eq!(part_cache::stats().misses, 0, "the probe regenerated nothing");

    assert_eq!(unsafe { cad::runmat_cad_seed(frames.as_ptr(), frames.len()) }, 1);
    let answer = cad::call(&probe);
    assert!(misses().is_empty());
    assert_eq!(answer, first);
}
