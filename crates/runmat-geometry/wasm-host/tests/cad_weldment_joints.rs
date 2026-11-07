//! The Weldments tab's joints from the canvas on OCCT: an end cap, a gusset, a weld bead and a
//! member extended to a plate's face, each written by the Desktop's edit and regenerated.

use serde_json::{json, Value};

const LAYOUT: &str = r#"runmat 0.7
std 1

part frame {
    sketch layout on origin.xy {
        bottom = line(start = [0 mm, 0 mm], end = [200 mm, 0 mm])
        right = line(start = [200 mm, 0 mm], end = [200 mm, 100 mm])
    }
    sketch w on origin.xy {
        l1 = line(start = [250 mm, -50 mm], end = [260 mm, -50 mm])
        l2 = line(start = [260 mm, -50 mm], end = [260 mm, 150 mm])
        l3 = line(start = [260 mm, 150 mm], end = [250 mm, 150 mm])
        l4 = line(start = [250 mm, 150 mm], end = [250 mm, -50 mm])
    }
    feature plate = extrude(profile = sketch.w, distance = 30 mm)
}
"#;

fn compile(text: &str) -> Value {
    let request = json!({ "fn": "compile", "args": ["/p/frame.part", text, "{}", "{}"] });
    serde_json::from_str(&runmat_geometry_wasm_host::cad::call(&request.to_string())).expect("json")
}

fn errors(v: &Value) -> Vec<Value> {
    v["diagnostics"].as_array().into_iter().flatten().filter(|d| d["severity"] == "error").cloned().collect()
}

/// The command's edit, then the compile of what it wrote; `None` when either fails.
fn apply(text: &str, command: &str, values: Value) -> Option<(String, Value)> {
    let request = json!({ "fn": "edit", "args": ["/p/frame.part", text, "{}", json!({ "command": command, "values": values }).to_string()] });
    let out: Value = serde_json::from_str(&runmat_geometry_wasm_host::cad::call(&request.to_string())).expect("json");
    if out["ok"] != true {
        return None;
    }
    let text = out["files"][0]["text"].as_str()?.to_owned();
    let v = compile(&text);
    errors(&v).is_empty().then_some((text, v))
}

fn bodies(v: &Value) -> Vec<Value> {
    v["display"]["data"]["bodies"].as_array().cloned().unwrap_or_default()
}

fn face_refs(v: &Value, contains: &str) -> Vec<String> {
    bodies(v)
        .iter()
        .flat_map(|b| b["faces"].as_array().cloned().unwrap_or_default())
        .filter_map(|f| f["reference"].as_str().map(str::to_owned))
        .filter(|r| r.contains(contains))
        .collect()
}

/// The largest x any edge point of the body holding face `face` reaches (metres).
fn max_x_of_body_with(v: &Value, face: &str) -> f64 {
    let body = bodies(v)
        .into_iter()
        .find(|b| b["faces"].as_array().is_some_and(|fs| fs.iter().any(|f| f["reference"] == face)))
        .expect("the body");
    body["edges"]
        .as_array()
        .into_iter()
        .flatten()
        .flat_map(|e| e["points"].as_array().cloned().unwrap_or_default().chunks(3).map(|p| p[0].as_f64().unwrap()).collect::<Vec<_>>())
        .fold(f64::NEG_INFINITY, f64::max)
}

#[test]
fn end_cap_gusset_weld_bead_and_extend_apply_from_the_canvas() {
    runmat_geometry_wasm_host::cad::install_occt();
    // The members, as Structural member writes them: one rectangular tube per layout line.
    let member = json!({ "profile": "rect_tube", "size": "20x20x2", "corner": "none", "segments": ["sketch:layout"] });
    let (frame, v) = apply(LAYOUT, "weldment.structural-member", member).expect("the members apply");
    let frame = frame.as_str();
    let start = bodies(&v).len();
    let ends = face_refs(&v, "member_end(bottom");
    // Outer walls only (a rectangular tube's inner walls are `inner_*`).
    let outer = |member: &str| -> Vec<String> {
        face_refs(&v, &format!("member_side(\"{member}/")).into_iter().filter(|r| !r.contains("/inner_")).collect()
    };
    let sides = outer("bottom");
    let right_sides = outer("right");
    assert!(ends.len() == 2 && !sides.is_empty() && !right_sides.is_empty(), "faces: {:?}", face_refs(&v, "feature.members"));

    // An end cap on the bottom member's free end: a plate body.
    let cap = ends.iter().find_map(|end| apply(frame, "weldment.end-cap", json!({ "face": [end], "thickness": 3 })));
    let (_, capped) = cap.unwrap_or_else(|| panic!("no end cap applied on {ends:?}"));
    assert_eq!(bodies(&capped).len(), start + 1);

    // A gusset between a side of each member at their corner: a plate body.
    let gusset = sides
        .iter()
        .flat_map(|a| right_sides.iter().map(move |b| (a, b)))
        .find_map(|(a, b)| apply(frame, "weldment.gusset", json!({ "faces": [a, b], "leg": 30, "thickness": 4 })));
    let (_, gusseted) = gusset.expect("a gusset between the members' sides");
    assert_eq!(bodies(&gusseted).len(), start + 1);

    // A fillet weld along one of the bottom member's straight edges.
    let edges: Vec<String> = bodies(&v)
        .iter()
        .flat_map(|b| b["edges"].as_array().cloned().unwrap_or_default())
        .filter_map(|e| e["reference"].as_str().map(str::to_owned))
        .filter(|r| r.contains("member_side(\"bottom/"))
        .collect();
    let weld = edges.iter().find_map(|e| apply(frame, "weldment.weld-bead", json!({ "edges": [e], "leg": 3 })));
    assert!(weld.is_some(), "no weld bead applied on {edges:?}");

    // The bottom member, picked by a side face, extends from x = 200 mm to the plate's face.
    let target = face_refs(&v, "feature.plate.face.side(w.l4)");
    assert_eq!(target.len(), 1, "{:?}", face_refs(&v, "feature.plate"));
    let (text, extended) = apply(frame, "weldment.trim-extend", json!({ "member": [sides[0]], "to": [target[0]] })).expect("extend applies");
    assert!(text.contains("weldment.trim_extend("), "{text}");
    assert_eq!(bodies(&extended).len(), start, "the plate it runs to stays a body of its own");
    let reach = max_x_of_body_with(&extended, &sides[0]);
    assert!((reach - 0.25).abs() < 1e-6, "the bottom member reaches x = {reach} m");
}
