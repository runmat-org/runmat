//! Surface features on OCCT through `runmat_cad_call`: each Surfaces command writes a
//! `surface.*` feature, and the part regenerates with the sheet it makes, which a thicken turns
//! into a solid whose volume shows the sheet's area.

use serde_json::{json, Value};

fn call(function: &str, args: &[&str]) -> Value {
    let request = json!({ "fn": function, "args": args });
    serde_json::from_str(&runmat_geometry_wasm_host::cad::call(&request.to_string())).expect("json")
}

/// A part with planar patches, one per (sketch, x0, x1) on the XY plane, 20 mm deep.
fn patches(spans: &[(&str, f64, f64)]) -> String {
    let mut text = String::from("runmat 0.7\nstd 1\n\npart skin {\n");
    for (name, x0, x1) in spans {
        text += &format!(
            "    sketch {name} on origin.xy {{\n        a = line(start = [{x0} mm, 0 mm], end = [{x1} mm, 0 mm])\n        b = line(start = [{x1} mm, 0 mm], end = [{x1} mm, 20 mm])\n        c = line(start = [{x1} mm, 20 mm], end = [{x0} mm, 20 mm])\n        e = line(start = [{x0} mm, 20 mm], end = [{x0} mm, 0 mm])\n    }}\n    feature p_{name} = surface.planar(profile = sketch.{name})\n"
        );
    }
    text + "}\n"
}

/// Runs the command on `text`: the new text and the name of the feature it added.
fn edit(text: &str, command: &str, values: Value) -> (String, String) {
    let request = json!({ "command": command, "values": values }).to_string();
    let out = call("edit", &["/p/skin.part", text, "{}", &request]);
    assert_eq!(out["ok"], true, "{command}: {out}");
    let next = out["files"][0]["text"].as_str().unwrap().to_owned();
    let added = out["summary"].as_str().unwrap().trim_start_matches("Added ").to_owned();
    (next, added)
}

/// The solids' volume in mm³ after the text compiles without errors.
fn volume(text: &str) -> f64 {
    let compiled = call("compile", &["/p/skin.part", text, "{}", "{}"]);
    let errors: Vec<&Value> = compiled["diagnostics"].as_array().into_iter().flatten().filter(|d| d["severity"] == "error").collect();
    assert!(errors.is_empty(), "{errors:?}\n{text}");
    let mass = call("analyze", &["/p/skin.part", text, "{}", &json!({ "command": "evaluate.mass-properties" }).to_string()]);
    assert_eq!(mass["ok"], true, "{mass}");
    mass["data"]["total"]["volume"].as_f64().unwrap() * 1e9
}

fn thickened(text: &str, body: &str) -> f64 {
    let (text, _) = edit(text, "surface.thicken", json!({ "surface": [body], "thickness": 1, "side": "one" }));
    volume(&text)
}

fn near(got: f64, want: f64, what: &str) {
    assert!((got - want).abs() < 0.02 * want, "{what}: {got:.1} mm³, want {want}");
}

#[test]
fn thicken_trim_and_knit_make_solids_of_the_sheets_they_keep() {
    // A 40 x 20 mm patch thickened 3 mm.
    let skin = patches(&[("s", 0.0, 40.0)]);
    let (text, _) = edit(&skin, "surface.thicken", json!({ "surface": ["feature.p_s.face.patch"], "thickness": 3, "side": "one" }));
    assert!(text.contains("surface.thicken("), "{text}");
    near(volume(&text), 2400.0, "thicken");
    // A patch across the right plane, trimmed to the side above it: 20 x 20 mm.
    let wide = patches(&[("s", -20.0, 20.0)]);
    let (text, _) = edit(&wide, "surface.trim", json!({ "bodies": ["feature.p_s.body"], "tool": ["origin:right"], "keep": "positive" }));
    assert!(text.contains("surface.trim("), "{text}");
    // A trim keeps the sheet's own name.
    near(thickened(&text, "feature.p_s.body"), 400.0, "trim");
    // Two patches sharing an edge, knit into one sheet: 40 x 20 mm.
    let pair = patches(&[("l", 0.0, 20.0), ("r", 20.0, 40.0)]);
    let (text, _) = edit(&pair, "surface.knit", json!({ "bodies": ["feature.p_l.body", "feature.p_r.body"], "solid": false }));
    assert!(text.contains("surface.knit("), "{text}");
    // The knit sheet keeps the first sheet's name.
    near(thickened(&text, "feature.p_l.body"), 800.0, "knit");
}

#[test]
fn fill_and_boundary_surfaces_patch_their_edges() {
    let skin = patches(&[("s", 0.0, 40.0)]);
    let edge = |e: &str| format!("feature.p_s.edge.boundary(s.{e})");
    // A fill over the patch's outline.
    let (text, fill) = edit(&skin, "surface.fill", json!({ "boundary": [edge("a"), edge("b"), edge("c"), edge("e")], "continuity": "g0" }));
    assert!(text.contains("surface.fill("), "{text}");
    near(thickened(&text, &format!("feature.{fill}.body")), 800.0, "fill");
    // Through the two long edges in one direction.
    let (text, through) = edit(&skin, "surface.boundary", json!({ "direction1": [edge("a"), edge("c")], "continuity": "g0" }));
    assert!(text.contains("surface.boundary("), "{text}");
    near(thickened(&text, &format!("feature.{through}.body")), 800.0, "boundary in one direction");
    // Bounded by the long edges in one direction and the short ones in the other.
    let (text, bounded) = edit(&skin, "surface.boundary", json!({ "direction1": [edge("a"), edge("c")], "direction2": [edge("b"), edge("e")], "continuity": "g0" }));
    near(thickened(&text, &format!("feature.{bounded}.body")), 800.0, "boundary in two directions");
}
