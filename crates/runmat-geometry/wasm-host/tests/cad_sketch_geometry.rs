//! Ellipses and text as sketch geometry on OCCT, through `runmat_cad_call`: the dispatcher draws
//! them, and the part regenerates with an exact elliptical boss and lettering cut into a plate.

use serde_json::{json, Value};

fn call(function: &str, args: &[&str]) -> Value {
    let request = json!({ "fn": function, "args": args });
    serde_json::from_str(&runmat_geometry_wasm_host::cad::call(&request.to_string())).expect("json")
}

/// The part after `command` on its sketch `s1`.
fn edited(part: &str, command: &str, values: Value) -> String {
    let request = json!({ "command": command, "values": values }).to_string();
    let out = call("edit", &["/p/p.part", part, "{}", &request]);
    assert_eq!(out["ok"], true, "{command}: {out}");
    out["files"][0]["text"].as_str().unwrap().to_owned()
}

/// The part's volume in mm³, after checking it compiles clean.
fn volume(text: &str) -> f64 {
    let compiled = call("compile", &["/p/p.part", text, "{}", "{}"]);
    let errors: Vec<&Value> = compiled["diagnostics"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|d| d["severity"] == "error")
        .collect();
    assert!(errors.is_empty(), "{errors:?}\n{text}");
    let mass = call(
        "analyze",
        &[
            "/p/p.part",
            text,
            "{}",
            &json!({ "command": "evaluate.mass-properties" }).to_string(),
        ],
    );
    assert_eq!(mass["ok"], true, "{mass}\n{text}");
    mass["data"]["total"]["volume"].as_f64().unwrap() * 1e9
}

const EMPTY: &str = "runmat 0.7\nstd 1\n\npart p {\n    sketch s1 on origin.xy {\n    }\n}\n";

fn with_feature(text: &str, feature: &str) -> String {
    let at = text.rfind("\n}\n").unwrap();
    format!("{}\n    {feature}\n    output p = {{ body = feature.boss.body, material = \"material.aluminum_6061\" }}{}", &text[..at], &text[at..])
}

#[test]
fn an_extruded_ellipse_is_an_exact_elliptic_cylinder() {
    let text = edited(
        EMPTY,
        "sketch.ellipse",
        json!({ "sketch": "s1", "type": "full", "center": { "x": 0, "y": 0 }, "major": 20, "minor": 10, "angle": 30 }),
    );
    let text = with_feature(
        &text,
        "feature boss = extrude(profile = sketch.s1, distance = 5 mm)",
    );
    let want = std::f64::consts::PI * 20.0 * 10.0 * 5.0;
    let got = volume(&text);
    assert!((got - want).abs() < 1e-6 * want, "{got} mm³, want {want}");
}

#[test]
fn half_an_ellipse_closed_by_a_line_extrudes_to_half_the_volume() {
    let text = edited(
        EMPTY,
        "sketch.ellipse",
        json!({ "sketch": "s1", "type": "partial", "center": { "x": 0, "y": 0 }, "major": 20, "minor": 10,
                "start": { "x": 30, "y": 0 }, "end": { "x": -30, "y": 0 } }),
    );
    let text = text.replace(
        "    }\n}\n",
        "        l1 = line(start = e1.end, end = e1.start)\n    }\n}\n",
    );
    let text = with_feature(
        &text,
        "feature boss = extrude(profile = sketch.s1, distance = 5 mm)",
    );
    let want = std::f64::consts::PI * 20.0 * 10.0 * 5.0 / 2.0;
    let got = volume(&text);
    assert!((got - want).abs() < 1e-6 * want, "{got} mm³, want {want}");
}

/// "HI" 6 mm high in a 1 mm pen, engraved 1 mm into the underside of a 40 x 20 x 5 mm plate. Each upright and bar
/// is a 1 mm wide stroke with round ends; their union is 15 + π/2 mm² for H and 9 + π/2 for I.
#[test]
fn engraved_text_cuts_its_outline_area_times_its_depth() {
    let plate = "runmat 0.7\nstd 1\n\npart p {\n    sketch s0 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [40 mm, 0 mm])\n        l2 = line(start = [40 mm, 0 mm], end = [40 mm, 20 mm])\n        l3 = line(start = [40 mm, 20 mm], end = [0 mm, 20 mm])\n        l4 = line(start = [0 mm, 20 mm], end = [0 mm, 0 mm])\n    }\n    feature boss = extrude(profile = sketch.s0, distance = 5 mm)\n    sketch s1 on origin.xy {\n    }\n}\n";
    let text = edited(
        plate,
        "sketch.text",
        json!({ "sketch": "s1", "text": "HI", "at": { "x": 10, "y": 7 }, "height": 6, "font": "single_line", "stroke": 1 }),
    );
    assert!(text.contains("t1 = text(text = \"HI\""), "{text}");
    let text = with_feature(
        &text,
        "feature engrave = extrude(profile = sketch.s1, distance = 1 mm, mode = cut)",
    );
    let cut = 24.0 + std::f64::consts::PI;
    let got = volume(&text);
    let want = 40.0 * 20.0 * 5.0 - cut;
    assert!(
        (got - want).abs() < 1e-4,
        "{got} mm³, want {want} (cut {cut})"
    );
}
