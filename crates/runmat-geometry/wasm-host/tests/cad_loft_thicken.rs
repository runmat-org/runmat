//! The Surfaces tab's Lofted surface from the canvas on OCCT: a sheet through two circle
//! sketches, thickened into a solid.

use serde_json::{json, Value};

const TWO: &str = "runmat 0.7\nstd 1\n\npart skin {\n    sketch a on origin.xy {\n        c = circle(center = [0 mm, 0 mm], radius = 10 mm)\n    }\n    feature up = plane.offset(base = origin.xy, distance = 20 mm)\n    sketch b on plane.up {\n        c = circle(center = [0 mm, 0 mm], radius = 6 mm)\n    }\n}\n";

fn call(f: &str, args: Value) -> Value {
    let request = json!({ "fn": f, "args": args });
    serde_json::from_str(&runmat_geometry_wasm_host::cad::call(&request.to_string())).expect("json")
}

fn edit(text: &str, command: &str, values: Value) -> String {
    let out = call(
        "edit",
        json!([
            "/p/skin.part",
            text,
            "{}",
            json!({ "command": command, "values": values }).to_string()
        ]),
    );
    assert_eq!(out["ok"], true, "{command}: {out}");
    out["files"][0]["text"].as_str().unwrap().to_owned()
}

#[test]
fn a_lofted_surface_thickens_into_a_solid() {
    runmat_geometry_wasm_host::cad::install_occt();
    let t = edit(
        TWO,
        "surface.loft",
        json!({ "profiles": ["sketch:a", "sketch:b"] }),
    );
    let v = call("compile", json!(["/p/skin.part", t, "{}", "{}"]));
    let errors: Vec<&Value> = v["diagnostics"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|d| d["severity"] == "error")
        .collect();
    assert!(errors.is_empty(), "{errors:?}\n{t}");
    let t = edit(
        &t,
        "surface.thicken",
        json!({ "surface": ["feature:loft1"], "thickness": 1.5, "side": "one" }),
    );
    let v = call("compile", json!(["/p/skin.part", t, "{}", "{}"]));
    let errors: Vec<&Value> = v["diagnostics"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|d| d["severity"] == "error")
        .collect();
    assert!(errors.is_empty(), "{errors:?}\n{t}");
    let b = &v["display"]["data"]["bounds"];
    let height = b["max"][2].as_f64().unwrap() - b["min"][2].as_f64().unwrap();
    // The wall stands 1.5 mm off the sheet, so its rim at the small end rises a little above
    // the 20 mm section; its outside passes the 10 mm section's radius.
    let width = b["max"][0].as_f64().unwrap() - b["min"][0].as_f64().unwrap();
    assert!(
        height >= 0.02 - 1e-6 && height < 0.0215,
        "the solid spans the two sections: {b}"
    );
    assert!(
        width > 0.0205 && width < 0.0235,
        "the wall is 1.5 mm thick: {b}"
    );
}
