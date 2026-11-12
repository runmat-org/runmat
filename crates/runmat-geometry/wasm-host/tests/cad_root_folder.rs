//! An assembly at the project root ("/house.assembly") finds the parts beside it, as one in a
//! folder ("/p/house.assembly") does: the house example's entry assembly and its skeleton.

use serde_json::{json, Value};

fn call(function: &str, args: &[&str]) -> Value {
    let request = json!({ "fn": function, "args": args });
    serde_json::from_str(&runmat_geometry_wasm_host::cad::call(&request.to_string())).expect("json")
}

const SKELETON: &str = "runmat 0.7\nstd 1\n\npart house_skeleton {\n    feature building = frame(frame = at, origin = point(0 in, 0 in, 0 in), z = origin.z, x = origin.x)\n    feature top_of_slab = plane.offset(base = origin.xy, distance = 0 in)\n    publish building = datum(datum = { datum = feature, feature = building })\n}\n";

const BLOCK: &str = "runmat 0.7\nstd 1\n\npart block {\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [10 mm, 0 mm])\n        l2 = line(start = [10 mm, 0 mm], end = [10 mm, 10 mm])\n        l3 = line(start = [10 mm, 10 mm], end = [0 mm, 10 mm])\n        l4 = line(start = [0 mm, 10 mm], end = [0 mm, 0 mm])\n    }\n    feature body = extrude(profile = sketch.s1, distance = 10 mm)\n    output block = { body = feature.body.body, material = \"material.aluminum_6061\" }\n}\n";

const HOUSE: &str = "runmat 0.7\nstd 1\n\nassembly house {\n    use part house_skeleton as skeleton\n    use part block as slab {\n        placement = { placement = attach, own = { anchor = origin }, to = { frame = interface, part = house_skeleton, name = building } }\n    }\n}\n";

const TOML: &str = "[package]\nname = \"house\"\nversion = \"0.1.0\"\n\n[sources]\nroots = [\".\"]\n\n[entrypoints.main]\npath = \"house.assembly\"\n";

fn errors(folder: &str) -> Vec<String> {
    let at = |name: &str| format!("{folder}{name}");
    let files = json!({
        at("house_skeleton.part"): SKELETON,
        at("block.part"): BLOCK,
        at("runmat.toml"): TOML,
    })
    .to_string();
    let out = call("compile", &[&at("house.assembly"), HOUSE, &files, "{}"]);
    out["diagnostics"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|d| d["severity"] == "error")
        .map(|d| d["message"].as_str().unwrap_or_default().to_owned())
        .collect()
}

#[test]
fn an_assembly_in_a_folder_finds_the_parts_beside_it() {
    assert_eq!(errors("/p/"), Vec::<String>::new());
}

#[test]
fn an_assembly_at_the_project_root_finds_the_parts_beside_it() {
    assert_eq!(errors("/"), Vec::<String>::new());
}
