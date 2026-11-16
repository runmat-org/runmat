//! Representations: a component set to `envelope` draws as its box (a sub-assembly, one box
//! around its parts) and still weighs its parts; `full` writes nothing; a part analysis
//! representation the part doesn't declare is refused.
use super::tests::{compile, files, run, Compiled, RIG};
use crate::edit::edit_json;
use crate::runtime::compile_assembly;
use serde_json::{json, Value};

fn represent(text: &str, c: &Compiled, path: &str, key: &str) -> String {
    let pick = format!("occurrence:{}", c.uuid(path));
    run(
        text,
        json!({ "command": "assembly.representation", "values": { "components": [pick], "representation": key } }),
    )
}

fn prototype<'a>(c: &'a Compiled, path: &str) -> &'a Value {
    let p = c.occurrence(path)["prototype"].as_u64().unwrap() as usize;
    &c.display["prototypes"][p]
}

fn bounds_mm(p: &Value) -> Vec<f64> {
    ["min", "max"]
        .iter()
        .flat_map(|k| p["bounds"][k].as_array().unwrap().clone())
        .map(|v| (v.as_f64().unwrap() * 1e3 * 100.0).round() / 100.0)
        .collect()
}

#[test]
fn an_envelope_component_draws_as_its_box_and_full_writes_nothing() {
    let c = compile(RIG);
    let text = represent(RIG, &c, "block", "envelope");
    assert!(text.contains("representation = envelope"), "{text}");
    let c = compile(&text);
    let block = prototype(&c, "block");
    assert_eq!(block["shape"], "box", "{block}");
    assert_eq!(bounds_mm(block), vec![0.0, 0.0, 0.0, 20.0, 20.0, 15.0]);
    let alpha = block["appearance"][3].as_f64().unwrap();
    assert!((alpha - 0.35).abs() < 1e-6, "{block}");
    // The others still draw their parts.
    assert_eq!(prototype(&c, "plate")["shape"], "mesh");
    assert_eq!(c.index["occurrences"].as_array().unwrap().iter().find(|r| r["path"] == "block").unwrap()["representation"], "envelope");
    let text = represent(&text, &c, "block", "full");
    assert!(!text.contains("representation"), "{text}");
    assert_eq!(prototype(&compile(&text), "block")["shape"], "mesh");
}

#[test]
fn an_analysis_representation_the_part_doesnt_declare_is_refused_and_an_unknown_key_too() {
    let c = compile(RIG);
    let pick = format!("occurrence:{}", c.uuid("pin"));
    let run_key = |key: &str| -> Value {
        serde_json::from_str(&edit_json(
            "/p/rig.assembly",
            RIG,
            &files(),
            &json!({ "command": "assembly.representation", "values": { "components": [pick], "representation": key } }).to_string(),
        ))
        .unwrap()
    };
    for key in ["midsurface", "beam", "defeatured"] {
        let out = run_key(key);
        assert_eq!(out["ok"], false, "{key}: {out}");
        assert!(out.to_string().contains(&format!("declares no {key} representation")), "{key}: {out}");
    }
    let out = run_key("mid_surface");
    assert_eq!(out["ok"], false, "{out}");
    assert!(
        out.to_string().contains("choose `full` or `envelope` or `defeatured` or `midsurface` or `beam`"),
        "{out}"
    );
}

#[test]
fn a_sub_assembly_at_envelope_draws_one_box_around_its_parts_and_weighs_them() {
    let sub = "runmat 0.7\nstd 1\n\nassembly pins {\n    use part pin as a\n    use part pin as b {\n        placement = { placement = offset, offset = { translation = [40 mm, 0 mm, 0 mm] } }\n    }\n}\n";
    let top = "runmat 0.7\nstd 1\n\nassembly rig {\n    use part plate as plate\n    use assembly pins as s {\n        placement = { placement = offset, offset = { translation = [0 mm, 0 mm, 10 mm] } }\n        representation = envelope\n    }\n}\n";
    let mut all = files();
    all.insert("/p/pins.assembly".into(), sub.into());
    let (index, display, _) = compile_assembly("/p/rig.assembly", top, &all).unwrap();
    let c = Compiled { index, display };
    // One box for the sub-assembly, in its frame: both pins (radius 5 mm, 30 mm long).
    let s = prototype(&c, "s");
    assert_eq!(s["shape"], "box", "{s}");
    assert_eq!(bounds_mm(s), vec![-5.0, -5.0, 0.0, 45.0, 5.0, 30.0]);
    let drawn: Vec<&str> = c.display["occurrences"]
        .as_array()
        .unwrap()
        .iter()
        .map(|o| o["path"].as_str().unwrap())
        .collect();
    assert_eq!(drawn, vec!["plate", "s"], "{drawn:?}");
    // Mass properties still weigh the parts the envelope stands for.
    let report = crate::mass::assembly_mass("/p/rig.assembly", top, &all, None).unwrap();
    let mut weighed: Vec<&str> = report.parts.iter().map(|p| p.path.as_str()).collect();
    weighed.sort();
    assert_eq!(weighed, vec!["plate", "s/a", "s/b"], "{report:?}");
    // An export writes the parts the envelope stands for, not its box.
    let meshes =
        crate::exchange::meshes::assembly_meshes("/p/rig.assembly", top, &all, None).unwrap();
    let mut exported: Vec<&str> = meshes
        .occurrences
        .iter()
        .map(|o| o.path.as_str())
        .collect();
    exported.sort();
    assert_eq!(exported, vec!["plate", "s/a", "s/b"]);
}
