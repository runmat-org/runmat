//! Product families through `design.export` and the `runmat-design` CLI: every configuration
//! of a part as STEP, a drawing's sheets once per configuration of its part, and an
//! assembly's BOM as CSV.

use runmat_design_tools::{call, Files};
use serde_json::{json, Value};

const PLATE: &str = r#"runmat 0.7
std 1

part plate {
    input width : length = 80 mm
    sketch s on origin.xy {
        l1 = line(start = [0 mm, 0 mm], end = [input.width, 0 mm])
        l2 = line(start = l1.end, end = [input.width, 50 mm])
        l3 = line(start = l2.end, end = [0 mm, 50 mm])
        l4 = line(start = l3.end, end = l1.start)
    }
    feature body = extrude(profile = sketch.s, distance = 10 mm)
    output plate = { body = feature.body.body, part_number = "PL-80" }
    configuration wide = { overrides = { width = 120 mm } }
    configuration narrow = { overrides = { width = 40 mm } }
}
"#;

const PAIR: &str = r#"runmat 0.7
std 1

assembly pair {
    use part plate from "plate.part" as lower
    use part plate from "plate.part" as upper {
        placement = { placement = offset, offset = { translation = [0 mm, 0 mm, 10 mm] } }
    }
}
"#;

fn files() -> Files {
    [("/p/plate.part", PLATE), ("/p/pair.assembly", PAIR)].into_iter().map(|(k, v)| (k.to_owned(), v.to_owned())).collect()
}

fn export(input: Value) -> Value {
    call("design.export", input, &files()).unwrap_or_else(|e| panic!("{e}"))
}

/// The x extent (mm) of a STEP file's points.
fn x_extent(step: &str) -> f64 {
    let xs: Vec<f64> = step
        .match_indices("CARTESIAN_POINT('',(")
        .filter_map(|(i, m)| step[i + m.len()..].split(',').next()?.trim().parse().ok())
        .collect();
    xs.iter().cloned().fold(f64::MIN, f64::max) - xs.iter().cloned().fold(f64::MAX, f64::min)
}

#[test]
fn every_configuration_of_a_part_exports_its_own_step() {
    let out = export(json!({ "path": "/p/plate.part", "format": "step", "allConfigurations": true }));
    let files = out["files"].as_array().unwrap();
    let names: Vec<&str> = files.iter().map(|f| f["name"].as_str().unwrap()).collect();
    assert_eq!(names, ["plate.step", "plate-wide.step", "plate-narrow.step"]);
    let widths: Vec<f64> = files.iter().map(|f| x_extent(f["content"].as_str().unwrap())).collect();
    for (w, want) in widths.iter().zip([80.0, 120.0, 40.0]) {
        assert!((w - want).abs() < 1e-6, "{widths:?}");
    }
    let one = export(json!({ "path": "/p/plate.part", "format": "step", "configuration": "wide" }));
    assert!((x_extent(one["files"][0]["content"].as_str().unwrap()) - 120.0).abs() < 1e-6);
}

#[test]
fn an_assembly_bom_exports_as_csv() {
    let out = export(json!({ "path": "/p/pair.assembly", "format": "bom" }));
    let csv = out["files"][0]["content"].as_str().unwrap();
    assert_eq!(out["files"][0]["name"], "pair-bom.csv");
    let lines: Vec<&str> = csv.lines().collect();
    assert_eq!(lines[0], "item,part_number,file,configuration,material,quantity");
    assert!(lines[1].starts_with("1,PL-80,") && lines[1].ends_with(",2"), "{csv}");
}

#[test]
fn the_cli_writes_a_family_to_a_folder() {
    let dir = std::env::temp_dir().join(format!("runmat-design-cli-{}", std::process::id()));
    let work = dir.join("work");
    std::fs::create_dir_all(&work).unwrap();
    std::fs::write(work.join("plate.part"), PLATE).unwrap();
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_runmat-design"))
        .args(["export"])
        .arg(work.join("plate.part"))
        .args(["--format", "step", "--all-configurations", "--out"])
        .arg(dir.join("out"))
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    for f in ["plate.step", "plate-wide.step", "plate-narrow.step"] {
        let text = std::fs::read_to_string(dir.join("out").join(f)).unwrap();
        assert!(text.starts_with("ISO-10303-21;"), "{f}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
