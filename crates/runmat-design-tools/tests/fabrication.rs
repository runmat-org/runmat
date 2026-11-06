//! `design.export` with `format: "fabrication"` on the quadcopter's flight-controller board:
//! the folders of the set, the ODB++ archive listed by the system `tar`, the PDF sheets, and
//! the options that narrow the set.

use base64::Engine;
use runmat_design_tools::{call, Files};
use serde_json::{json, Value};
use std::path::PathBuf;

fn files() -> Files {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../runmat-cad/wasm/tests/drone");
    [
        "flight_controller.circuit",
        "flight_controller.circuit.layout",
        "flight_controller.circuit.lock",
        "drone_models.lib",
    ]
    .iter()
    .filter_map(|x| {
        let name = x.to_string();
        Some((
            format!("/p/{name}"),
            std::fs::read_to_string(dir.join(&name)).ok()?,
        ))
    })
    .collect()
}

fn export(extra: Value) -> Value {
    let mut input =
        json!({ "path": "/p/flight_controller.circuit.layout", "format": "fabrication" });
    if let Value::Object(m) = extra {
        input.as_object_mut().unwrap().extend(m);
    }
    call("design.export", input, &files()).unwrap()
}

fn names(out: &Value) -> Vec<String> {
    out["files"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| f["name"].as_str().unwrap().to_owned())
        .collect()
}

fn bytes(out: &Value, name: &str) -> Vec<u8> {
    let f = out["files"]
        .as_array()
        .unwrap()
        .iter()
        .find(|f| f["name"] == name)
        .unwrap_or_else(|| panic!("no {name}"));
    let c = f["content"].as_str().unwrap();
    if f["encoding"] == "base64" {
        base64::engine::general_purpose::STANDARD.decode(c).unwrap()
    } else {
        c.as_bytes().to_vec()
    }
}

#[test]
fn the_whole_set_in_its_folders() {
    let out = export(json!({}));
    let n = names(&out);
    for folder in [
        "gerber/",
        "drill/",
        "drawing/",
        "assembly/",
        "ipc2581/",
        "odb/",
        "netlist/",
    ] {
        assert!(
            n.iter().any(|x| x.starts_with(folder)),
            "no {folder} in {n:?}"
        );
    }
    assert!(
        !n.iter().any(|x| x.starts_with("odb/flight_controller/")),
        "the ODB++ tree is archived"
    );
    for pdf in [
        "drill/flight_controller-drill-map.pdf",
        "drawing/flight_controller-fab-drawing.pdf",
    ] {
        let b = bytes(&out, pdf);
        assert!(
            b.starts_with(b"%PDF-1.4") && b.ends_with(b"%%EOF\n"),
            "{pdf}"
        );
    }
    let drawing =
        String::from_utf8_lossy(&bytes(&out, "drawing/flight_controller-fab-drawing.pdf"))
            .into_owned();
    assert!(drawing.contains("(LAYERS AND STACKUP"), "the stackup table");
    assert!(drawing.contains("(DRILL TABLE"), "the drill table");
    let summary = &out["fabrication"];
    assert_eq!(summary["board"], "flight_controller");
    let copper = summary["layers"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|l| l["kind"] == "copper")
        .count();
    assert_eq!(copper, 4);
}

#[test]
fn the_odb_archive_lists_with_the_system_tar() {
    let out = export(
        json!({ "fabrication": { "gerber": false, "drill": false, "drillMap": false, "drawing": false, "pickPlace": false, "bom": false, "ipc2581": false, "ipc356": false } }),
    );
    assert_eq!(names(&out), vec!["odb/flight_controller-odb.tgz"]);
    let dir = std::env::temp_dir().join(format!("runmat-odb-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let f = dir.join("board.tgz");
    std::fs::write(&f, bytes(&out, "odb/flight_controller-odb.tgz")).unwrap();
    let r = std::process::Command::new("tar")
        .arg("-tzf")
        .arg(&f)
        .output()
        .expect("tar");
    let _ = std::fs::remove_dir_all(&dir);
    assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
    let listed = String::from_utf8_lossy(&r.stdout);
    for p in [
        "flight_controller/matrix/matrix",
        "flight_controller/steps/pcb/eda/data",
        "flight_controller/steps/pcb/layers/top/features",
    ] {
        assert!(listed.lines().any(|l| l == p), "{p} not in {listed}");
    }
}

#[test]
fn a_schematic_without_its_board_is_refused() {
    let mut f = files();
    f.remove("/p/flight_controller.circuit.layout");
    let r = call(
        "design.export",
        json!({ "path": "/p/flight_controller.circuit", "format": "fabrication" }),
        &f,
    );
    assert!(r.is_err());
}

#[test]
fn netlists_in_each_format() {
    let f = files();
    let run = |format: &str| {
        call(
            "design.export",
            json!({ "path": "/p/flight_controller.circuit", "format": format }),
            &f,
        )
        .unwrap()
    };
    let net = run("kicad_netlist");
    assert_eq!(names(&net), vec!["flight_controller.net"]);
    assert!(String::from_utf8(bytes(&net, "flight_controller.net"))
        .unwrap()
        .starts_with("(export"));
    let edn = run("edif");
    assert!(String::from_utf8(bytes(&edn, "flight_controller.edn"))
        .unwrap()
        .contains("(edifVersion 2 0 0)"));
    let ipc = call(
        "design.export",
        json!({ "path": "/p/flight_controller.circuit.layout", "format": "ipc356" }),
        &f,
    )
    .unwrap();
    assert_eq!(names(&ipc), vec!["flight_controller.ipc"]);
    // The SPICE deck parses with the circuit engine's own netlist reader.
    let cir = run("spice");
    let deck = String::from_utf8(bytes(&cir, "flight_controller.cir")).unwrap();
    let parsed = runmat_circuit_sim::netlist::parse(&deck, &|_: &str| Some(String::new()))
        .unwrap_or_else(|e| panic!("{e:?}\n{deck}"));
    assert!(parsed.body.cards.len() > 10, "{deck}");
}

#[test]
fn circuit_checks_report_the_board_and_fail_on_an_unrouted_one() {
    let f = files();
    let r = runmat_design_tools::ci::check_circuit("/p/flight_controller.circuit.layout", &f, true)
        .unwrap();
    // The board is clean on ERC, DRC, routing and sync, and its simulations run; the one error
    // is the reset-strap review finding on the MCU (NRST without a pull-up).
    let errors: Vec<String> = r
        .findings
        .iter()
        .filter(|x| x.severity == "error")
        .map(|x| format!("{} {}", x.check, x.code))
        .collect();
    assert_eq!(
        errors,
        vec!["review straps.reset-pull-up"],
        "{:#?}",
        r.findings
    );
    assert!(!r.passed);
    for c in [
        "erc",
        "review",
        "drc",
        "simulation:buck_startup",
        "simulation:buck_tolerance",
    ] {
        assert!(r.ran.iter().any(|x| x == c), "{c} in {:?}", r.ran);
    }
    assert!(!r.measurements.is_empty());
    // Take one track out: the board now has an unrouted connection, and the check fails.
    let mut broken = f.clone();
    let layout = broken
        .get_mut("/p/flight_controller.circuit.layout")
        .unwrap();
    let line = layout
        .lines()
        .find(|l| l.trim_start().starts_with("track USB_DP"))
        .unwrap()
        .to_owned();
    *layout = layout.replacen(&format!("{line}\n"), "", 1);
    let r = runmat_design_tools::ci::check_circuit(
        "/p/flight_controller.circuit.layout",
        &broken,
        false,
    )
    .unwrap();
    assert!(!r.passed);
    assert!(
        r.findings.iter().any(|x| x.check == "unrouted"),
        "{:#?}",
        r.findings
    );
}
