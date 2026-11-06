//! `design.export` with `format: "fabrication"` on the quadcopter's flight-controller board:
//! the folders of the set, the ODB++ archive listed by the system `tar`, the PDF sheets, and
//! the options that narrow the set.

use base64::Engine;
use runmat_design_tools::{call, Files};
use serde_json::{json, Value};
use std::path::PathBuf;

fn files() -> Files {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../runmat-cad/wasm/tests/drone/electrical/flight_controller");
    [
        "flight_controller.circuit",
        "flight_controller.circuit.layout",
        "flight_controller.circuit.lock",
        "fc_models.lib",
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
    let out = export(json!({ "fabrication": { "preset": "full", "zip": false } }));
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
        json!({ "fabrication": { "gerber": false, "drill": false, "drillMap": false, "drawing": false, "ipc356": false, "odb": true, "zip": false } }),
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
    // The board is clean on ERC, the review suites, DRC, routing and sync, and its simulations
    // run.
    let errors: Vec<String> = r
        .findings
        .iter()
        .filter(|x| x.severity == "error")
        .map(|x| format!("{} {}", x.check, x.code))
        .collect();
    assert!(errors.is_empty(), "{errors:?}\n{:#?}", r.findings);
    assert!(r.passed);
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

/// The layout with `settings` saved into it, beside the circuit, as the Desktop leaves it.
fn files_with_saved(options: Value) -> Files {
    let mut f = files();
    let out = call(
        "design.export",
        json!({ "path": "/p/flight_controller.circuit.layout", "format": "fabrication", "fabrication": options }),
        &f,
    )
    .unwrap();
    let layout = &out["fabrication"]["layout"];
    f.insert(layout["path"].as_str().unwrap().to_owned(), layout["text"].as_str().unwrap().to_owned());
    f
}

#[test]
fn a_board_without_settings_makes_the_fab_house_set_zipped() {
    let out = export(json!({}));
    let n = names(&out);
    assert!(n.iter().any(|x| x.starts_with("gerber/")) && n.iter().any(|x| x.starts_with("netlist/")));
    assert!(!n.iter().any(|x| x.starts_with("assembly/") || x.starts_with("odb/")), "{n:?}");
    let zip = bytes(&out, "flight_controller-fab.zip");
    assert!(zip.starts_with(b"PK\x03\x04"));
    let summary = &out["fabrication"];
    assert_eq!(summary["settings"]["preset"], "fab_house");
    assert_eq!(summary["folder"], "fab/flight_controller");
    let layers = summary["layers"].as_array().unwrap();
    let row = |name: &str| layers.iter().find(|l| l["name"] == name).unwrap_or_else(|| panic!("no {name}"));
    assert_eq!(row("F.Cu")["included"], true);
    assert_eq!(row("F.Courtyard")["included"], false);
    assert!(row("F.Cu")["copperOz"].as_f64().is_some());
}

#[test]
fn a_plan_writes_nothing_and_reports_the_preflight() {
    let off = json!({ "gerber": false, "drill": false, "drillMap": false, "drawing": false, "pickPlace": false, "bom": false, "assemblyDrawing": false, "ipc2581": false, "odb": false, "ipc356": false, "preflight": true });
    let out = export(json!({ "fabrication": off }));
    assert!(names(&out).is_empty(), "{:?}", names(&out));
    let summary = &out["fabrication"];
    assert_eq!(summary["settings"]["preset"], "fab_house", "a plan leaves the saved settings alone");
    assert_eq!(summary["settings"]["outputs"], json!(["gerber", "drill", "drill_map", "drawing", "ipc356"]));
    assert!(summary["layers"].as_array().unwrap().len() > 10);
    let pre = &summary["preflight"];
    for k in ["drcErrors", "ercErrors", "unrouted", "pendingChanges"] {
        assert!(pre[k].as_u64().is_some(), "{k}: {pre}");
    }
}

#[test]
fn saved_settings_make_the_same_set_as_the_call_that_saved_them() {
    let options = json!({ "preset": "assembly", "zip": false, "layers": ["F.Cu", "Edge.Cuts"], "remember": true });
    let explicit = call(
        "design.export",
        json!({ "path": "/p/flight_controller.circuit.layout", "format": "fabrication", "fabrication": options }),
        &files(),
    )
    .unwrap();
    let saved = files_with_saved(options.clone());
    assert!(saved["/p/flight_controller.circuit.layout"].contains("fabrication = { preset = assembly"));
    // A later export with no options (the CLI's re-export, an agent's design.export) reads them.
    let again = call("design.export", json!({ "path": "/p/flight_controller.circuit.layout", "format": "fabrication" }), &saved).unwrap();
    assert_eq!(names(&again), names(&explicit));
    assert!(names(&again).iter().any(|n| n.ends_with("-assembly-top.pdf")), "{:?}", names(&again));
    assert_eq!(again["files"], explicit["files"]);
}

#[test]
fn output_formats_and_the_revision_are_saved_and_made_again() {
    let options = json!({
        "preset": "full", "zip": false, "revision": "B", "gerberUnits": "inch", "gerberDecimals": 5,
        "pickPlaceSides": "separate", "pickPlaceUnits": "inch", "bomColumns": ["designators", "mpn"], "bomGroupByValue": false, "remember": true
    });
    let explicit = call("design.export", json!({ "path": "/p/flight_controller.circuit.layout", "format": "fabrication", "fabrication": options }), &files()).unwrap();
    assert_eq!(explicit["fabrication"]["folder"], "fab/flight_controller-B");
    let n = names(&explicit);
    assert!(n.contains(&"assembly/flight_controller-pos-top.csv".to_string()) && n.contains(&"assembly/flight_controller-pos-bottom.csv".to_string()), "{n:?}");
    let cu = String::from_utf8(bytes(&explicit, "gerber/flight_controller-F_Cu.gbr")).unwrap();
    assert!(cu.contains("%MOIN*%") && cu.contains("%FSLAX25Y25*%"));
    assert!(String::from_utf8(bytes(&explicit, "assembly/flight_controller-bom.csv")).unwrap().starts_with("Designators,MPN\n"));
    let saved = files_with_saved(options);
    let layout = &saved["/p/flight_controller.circuit.layout"];
    assert!(layout.contains(r#"revision = "B""#) && layout.contains("gerber = { units = inch, decimals = 5 }"), "{}", layout.lines().find(|l| l.contains("fabrication")).unwrap_or(""));
    let again = call("design.export", json!({ "path": "/p/flight_controller.circuit.layout", "format": "fabrication" }), &saved).unwrap();
    assert_eq!(again["files"], explicit["files"]);
    // Bad values are refused with what's expected.
    let bad = call("design.export", json!({ "path": "/p/flight_controller.circuit.layout", "format": "fabrication", "fabrication": { "gerberUnits": "furlong" } }), &files());
    assert!(format!("{:?}", bad.unwrap_err()).contains("expected mm or inch"));
}

#[test]
fn a_plan_answers_one_layer_drawn_as_its_gerber() {
    let off = json!({ "gerber": false, "drill": false, "drillMap": false, "drawing": false, "pickPlace": false, "bom": false, "assemblyDrawing": false, "ipc2581": false, "odb": false, "ipc356": false });
    let with = |layer: &str| {
        let mut o = off.clone();
        o["previewLayer"] = json!(layer);
        export(json!({ "fabrication": o }))
    };
    let mask = with("F.Mask");
    let svg = mask["fabrication"]["preview"]["svg"].as_str().unwrap();
    assert!(svg.starts_with("<svg") && svg.contains(r#"data-layer="Soldermask,Top""#));
    // One mask opening per top pad at least: the MCU alone has dozens.
    let openings = svg.matches("<rect").count() + svg.matches("<circle").count() + svg.matches("<path").count();
    assert!(openings > 50, "{openings} openings");
    let paste = with("F.Paste");
    assert!(paste["fabrication"]["preview"]["svg"].as_str().unwrap().contains(r#"data-layer="Paste,Top""#));
    assert!(names(&paste).is_empty(), "a preview writes no files");
    // The fabrication drawing previews with the revision its title block names.
    let mut o = off.clone();
    o["previewLayer"] = json!("Fabrication drawing");
    o["revision"] = json!("C");
    let drawing = export(json!({ "fabrication": o }));
    let svg = drawing["fabrication"]["preview"]["svg"].as_str().unwrap();
    assert!(svg.contains(r#"data-layer="flight_controller fabrication drawing, revision C""#), "{}", &svg[..svg.len().min(300)]);
    assert!(svg.matches("<polyline").count() > 200, "the tables, drill map and text draw as strokes");
    assert_eq!(drawing["fabrication"]["preview"]["layer"], "Fabrication drawing");
    assert!(names(&drawing).is_empty(), "a preview writes no files");
    let bad = call("design.export", json!({ "path": "/p/flight_controller.circuit.layout", "format": "fabrication", "fabrication": { "previewLayer": "F.Nowhere" } }), &files());
    assert!(format!("{:?}", bad.unwrap_err()).contains("the board has no layer F.Nowhere"));
}

#[test]
fn an_export_reports_each_output_as_it_starts() {
    use std::{cell::RefCell, rc::Rc};
    let steps = Rc::new(RefCell::new(Vec::<String>::new()));
    let into = steps.clone();
    let fab = json!({ "gerber": true, "drill": true, "drillMap": false, "drawing": true, "pickPlace": true, "bom": true, "assemblyDrawing": false, "ipc2581": false, "odb": false, "ipc356": true, "preflight": true });
    let out = runmat_design_tools::progress::reporting(move |s| into.borrow_mut().push(s.to_owned()), || export(json!({ "fabrication": fab })));
    assert!(!names(&out).is_empty());
    assert_eq!(*steps.borrow(), ["board", "gerber", "drill", "drawing", "pick_place", "bom", "ipc356", "preflight"]);
}

#[test]
fn the_job_file_carries_the_date_it_is_given() {
    let fab = json!({ "gerber": true, "drill": false, "drillMap": false, "drawing": false, "pickPlace": false, "bom": false, "assemblyDrawing": false, "ipc2581": false, "odb": false, "ipc356": false, "zip": false });
    let job = |extra: Value| {
        let mut f = fab.clone();
        f.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
        let out = export(json!({ "fabrication": f }));
        String::from_utf8(bytes(&out, "gerber/flight_controller.gbrjob")).unwrap()
    };
    let dated = job(json!({ "date": "2026-11-06T09:30:00Z" }));
    assert!(dated.contains(r#""CreationDate": "2026-11-06T09:30:00Z""#), "{dated}");
    // Without a date the job file has none, so the same board makes the same files.
    assert!(!job(json!({})).contains("CreationDate"));
    let bad = call("design.export", json!({ "path": "/p/flight_controller.circuit.layout", "format": "fabrication", "fabrication": { "date": "yesterday" } }), &files());
    assert!(format!("{:?}", bad.unwrap_err()).contains("isn't ISO 8601"));
}
