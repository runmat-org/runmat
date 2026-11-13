//! Parts from the schematic canvas: placing one at a clicked point (turned or mirrored) with
//! its designator assigned, pasting copies under fresh designators, and moving a part with its
//! wires attached. Each is a `.circuit` edit and a lock edit, as the Desktop makes them.

use runmat_circuit::runtime::{self, CompileOptions, EditRequest, Files};
use serde_json::{json, Value};

const PATH: &str = "/p/reg.circuit";
const LOCK: &str = "/p/reg.circuit.lock";

const BLANK: &str = r#"runmat 0.7
std 1

circuit reg {
    root = main

    sheet main {
        component U1 = { part = "LDO-3V3-SOT23" }
        // The input capacitor.
        component C1 = { part = "C0603-1U-50V" }
        connect = [U1.VIN, "C1.1"]
    }
}
"#;

#[derive(Clone, Debug)]
struct Project {
    circuit: String,
    lock: Option<String>,
}

impl Project {
    fn files(&self) -> Files {
        let mut f: Files = [(PATH.to_owned(), self.circuit.clone())]
            .into_iter()
            .collect();
        if let Some(l) = &self.lock {
            f.insert(LOCK.to_owned(), l.clone());
        }
        f
    }

    fn run(&self, command: &str, values: Value) -> Result<(Project, String), String> {
        let request = EditRequest {
            command: command.into(),
            values: values.as_object().cloned().unwrap_or_default(),
            selection: Vec::new(),
            options: CompileOptions::default(),
        };
        let out =
            runtime::edit(PATH, &self.circuit, &self.files(), &request).map_err(|e| e.message)?;
        let mut next = self.clone();
        for f in &out.files {
            if f.path == PATH {
                next.circuit = f.text.clone();
            } else if f.path == LOCK {
                next.lock = Some(f.text.clone());
            }
        }
        Ok((next, out.summary))
    }

    fn step(&self, command: &str, values: Value) -> Project {
        self.run(command, values)
            .unwrap_or_else(|e| panic!("{command}: {e}\n{}", self.circuit))
            .0
    }

    fn index(&self) -> runmat_circuit::index::CircuitIndex {
        let out = runtime::compile(
            PATH,
            &self.circuit,
            &self.files(),
            &CompileOptions::default(),
        );
        assert!(
            out.diagnostics.is_empty(),
            "{:?}\n{}",
            out.diagnostics,
            self.circuit
        );
        out.index.expect("compiles")
    }

    /// A symbol as drawn: its place (page mm, y up), turn and mirror.
    fn symbol(&self, designator: &str) -> (f64, f64, f64, bool) {
        let index = self.index();
        let s = index
            .schematic
            .sheets
            .iter()
            .flat_map(|p| p.symbols.iter())
            .find(|s| s.designator == designator)
            .unwrap_or_else(|| panic!("{designator} is drawn"));
        (s.x_mm, s.y_mm, s.rotation, s.mirror)
    }

    /// The place line the lock records for `key` (`main/R1`).
    fn place_line(&self, key: &str) -> Option<String> {
        let prefix = format!("place {key} ");
        self.lock
            .as_deref()?
            .lines()
            .find(|l| l.starts_with(&prefix))
            .map(str::to_owned)
    }
}

fn blank() -> Project {
    // The first compile's layout recorded, as a saved project has it.
    let p = Project {
        circuit: BLANK.to_owned(),
        lock: None,
    };
    p.step("schematic.annotate", json!({}))
}

#[test]
fn a_part_lands_where_it_is_clicked_with_the_next_designator() {
    let p = blank();
    let (u1x, u1y, _, _) = p.symbol("U1");
    let (p2, summary) = p
        .run("schematic.place-part", json!({ "part": "R0402-10K", "designator": "R?", "value": "10 kohm ± 1 %", "at": "point:60.96,50.8" }))
        .unwrap();
    assert_eq!(summary, "placed R1 (R0402-10K) on sheet main");
    // One line added after the last component; the comment and the connect stay.
    assert!(p2.circuit.contains("        component C1 = { part = \"C0603-1U-50V\" }\n        component R1 = { part = \"R0402-10K\", value = 10 kohm ± 1 % }\n"), "{}", p2.circuit);
    assert!(p2.circuit.contains("// The input capacitor."));
    // Places snap to the grid the lock counts (y down from the paper's top edge), so a clicked
    // point lands within half a grid step.
    let near = |a: f64, b: f64| (a - b).abs() <= 0.635 + 1e-9;
    let (x, y, rot, mirror) = p2.symbol("R1");
    assert!(
        near(x, 60.96) && near(y, 50.8) && rot == 0.0 && !mirror,
        "{x} {y} {rot} {mirror}"
    );
    // Nothing else moved.
    assert_eq!(p2.symbol("U1").0, u1x);
    assert_eq!(p2.symbol("U1").1, u1y);
    assert!(p2.place_line("main/R1").is_some());

    // The next one takes R2; a taken designator is refused.
    let p3 = p2.step("schematic.place-part", json!({ "part": "R0402-10K", "designator": "R?", "at": "point:71.12,50.8", "rotation": 90, "mirror": true }));
    let (_, _, rot, mirror) = p3.symbol("R2");
    assert_eq!((rot, mirror), (90.0, true));
    let err = p3
        .run(
            "schematic.place-part",
            json!({ "part": "R0402-10K", "designator": "R1" }),
        )
        .unwrap_err();
    assert!(err.contains("R1 is already in the design"), "{err}");
    let err = p3
        .run("schematic.place-part", json!({ "part": "NO-SUCH-PART" }))
        .unwrap_err();
    assert!(err.contains("no part `NO-SUCH-PART`"), "{err}");
    // With no designator the part's own prefix is used; with no point the layout finds room.
    let (p4, summary) = p3
        .run("schematic.place-part", json!({ "part": "C0603-1U-50V" }))
        .unwrap();
    assert!(summary.starts_with("placed C2 (C0603-1U-50V)"), "{summary}");
    assert!(p4.place_line("main/C2").is_some(), "{:?}", p4.lock);
}

#[test]
fn a_library_record_is_embedded_once_and_placed() {
    let p = blank();
    let record = json!({
        "mpn": "TEST-BUCK-1", "manufacturer": "Example", "category": "regulator", "package": "SOT23-6",
        "pins": [
            { "name": "VIN", "pads": ["1"] }, { "name": "GND", "pads": ["2"] }, { "name": "EN", "pads": ["3"] },
            { "name": "FB", "pads": ["4"] }, { "name": "SW", "pads": ["5"] }, { "name": "BST", "pads": ["6"] }
        ]
    });
    let p2 = p.step(
        "schematic.place-part",
        json!({ "part": record, "at": "point:101.6,60.96" }),
    );
    assert_eq!(
        p2.circuit.matches("part \"TEST-BUCK-1\"").count(),
        1,
        "{}",
        p2.circuit
    );
    assert!(
        p2.circuit
            .contains("component U2 = { part = \"TEST-BUCK-1\" }"),
        "{}",
        p2.circuit
    );
    let (x, y, _, _) = p2.symbol("U2");
    assert!(
        (x - 101.6).abs() <= 0.64 && (y - 60.96).abs() <= 0.64,
        "{x} {y}"
    );
    // A second placement reuses the declaration.
    let p3 = p2.step(
        "schematic.place-part",
        json!({ "part": record, "at": "point:127,60.96" }),
    );
    assert_eq!(p3.circuit.matches("part \"TEST-BUCK-1\"").count(), 1);
    assert!(p3
        .circuit
        .contains("component U3 = { part = \"TEST-BUCK-1\" }"));
}

#[test]
fn paste_copies_parts_under_fresh_designators_with_their_properties() {
    let p = blank().step("schematic.place-part", json!({ "part": "R0402-10K", "value": "4.7 kohm ± 1 %", "at": "point:60.96,50.8", "rotation": 90 }));
    let (p2, summary) = p
        .run(
            "schematic.paste",
            json!({ "items": ["component:R1", "component:C1"], "dx": 12.7, "dy": -12.7 }),
        )
        .unwrap();
    assert_eq!(summary, "pasted R1 as R2, C1 as C2");
    assert!(
        p2.circuit
            .contains("component R2 = { part = \"R0402-10K\", value = 4.7 kohm ± 1 % }"),
        "{}",
        p2.circuit
    );
    assert!(p2
        .circuit
        .contains("component C2 = { part = \"C0603-1U-50V\" }"));
    // The originals keep their designators and connections; the copies join no net.
    assert!(p2
        .circuit
        .contains("component R1 = { part = \"R0402-10K\", value = 4.7 kohm ± 1 % }"));
    assert!(p2.circuit.contains(r#"connect = [U1.VIN, "C1.1"]"#));
    let (x0, y0, _, _) = p2.symbol("R1");
    let (x, y, rot, _) = p2.symbol("R2");
    assert!(
        ((x - x0) - 12.7).abs() < 1e-6 && ((y - y0) + 12.7).abs() < 1e-6 && rot == 90.0,
        "{x0} {y0} → {x} {y} {rot}"
    );
    let index = p2.index();
    let c2 = index
        .schematic
        .sheets
        .iter()
        .flat_map(|s| s.symbols.iter())
        .find(|s| s.designator == "C2")
        .unwrap();
    assert!(
        c2.pins.iter().all(|pin| pin.net.is_none()),
        "a pasted copy joins no net"
    );
    assert!(p2
        .run("schematic.paste", json!({ "items": [] }))
        .unwrap_err()
        .contains("copy one or more parts"));
}

#[test]
fn a_pasted_group_keeps_the_wires_between_its_parts_and_its_rails() {
    // U1 and C1 share the VIN net; C1.2 and U1.GND sit on the global GND rail.
    let text = BLANK.replace(
        "connect = [U1.VIN, \"C1.1\"]",
        "net GND = { global = true }\n        connect = [U1.VIN, \"C1.1\"]\n        connect = [net.GND, U1.GND, \"C1.2\"]",
    );
    let p = Project {
        circuit: text,
        lock: None,
    }
    .step("schematic.annotate", json!({}));
    let (p2, summary) = p
        .run(
            "schematic.paste",
            json!({ "items": ["component:U1", "component:C1"], "dx": 25.4, "dy": 0 }),
        )
        .unwrap();
    assert_eq!(summary, "pasted U1 as U2, C1 as C2 with 2 connections");
    // The copies are wired to each other on a new net, and to the rail they were on.
    assert!(p2.circuit.contains("component C2 = { part = \"C0603-1U-50V\" }\n        connect = [U1.GND, U2.GND, \"C2.2\"]\n        connect = [U2.VIN, \"C2.1\"]\n"), "{}", p2.circuit);
    let index = p2.index();
    let pins: Vec<_> = index
        .schematic
        .sheets
        .iter()
        .flat_map(|s| s.symbols.iter())
        .flat_map(|s| {
            s.pins
                .iter()
                .map(move |p| (format!("{}.{}", s.designator, p.name), p.net.clone()))
        })
        .collect();
    let net = |pin: &str| {
        pins.iter()
            .find(|(n, _)| n == pin)
            .and_then(|(_, net)| net.clone())
            .unwrap_or_else(|| panic!("{pin} has a net: {pins:?}"))
    };
    assert_eq!(net("U2.VIN"), net("C2.1"));
    assert_ne!(
        net("U2.VIN"),
        net("U1.VIN"),
        "the copy's VIN is a net of its own"
    );
    assert_eq!(net("U2.GND"), "GND");
    assert_eq!(net("C2.2"), "GND");

    // A part copied alone keeps its rail but not the net it shared with a part left behind.
    let (p3, summary) = p
        .run("schematic.paste", json!({ "items": ["component:C1"] }))
        .unwrap();
    assert_eq!(summary, "pasted C1 as C2 with 1 connection");
    assert!(
        p3.circuit.contains("connect = [\"C1.2\", \"C2.2\"]"),
        "{}",
        p3.circuit
    );
}

#[test]
fn a_pasted_group_names_its_copy_of_a_named_net_after_it() {
    // U1's output joins C1 by the local net +3V3, drawn by its labels.
    let text = BLANK.replace(
        "connect = [U1.VIN, \"C1.1\"]",
        "net \"+3V3\"\n        connect = [\"net.+3V3\", U1.VOUT, \"C1.1\"]",
    );
    let p = Project {
        circuit: text,
        lock: None,
    }
    .step("schematic.annotate", json!({}));
    let (p2, summary) = p
        .run(
            "schematic.paste",
            json!({ "items": ["component:U1", "component:C1"], "dx": 0, "dy": -25.4 }),
        )
        .unwrap();
    assert_eq!(summary, "pasted U1 as U2, C1 as C2 with 1 connection");
    assert!(
        p2.circuit
            .contains("net \"+3V3_2\"\n        connect = [\"net.+3V3_2\", U2.VOUT, \"C2.1\"]"),
        "{}",
        p2.circuit
    );
    let index = p2.index();
    let pins: Vec<_> = index
        .schematic
        .sheets
        .iter()
        .flat_map(|s| s.symbols.iter())
        .flat_map(|s| {
            s.pins
                .iter()
                .map(move |p| (format!("{}.{}", s.designator, p.name), p.net.clone()))
        })
        .collect();
    let net = |pin: &str| {
        pins.iter()
            .find(|(n, _)| n == pin)
            .and_then(|(_, net)| net.clone())
            .unwrap_or_default()
    };
    assert_eq!(net("U2.VOUT"), "+3V3_2");
    assert_eq!(net("C2.1"), "+3V3_2");
    assert_eq!(net("U1.VOUT"), "+3V3", "the originals keep their net");
}

#[test]
fn a_moved_part_keeps_its_wires_and_mirror_flips_it() {
    let p = blank();
    let (x, y, _, _) = p.symbol("C1");
    let p2 = p.step(
        "schematic.move",
        json!({ "items": ["component:C1"], "dx": 25.4, "dy": 12.7, "mirror": true }),
    );
    let (x2, y2, _, mirror) = p2.symbol("C1");
    assert_eq!(((x2 - x) * 100.0).round() / 100.0, 25.4);
    assert_eq!(((y2 - y) * 100.0).round() / 100.0, 12.7);
    assert!(mirror);
    // The wire still joins U1.VIN and C1.1, re-routed to the new place.
    assert_eq!(p2.circuit, p.circuit, "a move is a lock edit only");
    let index = p2.index();
    let page = &index.schematic.sheets[0];
    let pin = |d: &str, n: &str| {
        let s = page.symbols.iter().find(|s| s.designator == d).unwrap();
        let q = s
            .pins
            .iter()
            .find(|q| q.name == n || q.number == n)
            .unwrap();
        ([q.x_mm, q.y_mm], q.net.clone())
    };
    let (vin, net) = pin("U1", "VIN");
    let (c1, net2) = pin("C1", "1");
    assert_eq!(net, net2);
    let ends: Vec<[f64; 2]> = page
        .wires
        .iter()
        .flat_map(|w| [w.points[0], *w.points.last().unwrap()])
        .collect();
    assert!(
        ends.contains(&vin) && ends.contains(&c1),
        "the wire reaches both pins after the move"
    );
}

/// A root sheet with one child instance whose name isn't its sheet's (`sens` draws `sensors`).
const NESTED: &str = r#"runmat 0.7
std 1

circuit reg {
    root = main

    sheet sensors {
        component R1 = { part = "R0402-10K" }
    }

    sheet main {
        component U1 = { part = "LDO-3V3-SOT23" }
        instance sens = { sheet = sensors }
    }
}
"#;

#[test]
fn a_part_placed_on_a_child_page_lands_on_the_sheet_that_instance_draws() {
    let p = Project {
        circuit: NESTED.to_owned(),
        lock: None,
    };
    // The page as the sheet tab and the Outline's hierarchy row pick it (`sheet:/sens`), and a
    // later page of the same instance (`#2`).
    for page in ["sheet:/sens", "sheet:/sens#2"] {
        let request = EditRequest {
            command: "schematic.place-part".into(),
            values: json!({ "part": "C0603-1U-50V", "at": ["point:50,50"] })
                .as_object()
                .cloned()
                .unwrap(),
            selection: vec![page.to_owned()],
            options: CompileOptions::default(),
        };
        let out = runtime::edit(PATH, &p.circuit, &p.files(), &request)
            .unwrap_or_else(|e| panic!("{page}: {}", e.message));
        assert_eq!(
            out.summary, "placed C1 (C0603-1U-50V) on sheet sensors",
            "{page}"
        );
        let text = &out.files.iter().find(|f| f.path == PATH).unwrap().text;
        let sensors =
            &text[text.find("sheet sensors {").unwrap()..text.find("sheet main {").unwrap()];
        assert!(
            sensors.contains("component C1 = { part = \"C0603-1U-50V\" }"),
            "{page}: {text}"
        );
    }
    // The root page places on the root sheet.
    let (root, summary) = p
        .run(
            "schematic.place-part",
            json!({ "part": "C0603-1U-50V", "at": ["point:50,50"] }),
        )
        .unwrap();
    assert_eq!(summary, "placed C1 (C0603-1U-50V) on sheet main");
    assert!(
        root.circuit[root.circuit.find("sheet main {").unwrap()..].contains("component C1"),
        "{}",
        root.circuit
    );
}
