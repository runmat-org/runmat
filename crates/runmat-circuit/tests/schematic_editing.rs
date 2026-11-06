//! Drawing a schematic from the canvas: each tool's run is a small text edit to the
//! `.circuit` file, made through `runmat_circuit::runtime::edit` as the Desktop makes it.
//!
//! A 3.3 V regulator (connector, LDO, input and output capacitors) starts with its parts
//! placed and nothing wired, and is wired, labelled and marked until ERC is clean.

use runmat_circuit::runtime::{self, CompileOptions, EditRequest, Files};
use serde_json::{json, Value};

const PATH: &str = "/p/reg.circuit";

const BLANK: &str = r#"runmat 0.7
std 1

// A 3.3 V rail from a 5 V connector.
circuit reg {
    root = main
    net_class power = { track_width = "0.50 mm" }

    sheet main {
        component J1 = { part = "CONN-SH-4P" }
        component U1 = { part = "LDO-3V3-SOT23" }
        // Input and output capacitors, beside the regulator.
        component C1 = { part = "C0603-1U-50V" }
        component C2 = { part = "C0402-1U" }
    }
}
"#;

fn files(text: &str) -> Files {
    [(PATH.to_owned(), text.to_owned())].into_iter().collect()
}

fn index(text: &str) -> runmat_circuit::index::CircuitIndex {
    let out = runtime::compile(PATH, text, &files(text), &CompileOptions::default());
    assert!(out.diagnostics.is_empty(), "{:?}\n{text}", out.diagnostics);
    out.index.expect("compiles")
}

/// The net a pin is on, as the canvas names it (`net:<name>`).
fn net_of(text: &str, designator: &str, pin: &str) -> String {
    let index = index(text);
    let found = index
        .schematic
        .sheets
        .iter()
        .flat_map(|s| s.symbols.iter())
        .filter(|s| s.designator == designator)
        .flat_map(|s| s.pins.iter())
        .find(|p| p.name == pin)
        .and_then(|p| p.net.clone());
    format!("net:{}", found.unwrap_or_else(|| panic!("{designator}.{pin} is on no net")))
}

fn run(text: &str, command: &str, values: Value) -> Result<(String, String), String> {
    let request = EditRequest {
        command: command.into(),
        values: values.as_object().cloned().unwrap_or_default(),
        selection: Vec::new(),
        options: CompileOptions::default(),
    };
    let out = runtime::edit(PATH, text, &files(text), &request).map_err(|e| e.message)?;
    let next = out.files.iter().find(|f| f.path == PATH).map(|f| f.text.clone()).unwrap_or_else(|| text.to_owned());
    Ok((next, out.summary))
}

fn step(text: &str, command: &str, values: Value) -> String {
    run(text, command, values).unwrap_or_else(|e| panic!("{command}: {e}\n{text}")).0
}

/// Lines added and removed between two texts (a minimal edit touches few lines).
fn changed_lines(a: &str, b: &str) -> usize {
    let (la, lb): (Vec<&str>, Vec<&str>) = (a.lines().collect(), b.lines().collect());
    la.iter().filter(|l| !lb.contains(l)).count() + lb.iter().filter(|l| !la.contains(l)).count()
}

fn wired_regulator() -> String {
    let mut t = BLANK.to_owned();
    // Two free pins get a statement of their own.
    t = step(&t, "schematic.place-wire", json!({ "from": ["pin:J1.1"], "to": ["pin:U1.VIN"] }));
    assert!(t.contains(r#"connect = ["J1.1", U1.VIN]"#), "{t}");
    // Continuing from the wire: the cap joins the same statement.
    let vin = net_of(&t, "U1", "VIN");
    let before = t.clone();
    t = step(&t, "schematic.place-wire", json!({ "from": [vin], "to": ["pin:C1.1"] }));
    assert_eq!(changed_lines(&before, &t), 2, "one statement rewritten:\n{t}");
    assert!(t.contains(r#"connect = ["J1.1", U1.VIN, "C1.1"]"#), "{t}");
    t = step(&t, "schematic.place-wire", json!({ "from": ["pin:U1.EN"], "to": ["pin:U1.VIN"] }));
    // The label names the net; the name leads the statement.
    t = step(&t, "schematic.net-label", json!({ "name": "VIN", "on": ["pin:U1.VIN"] }));
    assert!(t.contains(r#"connect = [net.VIN, "J1.1", U1.VIN, "C1.1", U1.EN]"#), "{t}");
    // Power ports declare the rail once and join it.
    // The connector brings the input and ground onto the board: their power ports carry a
    // power flag, so ERC sees them driven.
    t = step(&t, "schematic.power-port", json!({ "net": "VIN", "on": ["pin:J1.1"], "power_flag": true }));
    assert!(t.contains("net VIN = { global = true, power_flag = true }"), "{t}");
    t = step(&t, "schematic.power-port", json!({ "net": "GND", "on": ["pin:J1.2"], "power_flag": true }));
    assert!(t.contains("net GND = { global = true, power_flag = true }"), "{t}");
    for pin in ["pin:U1.GND", "pin:C1.2", "pin:C2.2"] {
        t = step(&t, "schematic.power-port", json!({ "net": "GND", "on": [pin] }));
    }
    assert_eq!(t.matches("net GND").count(), 1, "{t}");
    assert!(t.contains(r#"connect = [net.GND, "J1.2", U1.GND, "C1.2", "C2.2"]"#), "{t}");
    t = step(&t, "schematic.power-port", json!({ "net": "+3V3", "on": ["pin:U1.VOUT"] }));
    let out = net_of(&t, "U1", "VOUT");
    t = step(&t, "schematic.place-wire", json!({ "from": ["pin:C2.1"], "to": [out] }));
    for pin in ["pin:U1.NC", "pin:J1.3", "pin:J1.4"] {
        t = step(&t, "schematic.no-erc", json!({ "pin": [pin] }));
    }
    t
}

#[test]
fn a_regulator_is_drawn_from_blank_to_wired_and_erc_is_clean() {
    let t = wired_regulator();
    // Comments and the statements no tool touched keep their text.
    assert!(t.contains("// A 3.3 V rail from a 5 V connector."));
    assert!(t.contains("        // Input and output capacitors, beside the regulator.\n"));
    assert!(t.contains(r#"no_erc = [U1.NC, "J1.3", "J1.4"]"#), "{t}");
    let index = index(&t);
    assert!(index.erc.is_empty(), "ERC: {:#?}\n{t}", index.erc);
    // The drawing has the wires and the rails' power ports.
    let page = &index.schematic.sheets[0];
    assert!(page.wires.iter().any(|w| w.net == "VIN"), "{:?}", page.wires);
    assert!(page.labels.iter().any(|l| l.kind == "power" && l.text == "GND"), "{:?}", page.labels);
}

#[test]
fn wires_refuse_what_a_drawing_cant_mean() {
    let t = wired_regulator();
    // Already on one net: nothing to write.
    let (same, summary) = run(&t, "schematic.place-wire", json!({ "from": ["pin:U1.VIN"], "to": ["pin:C1.1"] })).unwrap();
    assert_eq!(same, t);
    assert!(summary.contains("already on"), "{summary}");
    // A wire from a pin to itself, and to a pin that doesn't exist.
    assert!(run(&t, "schematic.place-wire", json!({ "from": ["pin:U1.VIN"], "to": ["pin:U1.VIN"] })).is_err());
    let err = run(&t, "schematic.place-wire", json!({ "from": ["pin:U1.VIN"], "to": ["pin:U9.1"] })).unwrap_err();
    assert!(err.contains("U9"), "{err}");
    // Joining two named rails is the compiler's refusal, and nothing is written.
    let err = run(&t, "schematic.place-wire", json!({ "from": ["pin:C1.1"], "to": ["pin:C2.1"] })).unwrap_err();
    assert!(err.contains("VIN") && err.contains("+3V3"), "{err}");
}

#[test]
fn disconnect_takes_a_pin_out_and_a_lone_pin_leaves_no_statement() {
    let t = wired_regulator();
    let t = step(&t, "schematic.disconnect", json!({ "pin": ["pin:C2.1"] }));
    assert!(!t.contains("\"C2.1\""), "{t}");
    // Re-wiring puts it back.
    let out = net_of(&t, "U1", "VOUT");
    let t = step(&t, "schematic.place-wire", json!({ "from": [out], "to": ["pin:C2.1"] }));
    assert!(t.contains("\"C2.1\""), "{t}");
    // A two-pin statement with one pin out is removed whole.
    let mut b = BLANK.to_owned();
    b = step(&b, "schematic.place-wire", json!({ "from": ["pin:C1.1"], "to": ["pin:C2.1"] }));
    b = step(&b, "schematic.disconnect", json!({ "pin": ["pin:C2.1"] }));
    assert!(!b.contains("connect ="), "{b}");
    assert_eq!(b, BLANK);
}

#[test]
fn a_net_label_renames_a_named_net_and_keeps_its_identity() {
    let mut t = wired_regulator();
    t = step(&t, "schematic.directive", json!({ "type": "net_class", "nets": ["net:VIN"], "value": "power" }));
    assert!(t.contains("net VIN = { class = power, global = true, power_flag = true }"), "{t}");
    t = step(&t, "schematic.net-label", json!({ "name": "VBUS", "on": ["net:VIN"] }));
    assert!(t.contains("connect = [net.VBUS,"), "{t}");
    assert!(t.contains("net VBUS = { class = power, global = true, renamed_from = VIN, power_flag = true }"), "{t}");
    assert!(!t.contains("net.VIN"), "{t}");
    index(&t);
    let err = run(&t, "schematic.directive", json!({ "type": "net_class", "nets": ["net:VBUS"], "value": "nope" })).unwrap_err();
    assert!(err.contains("power"), "{err}");
}

#[test]
fn hierarchy_sheet_symbols_entries_ports_and_channels() {
    let mut t = wired_regulator();
    t = step(&t, "schematic.sheet-symbol", json!({ "name": "led", "sheet": "indicator", "channels": 1 }));
    assert!(t.contains("instance led = { sheet = indicator }"), "{t}");
    assert!(t.contains("sheet indicator {\n    }"), "{t}");
    t = step(&t, "schematic.sheet-entry", json!({ "symbol": ["sheet:/led"], "port": "PWR" }));
    assert!(t.contains("sheet indicator {\n        port PWR : passive\n    }"), "{t}");
    t = step(&t, "schematic.repeat-channel", json!({ "symbol": ["sheet:/led"], "channels": 3 }));
    assert!(t.contains("instance led = { sheet = indicator, repeat = 3 }"), "{t}");
    let index = index(&t);
    assert!(index.schematic.sheets.iter().any(|s| s.path == "/led[2]"), "{:?}", index.schematic.sheets.iter().map(|s| &s.path).collect::<Vec<_>>());
    // A port on the root sheet, joined to the output rail's pin.
    let t = step(&t, "schematic.port", json!({ "name": "OUT", "direction": "output", "on": ["pin:C2.1"] }));
    assert!(t.contains("port OUT : out"), "{t}");
    assert!(t.contains("port.OUT"), "{t}");
}

#[test]
fn buses_expand_their_members() {
    let t = step(&wired_regulator(), "schematic.place-bus", json!({ "name": "D[0..7]" }));
    assert!(t.contains("bus D = [D0, D1, D2, D3, D4, D5, D6, D7]"), "{t}");
    index(&t);
    assert!(run(&t, "schematic.place-bus", json!({ "name": "D[0..7" })).is_err());
}
