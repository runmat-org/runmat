//! Hand-drawn schematic wires: a wire's corners clicked on the canvas are stored as waypoints in
//! the lock (the `.circuit` file stays connectivity only). The router draws them as stored and
//! routes the rest; they survive a recompile and other edits, a moved part keeps them, and they
//! go when their connection does.

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
        component C1 = { part = "C0603-1U-50V" }
        component C2 = { part = "C0402-1U" }
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

    fn step(&self, command: &str, values: Value) -> Project {
        let request = EditRequest {
            command: command.into(),
            values: values.as_object().cloned().unwrap_or_default(),
            selection: Vec::new(),
            options: CompileOptions::default(),
        };
        let out = runtime::edit(PATH, &self.circuit, &self.files(), &request)
            .unwrap_or_else(|e| panic!("{command}: {}", e.message));
        let mut next = self.clone();
        for f in &out.files {
            if f.path == PATH {
                next.circuit = f.text.clone();
            } else if f.path == LOCK {
                next.lock = Some(f.text.clone());
            }
        }
        next
    }

    fn page(&self) -> runmat_circuit::schematic::SchematicSheet {
        let out = runtime::compile(
            PATH,
            &self.circuit,
            &self.files(),
            &CompileOptions::default(),
        );
        assert!(out.diagnostics.is_empty(), "{:?}", out.diagnostics);
        out.index.expect("compiles").schematic.sheets[0].clone()
    }

    fn tip(&self, designator: &str, pin: &str) -> [f64; 2] {
        let page = self.page();
        let s = page
            .symbols
            .iter()
            .find(|s| s.designator == designator)
            .unwrap();
        let p = s
            .pins
            .iter()
            .find(|p| p.name == pin || p.number == pin)
            .unwrap();
        [p.x_mm, p.y_mm]
    }

    /// The drawn wires of the net `U1.VIN` is on.
    fn vin_wires(&self) -> Vec<Vec<[f64; 2]>> {
        let page = self.page();
        let u1 = page.symbols.iter().find(|s| s.designator == "U1").unwrap();
        let net = u1
            .pins
            .iter()
            .find(|p| p.name == "VIN")
            .unwrap()
            .net
            .clone()
            .expect("on a net");
        page.wires
            .iter()
            .filter(|w| w.net == net)
            .map(|w| w.points.clone())
            .collect()
    }

    fn wire_records(&self) -> Vec<String> {
        self.lock
            .as_deref()
            .unwrap_or_default()
            .lines()
            .filter(|l| l.starts_with("wire "))
            .map(str::to_owned)
            .collect()
    }
}

fn point(p: [f64; 2]) -> String {
    format!("point:{},{}", p[0], p[1])
}

fn round(p: [f64; 2]) -> [f64; 2] {
    [
        (p[0] * 100.0).round() / 100.0,
        (p[1] * 100.0).round() / 100.0,
    ]
}

/// The regulator with its places recorded and the VIN wire drawn by hand: up from U1.VIN by
/// 10.16 mm, across over C1, then down into C1.1.
fn drawn() -> (Project, [[f64; 2]; 2]) {
    let p = Project {
        circuit: BLANK.to_owned(),
        lock: None,
    }
    .step("schematic.annotate", json!({}));
    let (vin, c1) = (p.tip("U1", "VIN"), p.tip("C1", "1"));
    let top = vin[1].max(c1[1]) + 10.16;
    let corners = [round([vin[0], top]), round([c1[0], top])];
    let p = p.step(
        "schematic.place-wire",
        json!({ "from": ["pin:U1.VIN"], "to": ["pin:C1.1"], "path": [point(corners[0]), point(corners[1])], "mode": "ortho" }),
    );
    (p, corners)
}

#[test]
fn a_hand_drawn_wire_keeps_its_corners_through_recompiles_and_edits() {
    let (p, corners) = drawn();
    // The connection was already there: the .circuit file is unchanged and the lock holds the path.
    assert_eq!(p.circuit, BLANK);
    let records = p.wire_records();
    assert_eq!(records.len(), 1, "{records:?}");
    assert!(
        records[0].starts_with("wire main C1.1 U1.VIN ortho "),
        "{}",
        records[0]
    );
    let wires = p.vin_wires();
    assert_eq!(wires.len(), 1, "{wires:?}");
    let pts: Vec<[f64; 2]> = wires[0].iter().map(|q| round(*q)).collect();
    for c in &corners {
        assert!(pts.contains(c), "{c:?} in {pts:?}");
    }
    // A recompile draws the same wire; so does one after an unrelated edit.
    assert_eq!(p.vin_wires(), wires);
    let q = p.step(
        "schematic.place-part",
        json!({ "part": "R0402-10K", "at": "point:40.64,40.64" }),
    );
    assert_eq!(q.wire_records(), records);
    assert_eq!(q.vin_wires(), wires);
}

#[test]
fn a_moved_part_keeps_the_corners_and_its_end_follows_the_pin() {
    let (p, corners) = drawn();
    let moved = p.step(
        "schematic.move",
        json!({ "items": ["component:C1"], "dx": 12.7, "dy": 0 }),
    );
    let c1 = round(moved.tip("C1", "1"));
    let wires = moved.vin_wires();
    assert_eq!(wires.len(), 1, "{wires:?}");
    let pts: Vec<[f64; 2]> = wires[0].iter().map(|q| round(*q)).collect();
    assert!(pts.contains(&corners[0]), "the first corner stays: {pts:?}");
    assert!(
        pts.first() == Some(&c1) || pts.last() == Some(&c1),
        "the wire ends on the moved pin {c1:?}: {pts:?}"
    );
    // Every segment still runs across or up and down.
    for w in pts.windows(2) {
        assert!(w[0][0] == w[1][0] || w[0][1] == w[1][1], "{pts:?}");
    }
    assert_eq!(moved.wire_records().len(), 1);
}

#[test]
fn a_removed_connection_drops_its_hand_drawn_wire() {
    let (p, _) = drawn();
    let cut = p.step("schematic.disconnect", json!({ "pin": ["pin:C1.1"] }));
    assert!(!cut.circuit.contains(r#""C1.1""#), "{}", cut.circuit);
    assert!(cut.wire_records().is_empty(), "{:?}", cut.lock);
    let page = cut.page();
    let c1 = page.symbols.iter().find(|s| s.designator == "C1").unwrap();
    assert!(c1
        .pins
        .iter()
        .find(|q| q.name == "1" || q.number == "1")
        .unwrap()
        .net
        .is_none());
}

#[test]
fn a_45_degree_wire_draws_its_diagonal() {
    let p = Project {
        circuit: BLANK.to_owned(),
        lock: None,
    }
    .step("schematic.annotate", json!({}));
    let (c1, c2) = (p.tip("C1", "2"), p.tip("C2", "1"));
    // One corner 2.54 mm up and 2.54 mm along from C1.2: the wire turns there at 45°.
    let corner = round([c1[0] + 2.54, c1[1] + 2.54]);
    let p = p.step(
        "schematic.place-wire",
        json!({ "from": ["pin:C1.2"], "to": ["pin:C2.1"], "path": [point(corner)], "mode": "diagonal" }),
    );
    assert!(
        p.circuit.contains(r#"connect = ["C1.2", "C2.1"]"#),
        "{}",
        p.circuit
    );
    assert!(
        p.wire_records()[0].contains(" diagonal "),
        "{:?}",
        p.wire_records()
    );
    let page = p.page();
    let c1_net = page
        .symbols
        .iter()
        .find(|s| s.designator == "C1")
        .unwrap()
        .pins
        .iter()
        .find(|q| q.number == "2")
        .unwrap()
        .net
        .clone()
        .unwrap();
    let wire = page.wires.iter().find(|w| w.net == c1_net).expect("a wire");
    let pts: Vec<[f64; 2]> = wire.points.iter().map(|q| round(*q)).collect();
    // The corner is on the drawn path (a corner in line with its neighbours merges into the run).
    let on = |w: &[[f64; 2]]| {
        let (a, b) = (w[0], w[1]);
        let cross = (b[0] - a[0]) * (corner[1] - a[1]) - (b[1] - a[1]) * (corner[0] - a[0]);
        cross.abs() < 1e-6
            && corner[0] >= a[0].min(b[0]) - 1e-6
            && corner[0] <= a[0].max(b[0]) + 1e-6
    };
    assert!(pts.windows(2).any(on), "{corner:?} on {pts:?}");
    let diagonal = pts.windows(2).any(|w| {
        ((w[1][0] - w[0][0]).abs() - (w[1][1] - w[0][1]).abs()).abs() < 1e-6 && w[1][0] != w[0][0]
    });
    assert!(diagonal, "a 45° segment: {pts:?} (C2.1 at {c2:?})");
}
