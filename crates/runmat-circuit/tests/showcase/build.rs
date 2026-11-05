//! The command script that builds the showcase board from its schematic.

use runmat_circuit::runtime::{self, CompileOptions, EditRequest, Files};
use runmat_circuit::syntax::board::parse_board;
use serde_json::{json, Value};
use std::path::Path;

pub const CIRCUIT: &str = "/p/daq.circuit";
pub const LAYOUT: &str = "/p/daq.circuit.layout";
pub const LOCK: &str = "/p/daq.circuit.lock";

pub struct Build {
    pub files: Files,
}

impl Build {
    pub fn run(&mut self, command: &str, values: Value, selection: &[&str]) -> String {
        self.run_on(LAYOUT, command, values, selection)
    }

    pub fn run_on(&mut self, path: &str, command: &str, values: Value, selection: &[&str]) -> String {
        let request = EditRequest {
            command: command.into(),
            values: values.as_object().cloned().unwrap_or_default(),
            selection: selection.iter().map(|s| s.to_string()).collect(),
            options: CompileOptions::default(),
        };
        let out = runtime::edit(path, &self.files[path], &self.files, &request)
            .unwrap_or_else(|e| panic!("{command}: {} {:?}", e.message, e.diagnostics));
        for f in out.files {
            self.files.insert(f.path.clone(), f.text.clone());
        }
        eprintln!("{command}: {}", out.summary);
        out.summary
    }

    pub fn board(&self) -> runmat_circuit::layout::Board {
        parse_board(LAYOUT, &self.files[LAYOUT]).board.expect("the layout reads")
    }
}

pub fn build(dir: &Path) {
    let start = include_str!("start.circuit.layout");
    let mut files = Files::new();
    files.insert(CIRCUIT.into(), std::fs::read_to_string(dir.join("daq.circuit")).unwrap());
    files.insert(LAYOUT.into(), start.to_owned());
    let mut b = Build { files };
    b.run_on(CIRCUIT, "schematic.annotate", json!({}), &[]);
    b.run("board.layer-stack", json!({ "layers": 6, "thickness": 1.6 }), &[]);
    let index = runtime::compile(LAYOUT, &b.files[LAYOUT], &b.files, &CompileOptions::default());
    for d in &index.diagnostics {
        eprintln!("diag {:?}", d);
    }
    let eco = &index.index.as_ref().expect("compiles").eco;
    let changes: Vec<String> = eco
        .groups
        .iter()
        .flat_map(|g| g.rows.iter())
        .filter(|r| r.status != "error")
        .map(|r| format!("eco:{}", r.id))
        .collect();
    b.run("schematic.update-board", json!({ "changes": changes }), &[]);
    place(&mut b);
    pours(&mut b);
    route(&mut b);
    report(&b);
    write(&b, dir);
    variants(&b, dir);
    if let Ok(p) = std::env::var("RUNMAT_SHOWCASE_DUMP") {
        std::fs::write(p, &b.files[LAYOUT]).ok();
    }
}

/// Channel 0's parts in its room (room-local millimetres, from the room's lower-left corner):
/// input resistor and filter, the op-amp as a gain-of-two stage, the charge-bucket RC and the
/// ADC, each supply pin's decoupling capacitor beside it.
const CHANNEL: &[(&str, f64, f64, f64)] = &[
    ("R4", 4.0, 9.5, 0.0),
    ("C4", 6.2, 7.6, 270.0),
    ("U6", 9.0, 8.2, 270.0),
    ("R2", 6.6, 5.6, 180.0),
    ("C1", 11.9, 7.3, 0.0),
    ("R1", 12.2, 5.4, 0.0),
    ("R3", 13.4, 9.5, 0.0),
    ("C5", 15.4, 7.6, 270.0),
    ("C6", 17.4, 11.6, 90.0),
    ("U5", 20.0, 8.0, 0.0),
    ("C3", 24.6, 9.4, 0.0),
    ("C2", 24.6, 10.9, 0.0),
];

/// Room corners: x from 9 to 37 mm, 14 mm tall.
const ROOMS: &[(&str, f64)] = &[("ch[0]", 41.0), ("ch[1]", 23.0), ("ch[2]", 5.0)];
const ROOM_X: f64 = 9.0;

/// The rest of the board (absolute millimetres): the analog input header at the left edge,
/// the FPGA in the middle with its I/O row facing the channels, the 1.2 V core regulator above
/// it where the core pour leaves the package, the 3.3 V regulator below, USB-C power at the
/// top edge, the reference between the channels and the FPGA, M3 holes in the corners.
const MAIN: &[(&str, f64, f64, f64)] = &[
    ("J2", 5.0, 30.0, 0.0),
    ("U1", 56.0, 30.0, 90.0),
    ("U4", 42.5, 48.0, 0.0),
    ("C108", 37.6, 48.0, 90.0),
    ("C109", 47.8, 47.6, 90.0),
    ("C110", 47.8, 44.0, 90.0),
    // 1.2 V core: regulator, inductor, output and soft-start, feedback divider.
    ("U3", 74.0, 46.0, 0.0),
    ("L2", 74.0, 41.6, 0.0),
    ("C105", 70.4, 46.5, 90.0),
    ("C106", 78.6, 41.6, 90.0),
    ("R107", 76.9, 44.3, 0.0),
    ("R106", 78.7, 44.3, 0.0),
    ("C107", 77.6, 47.4, 0.0),
    ("C111", 56.0, 40.4, 0.0),
    ("C112", 58.2, 40.4, 0.0),
    ("C113", 60.4, 40.4, 0.0),
    ("C114", 62.6, 40.4, 0.0),
    ("C115", 65.6, 41.0, 90.0),
    // 3.3 V I/O: regulator, inductor, outputs and soft-start, divider, power-good LED.
    ("U2", 76.0, 20.0, 0.0),
    ("L1", 76.0, 15.6, 0.0),
    ("C101", 72.4, 20.5, 90.0),
    ("C102", 80.6, 15.6, 90.0),
    ("C103", 83.2, 15.6, 90.0),
    ("R105", 78.9, 18.3, 0.0),
    ("R104", 80.7, 18.3, 0.0),
    ("C104", 79.6, 21.4, 0.0),
    ("D1", 84.0, 25.0, 0.0),
    ("R101", 84.0, 27.2, 0.0),
    ("C116", 66.8, 23.0, 90.0),
    ("C117", 66.8, 25.0, 90.0),
    ("C118", 66.8, 27.0, 90.0),
    ("C119", 66.8, 29.0, 90.0),
    ("C120", 69.4, 26.0, 90.0),
    // USB-C sink: the receptacle at the top edge, CC pull-downs beside its pins.
    ("J1", 72.0, 56.4, 0.0),
    ("R102", 69.2, 50.4, 90.0),
    ("R103", 74.8, 50.4, 90.0),
    ("H1", 4.0, 4.0, 0.0),
    ("H2", 86.0, 4.0, 0.0),
    ("H3", 4.0, 56.0, 0.0),
    ("H4", 86.0, 56.0, 0.0),
];

fn place_at(b: &mut Build, reference: &str, x: f64, y: f64, rotation: f64) {
    let board = b.board();
    let f = board
        .footprint_by_ref(reference)
        .unwrap_or_else(|| panic!("{reference} is on the board"));
    let (fx, fy) = (to_mm(f.at.x), to_mm(f.at.y));
    let turn = (rotation - f.rotation).rem_euclid(360.0);
    b.run(
        "board.move",
        json!({ "dx": round(x - fx), "dy": round(y - fy), "rotation": turn }),
        &[&format!("component:{reference}")],
    );
}

fn round(v: f64) -> f64 {
    (v * 1e6).round() / 1e6
}

fn place(b: &mut Build) {
    for (name, y0) in ROOMS {
        let (x0, x1, y1) = (ROOM_X, ROOM_X + 28.0, y0 + 14.0);
        b.run(
            "board.room",
            json!({ "name": name, "points": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]] }),
            &[],
        );
    }
    let y0 = ROOMS[0].1;
    for (r, x, y, rot) in CHANNEL {
        place_at(b, r, ROOM_X + x, y0 + y, *rot);
    }
    for (r, x, y, rot) in MAIN {
        place_at(b, r, *x, *y, *rot);
    }
}

fn pours(b: &mut Build) {
    let all = [[0.5, 0.5], [89.5, 0.5], [89.5, 59.5], [0.5, 59.5]];
    for (net, layer) in [
        ("GND", "In1.Cu"),
        ("+5V", "In2.Cu"),
        ("+3V3", "In3.Cu"),
        ("GND", "In4.Cu"),
        ("GND", "F.Cu"),
        ("GND", "B.Cu"),
    ] {
        // Outer-layer pours meet SMD pads solid (reflow needs no relief there); the planes
        // keep thermal reliefs on the through-hole pads.
        let connect = if layer.starts_with("In") { "relief" } else { "relief-tht" };
        b.run(
            "board.pour",
            json!({ "net": format!("net:{net}"), "layer": layer, "points": all, "connect": connect }),
            &[],
        );
    }
    // The 1.2 V core island in the 3.3 V plane: over the FPGA's centre balls, out through the
    // unused I/O between them and the package edge, and over the core regulator above.
    let board = b.board();
    let t = board.footprint_by_ref("U1").expect("U1").transform();
    let local = [
        (-3.6, -3.6),
        (3.6, -3.6),
        (3.6, -1.6),
        (9.5, -1.6),
        (9.5, -26.0),
        (19.5, -26.0),
        (19.5, 1.6),
        (3.6, 1.6),
        (3.6, 3.6),
        (-3.6, 3.6),
    ];
    let points: Vec<[f64; 2]> = local
        .iter()
        .map(|(x, y)| {
            let p = t.apply(runmat_circuit::geom::Point::mm(*x, *y));
            [round(to_mm(p.x)), round(to_mm(p.y))]
        })
        .collect();
    b.run(
        "board.pour",
        json!({ "net": "net:+1V2", "layer": "In3.Cu", "points": points, "priority": 1 }),
        &[],
    );
}

fn route(b: &mut Build) {
    let local: Vec<String> = ["INF", "FB", "AMP", "AIN"]
        .iter()
        .map(|n| format!("net:/ch[0]/{n}"))
        .collect();
    b.run("board.autoroute", json!({ "nets": local }), &[]);
    b.run("board.copy-room-formats", json!({ "source": ["room:ch[0]"] }), &[]);
    let board = b.board();
    let smd: Vec<String> = board
        .footprints
        .values()
        .filter(|f| f.package.pads.iter().any(|p| p.kind == runmat_circuit::library::PadKind::Smd))
        .map(|f| format!("component:{}", f.reference))
        .collect();
    b.run("board.fanout", json!({ "component": smd }), &[]);
    b.run("board.autoroute", json!({}), &[]);
}

/// What the built board's compile says: findings by code and the airwires left.
pub fn report(b: &Build) {
    let out = runtime::compile(LAYOUT, &b.files[LAYOUT], &b.files, &CompileOptions::default());
    for d in &out.diagnostics {
        eprintln!("diagnostic: {} {}", d.path, d.message);
    }
    let index = out.index.expect("the board compiles");
    let board = index.board.as_ref().expect("a board");
    let mut codes: std::collections::BTreeMap<&str, usize> = Default::default();
    for r in &board.drc {
        *codes.entry(r.code.as_str()).or_default() += 1;
    }
    eprintln!("drc {:?}", codes);
    for r in board.drc.iter() {
        eprintln!("  {} {} {:?} {:?}", r.code, r.message, r.items, r.at);
    }
    let parsed = b.board();
    for r in &board.drc {
        for it in &r.items {
            let Some(n) = it.strip_prefix("track ").and_then(|n| n.parse::<u64>().ok()) else {
                continue;
            };
            if let Some(t) = parsed.tracks.get(&runmat_circuit::layout::ItemId(n)) {
                let (a, c) = t.endpoints();
                eprintln!(
                    "    track {n}: {} L{} w{} {a} -> {c}",
                    t.net.map(|x| parsed.net_name(x).to_owned()).unwrap_or_default(),
                    t.layer,
                    to_mm(t.width)
                );
            }
        }
    }
    eprintln!("unrouted {}", board.unrouted.len());
    for a in board.unrouted.iter() {
        eprintln!("  {} {:?} {:?}", a.net, a.from, a.to);
    }
    eprintln!("eco {}", index.eco.total);
    for g in &index.eco.groups {
        for r in &g.rows {
            eprintln!("  {} {} {:?}", g.group, r.id, r.status);
        }
    }
    eprintln!("erc {}", index.erc.len());
    for r in index.erc.iter().take(20) {
        eprintln!("  {} {} {:?}", r.code, r.message, r.items);
    }
}

fn write(b: &Build, dir: &Path) {
    for (path, name) in [(CIRCUIT, "daq.circuit"), (LAYOUT, "daq.circuit.layout"), (LOCK, "daq.circuit.lock")] {
        if let Some(text) = b.files.get(path) {
            std::fs::write(dir.join(name), text).unwrap();
        }
    }
}

fn to_mm(v: i64) -> f64 {
    runmat_circuit::geom::to_mm(v)
}

/// The DRC and ERC views' variants: the board with three seeded violations (a via too close to a
/// header pad, a track under the minimum width, a capacitor pushed into its neighbour), and the
/// schematic with an unsourced rail and an unmarked unused pin.
fn variants(b: &Build, dir: &Path) {
    let mut v = Build {
        files: b.files.clone(),
    };
    v.run(
        "board.place-via",
        json!({ "points": [[3.73, 36.2]], "net": "net:+5V", "size": "0.2/0.45" }),
        &[],
    );
    v.run(
        "board.route",
        json!({ "net": "net:GND", "points": [[40.0, 8.0], [44.0, 8.0]], "width": 0.1, "layer": "Bottom", "mode": "ignore", "corner": "any" }),
        &[],
    );
    v.run(
        "board.move",
        json!({ "dx": 0.0, "dy": -0.75, "rotation": 0.0 }),
        &["component:C2"],
    );
    let layout = v.files[LAYOUT].clone();
    std::fs::write(dir.join("daq-drc.circuit.layout"), layout).unwrap();
    let circuit = b.files[CIRCUIT]
        .replace(
            "net \"+1V2\" = { class = power, power_flag = true }",
            "net \"+1V2\" = { class = power }",
        )
        .replace("U3.PG, U4.TEMP,", "U3.PG,");
    std::fs::write(dir.join("daq-erc.circuit"), circuit).unwrap();
}
