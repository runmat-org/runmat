//! The command scripts that build the quadcopter's two boards from their schematics: annotate,
//! stack-up, update the board from the schematic, place (rooms per ESC channel), pour the
//! planes, route the local nets, copy the channel room's formats to the other three channels,
//! fan out and autoroute the rest.

use runmat_circuit::runtime::{self, CompileOptions, EditRequest, Files};
use runmat_circuit::syntax::board::parse_board;
use serde_json::{json, Value};
use std::path::Path;

pub struct Build {
    pub files: Files,
    pub circuit: String,
    pub layout: String,
}

impl Build {
    pub fn new(dir: &Path, name: &str, start: &str) -> Self {
        let mut files = Files::new();
        let circuit = format!("/p/{name}.circuit");
        let layout = format!("/p/{name}.circuit.layout");
        files.insert(circuit.clone(), std::fs::read_to_string(dir.join(format!("{name}.circuit"))).unwrap());
        files.insert(layout.clone(), start.to_owned());
        Build { files, circuit, layout }
    }

    pub fn run(&mut self, command: &str, values: Value, selection: &[&str]) -> String {
        let path = self.layout.clone();
        self.run_on(&path, command, values, selection)
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
        parse_board(&self.layout, &self.files[&self.layout]).board.expect("the layout reads")
    }

    /// Annotate, set the stack-up and bring every component onto the board.
    pub fn start(&mut self, layers: u32) {
        let circuit = self.circuit.clone();
        self.run_on(&circuit, "schematic.annotate", json!({}), &[]);
        self.run("board.layer-stack", json!({ "layers": layers, "thickness": 1.6 }), &[]);
        let index = runtime::compile(&self.layout, &self.files[&self.layout], &self.files, &CompileOptions::default());
        let eco = &index.index.as_ref().expect("compiles").eco;
        let changes: Vec<String> = eco
            .groups
            .iter()
            .flat_map(|g| g.rows.iter())
            .filter(|r| r.status != "error")
            .map(|r| format!("eco:{}", r.id))
            .collect();
        self.run("schematic.update-board", json!({ "changes": changes }), &[]);
    }

    /// The designator of the component at schematic path `path` (`/pwr/U`, or `U1` on the root).
    pub fn designator(&self, path: &str) -> String {
        let board = self.board();
        let want = if path.starts_with('/') { path.to_owned() } else { format!("/{path}") };
        board
            .footprints
            .values()
            .find(|f| f.component.as_deref() == Some(want.as_str()) || f.reference == path)
            .map(|f| f.reference.clone())
            .unwrap_or_else(|| {
                let all: Vec<String> = board.footprints.values().map(|f| format!("{}={}", f.reference, f.component.clone().unwrap_or_default())).collect();
                panic!("no footprint for {path}: {all:?}")
            })
    }

    /// Moves the component at schematic path `path` to (x, y) mm at `rotation` degrees.
    pub fn place(&mut self, path: &str, x: f64, y: f64, rotation: f64) {
        let reference = self.designator(path);
        let board = self.board();
        let f = board.footprint_by_ref(&reference).unwrap();
        let (fx, fy) = (to_mm(f.at.x), to_mm(f.at.y));
        let turn = (rotation - f.rotation).rem_euclid(360.0);
        self.run("board.move", json!({ "dx": round(x - fx), "dy": round(y - fy), "rotation": turn }), &[&format!("component:{reference}")]);
    }

    pub fn pour(&mut self, net: &str, layer: &str, points: &[[f64; 2]], connect: &str) {
        self.run("board.pour", json!({ "net": format!("net:{net}"), "layer": layer, "points": points, "connect": connect }), &[]);
    }

    /// Fans out every SMD part, then autoroutes what is left.
    pub fn finish(&mut self) {
        let board = self.board();
        let smd: Vec<String> = board
            .footprints
            .values()
            .filter(|f| f.package.pads.iter().any(|p| p.kind == runmat_circuit::library::PadKind::Smd))
            // Connectors route straight from their pads.
            .filter(|f| !f.reference.starts_with('J'))
            .map(|f| format!("component:{}", f.reference))
            .collect();
        self.run("board.fanout", json!({ "component": smd }), &[]);
        self.run("board.autoroute", json!({}), &[]);
    }

    /// The compile's findings: DRC by code, airwires left, ECO rows and ERC.
    pub fn report(&self) -> (usize, usize, usize, usize) {
        let out = runtime::compile(&self.layout, &self.files[&self.layout], &self.files, &CompileOptions::default());
        for d in &out.diagnostics {
            eprintln!("diagnostic: {} {}", d.path, d.message);
        }
        let index = out.index.expect("the board compiles");
        let board = index.board.as_ref().expect("a board");
        let mut codes: std::collections::BTreeMap<&str, usize> = Default::default();
        for r in &board.drc {
            *codes.entry(r.code.as_str()).or_default() += 1;
        }
        eprintln!("drc {codes:?}");
        for r in board.drc.iter().take(40) {
            eprintln!("  {} {} {:?} {:?}", r.code, r.message, r.items, r.at);
        }
        eprintln!("unrouted {}", board.unrouted.len());
        for a in board.unrouted.iter().take(40) {
            eprintln!("  {} {:?} {:?}", a.net, a.from, a.to);
        }
        eprintln!("eco {} erc {}", index.eco.total as usize, index.erc.len());
        for r in index.erc.iter().take(20) {
            eprintln!("  {} {} {:?}", r.code, r.message, r.items);
        }
        (board.drc.len(), board.unrouted.len(), index.eco.total as usize, index.erc.len())
    }

    pub fn write(&self, dir: &Path, name: &str) {
        for (path, file) in [
            (self.circuit.clone(), format!("{name}.circuit")),
            (self.layout.clone(), format!("{name}.circuit.layout")),
            (format!("/p/{name}.circuit.lock"), format!("{name}.circuit.lock")),
        ] {
            if let Some(text) = self.files.get(&path) {
                std::fs::write(dir.join(file), text).unwrap();
            }
        }
    }
}

pub fn round(v: f64) -> f64 {
    (v * 1e6).round() / 1e6
}

pub fn to_mm(v: i64) -> f64 {
    runmat_circuit::geom::to_mm(v)
}
