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

    /// Annotate, set a standard 1.6 mm stack-up of `layers` copper layers (none: keep the start
    /// file's own stack-up) and bring every component onto the board.
    pub fn start(&mut self, layers: Option<u32>) {
        let circuit = self.circuit.clone();
        self.run_on(&circuit, "schematic.annotate", json!({}), &[]);
        if let Some(layers) = layers {
            self.run("board.layer-stack", json!({ "layers": layers, "thickness": 1.6 }), &[]);
        }
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

    /// Puts the component at schematic path `path` on the bottom side, where it stands.
    pub fn flip(&mut self, path: &str) {
        let reference = self.designator(path);
        self.run("board.flip", json!({}), &[&format!("component:{reference}")]);
    }

    pub fn pour(&mut self, net: &str, layer: &str, points: &[[f64; 2]], connect: &str) {
        self.run("board.pour", json!({ "net": format!("net:{net}"), "layer": layer, "points": points, "connect": connect }), &[]);
    }

    /// Fans out every SMD part, then autoroutes what is left, apart from the nets in `skip`.
    pub fn finish(&mut self, skip: &[&str]) {
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
        let nets: Vec<String> = board.nets.iter().filter(|n| !skip.contains(&n.name.as_str())).map(|n| format!("net:{}", n.name)).collect();
        self.run("board.autoroute", json!({ "nets": if skip.is_empty() { Vec::new() } else { nets } }), &[]);
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
        let mut drc: Vec<_> = board.drc.iter().collect();
        drc.sort_by_key(|r| r.code == "edge-clearance");
        let parsed = self.board();
        let describe = |item: &str| -> String {
            let Some(id) = item.strip_prefix("track ").and_then(|n| n.parse::<u64>().ok()) else { return item.to_owned() };
            let Some(t) = parsed.tracks.get(&runmat_circuit::layout::ItemId(id)) else { return item.to_owned() };
            let (a, b) = t.endpoints();
            let net = t.net.map(|n| parsed.net_name(n).to_owned()).unwrap_or_default();
            format!("{item} {net} L{} w{:.2} ({:.3},{:.3})-({:.3},{:.3})", t.layer, to_mm(t.width), to_mm(a.x), to_mm(a.y), to_mm(b.x), to_mm(b.y))
        };
        for r in drc.into_iter().take(40) {
            let items: Vec<String> = r.items.iter().map(|i| describe(i)).collect();
            eprintln!("  {} {} {:?} {:?}", r.code, r.message, items, r.at);
        }
        eprintln!("unrouted {}", board.unrouted.len());
        // Channel 0 and the center column first: the copied channels repeat what they show.
        let copied = |n: &str| ["/ch[1]/", "/ch[2]/", "/ch[3]/"].iter().any(|c| n.starts_with(c));
        let mut wires: Vec<_> = board.unrouted.iter().collect();
        wires.sort_by_key(|a| copied(&a.net));
        for a in wires.into_iter().take(40) {
            eprintln!("  {} {:?} {:?}", a.net, a.from, a.to);
        }
        eprintln!("eco {} erc {}", index.eco.total as usize, index.erc.len());
        for r in index.erc.iter().take(20) {
            eprintln!("  {} {} {:?}", r.code, r.message, r.items);
        }
        (board.drc.len(), board.unrouted.len(), index.eco.total as usize, index.erc.len())
    }

    /// Each footprint's courtyard box in board millimetres and its side, for checking a placement.
    pub fn courtyards(&self) {
        let board = self.board();
        for f in board.footprints.values() {
            let t = f.transform();
            let pts: Vec<_> = f.package.courtyard.iter().flatten().map(|p| t.apply(*p)).collect();
            if pts.is_empty() {
                continue;
            }
            let (x0, x1) = pts.iter().fold((f64::MAX, f64::MIN), |a, p| (a.0.min(to_mm(p.x)), a.1.max(to_mm(p.x))));
            let (y0, y1) = pts.iter().fold((f64::MAX, f64::MIN), |a, p| (a.0.min(to_mm(p.y)), a.1.max(to_mm(p.y))));
            eprintln!("courtyard {} {:?} {x0:.2} {y0:.2} {x1:.2} {y1:.2}", f.reference, f.side);
            for pad in f.pads(board.copper_count()) {
                let r = pad.prim.bbox();
                let net = pad.net.map(|n| board.net_name(n).to_owned()).unwrap_or_default();
                eprintln!("pad {}.{} {:?} {:.3} {:.3} {:.3} {:.3} {net}", f.reference, pad.name, pad.layers, to_mm(r.min.x), to_mm(r.min.y), to_mm(r.max.x), to_mm(r.max.y));
            }
        }
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
