//! `study.new` on a board: a new `.fea` study beside the layout whose geometry is the board
//! itself, read through FEA's `pcb` source (the form FEA set in
//! sister-comms/2026-11-10-design-to-fea-pcb-board-studies.md). The study names the layout,
//! never a copy of it: the stackup, copper, component blocks and power come from the layout and
//! the circuit each time FEA reads it, so an edit to the board reaches the study unchanged.
//!
//! - **Thermal:** the board with component blocks, the stackup homogenized, each component's
//!   authored power on its block, and still-air convection on both faces.
//! - **IR drop:** the copper of the board's main power net, a fixed potential on its source pad
//!   (a connector or wire pad) and a current drawn at the load pin farthest from it.
//! - **Warpage:** the board as a layered shell cooling from the reflow peak (245 °C) to 20 °C on
//!   a 3-2-1 support, with bow and twist checked against the IPC-6012 0.75 % limit.

use super::edit::{EditError, EditOutput, EditRequest};
use super::{resolve_path, Files, FileChange, Project};
use crate::fea::{pcb_source, PadRegion, PcbFeaSource};
use crate::layout::Board;
use serde_json::Value;
use std::collections::BTreeMap;
use std::fmt::Write;

/// The physics a board study takes, as the `physics` input names them.
pub const BOARD_PHYSICS: &[&str] = &["thermal", "ir_drop", "warpage"];

/// Runs `study.new` on the board of `project`: one new file, `<board>_<physics>.fea`.
pub(super) fn run(project: &Project, files: &Files, request: &EditRequest) -> Result<EditOutput, EditError> {
    let layout = project.layout_or_err()?;
    let board = project.board().expect("read");
    let physics = request.values.get("physics").and_then(Value::as_str).unwrap_or("thermal");
    if !BOARD_PHYSICS.contains(&physics) {
        return Err(EditError::new(format!(
            "a board study is Thermal, IR drop or Warpage; `{physics}` studies run on a part or an assembly"
        )));
    }
    let layout_name = layout.path.rsplit('/').next().unwrap_or(&layout.path).to_owned();
    let stem = layout_name.strip_suffix(super::LAYOUT_SUFFIX).unwrap_or(&layout_name);
    let name = match request.values.get("name").and_then(Value::as_str).map(str::trim) {
        Some(n) if !n.is_empty() && n != "study.fea" => {
            if n.ends_with(".fea") { n.to_owned() } else { format!("{n}.fea") }
        }
        _ => format!("{stem}_{physics}.fea"),
    };
    let path = resolve_path(&layout.path, &format!("./{name}"));
    if files.contains_key(&path) {
        return Err(EditError::new(format!("{name} already exists; open it, or give the new study another name")));
    }
    let source = pcb_source(board, Some(&project.netlist), Some(&project.pool), &BTreeMap::new());
    let text = study_text(board, &source, &layout_name, physics).map_err(EditError::new)?;
    Ok(EditOutput {
        summary: format!("New {} study {name} on {layout_name}", label(physics)),
        files: vec![FileChange { path, text, edits: Vec::new(), reprinted: true }],
    })
}

fn label(physics: &str) -> &'static str {
    match physics {
        "ir_drop" => "IR drop",
        "warpage" => "warpage",
        _ => "thermal",
    }
}

/// The study's text for `physics` on the board `source` was read from (`layout`: the layout's
/// file name, which the study names relative to itself).
pub fn study_text(board: &Board, source: &PcbFeaSource, layout: &str, physics: &str) -> Result<String, String> {
    let id = identifier(&board.name);
    match physics {
        "thermal" => Ok(thermal(&id, source, layout)),
        "ir_drop" => ir_drop(&id, board, source, layout),
        "warpage" => Ok(warpage(&id, source, layout)),
        other => Err(format!("`{other}` is not a board study")),
    }
}

/// An FEA identifier from a board name: lower case, `_` for anything else, a letter first.
fn identifier(name: &str) -> String {
    let mut id: String = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '_' })
        .collect();
    if !id.starts_with(|c: char| c.is_ascii_alphabetic()) {
        id.insert_str(0, "board_");
    }
    id
}

/// `1.2` from 1.2, `10` from 10.0: a decimal with no trailing zeros.
fn num(v: f64) -> String {
    let s = format!("{v:.3}");
    s.trim_end_matches('0').trim_end_matches('.').to_owned()
}

/// A tetrahedron size that puts about 60 elements across the board's diagonal, from 0.5 to 3 mm.
fn mesh_size(source: &PcbFeaSource) -> f64 {
    let pts = &source.outline.outer;
    let (mut lo, mut hi) = ((f64::MAX, f64::MAX), (f64::MIN, f64::MIN));
    for p in pts {
        lo = (lo.0.min(p.x as f64), lo.1.min(p.y as f64));
        hi = (hi.0.max(p.x as f64), hi.1.max(p.y as f64));
    }
    let diagonal_mm = if pts.is_empty() { 50.0 } else { ((hi.0 - lo.0).hypot(hi.1 - lo.1)) * 1e-6 };
    ((diagonal_mm / 60.0).clamp(0.5, 3.0) * 10.0).round() / 10.0
}

fn region(out: &mut String, name: &str, role: &str, select: &str, expect: &str) {
    let _ = writeln!(out, "    region {name} : {role} {{ select name \"{select}\"; expect {expect} }}");
}

fn thermal(id: &str, source: &PcbFeaSource, layout: &str) -> String {
    let powered: Vec<_> = source.components.iter().filter(|c| c.power.is_some()).collect();
    let total: f64 = powered.iter().filter_map(|c| c.power).sum();
    let unknown = source.components.len() - powered.len();
    let mut o = String::new();
    let _ = writeln!(o, "// Board thermal: the steady temperature of {} under its components' power, cooled by still", source.board);
    let _ = writeln!(o, "// air on both faces.");
    let _ = writeln!(o, "//");
    let _ = writeln!(o, "// The board is read from {layout} through the pcb source: its outline, its stackup");
    let _ = writeln!(o, "// homogenized into in-plane and through-plane conductivity, a block for each component, and");
    let _ = writeln!(o, "// each component's power from the circuit. Board edits reach this study as they are made.");
    let _ = writeln!(
        o,
        "// {} of {} components have an authored power ({} W in all). The other {unknown} dissipate nothing here:",
        powered.len(),
        source.components.len(),
        num(total)
    );
    let _ = writeln!(o, "// give a component `power` in the circuit to include it.");
    let _ = writeln!(o);
    let _ = writeln!(o, "geometry {id} {{");
    let _ = writeln!(o, "    pcb = \"./{layout}\"");
    let _ = writeln!(o, "    body = board");
    let _ = writeln!(o, "    components = blocks");
    region(&mut o, "board", "solid", &source.region, "exactly_one");
    region(&mut o, "top", "face", &format!("{}/surface/top", base(source)), "exactly_one");
    region(&mut o, "bottom", "face", &format!("{}/surface/bottom", base(source)), "exactly_one");
    let _ = writeln!(o, "}}\n");
    mesh(&mut o, id, "tetrahedra", mesh_size(source));
    let _ = writeln!(o, "model board_heat {{");
    let _ = writeln!(o, "    thermal {{ reference_temperature = 298.15 K }}");
    let _ = writeln!(o, "    assign stackup from geometry.{id} {{ treatment = homogenized }}");
    let _ = writeln!(o, "    load components {{ heat power_map from geometry.{id} {{ unknown = 0 W }} }}");
    let _ = writeln!(o, "    // Still air at 25 °C: a heated face that looks up sheds about twice what one facing down does.");
    convection(&mut o, id, "top_air", "top", 10.0);
    convection(&mut o, id, "bottom_air", "bottom", 5.0);
    let _ = writeln!(o, "}}\n");
    let _ = writeln!(o, "study {id}_thermal {{");
    study_head(&mut o, id, "board_heat");
    let _ = writeln!(o, "    analysis heat {{ thermal steady {{}} }}");
    let _ = writeln!(o, "    output temperature = analysis.heat.field.node.temperature");
    let _ = writeln!(o, "    output peak_board_temperature = max(analysis.heat.field.node.temperature, over = geometry.{id}.region.board)");
    let _ = writeln!(o, "    // Per component: peak case temperature, junction = case + P·θJC, and its margin to Tj,max.");
    let _ = writeln!(o, "    output junction_temperature = analysis.heat.pcb.junction_temperature");
    let _ = writeln!(o, "}}");
    o
}

fn base(source: &PcbFeaSource) -> String {
    source.region.strip_suffix("/board").unwrap_or(&source.region).to_owned()
}

fn mesh(o: &mut String, id: &str, kind: &str, size: f64) {
    let _ = writeln!(o, "mesh {id} on geometry.{id} {{");
    let _ = writeln!(o, "    {kind} {{");
    if kind == "tetrahedra" {
        let _ = writeln!(o, "        order = linear");
    }
    let _ = writeln!(o, "        target_size = {} mm", num(size));
    let _ = writeln!(o, "        maximum_deviation = 0.05 mm");
    let _ = writeln!(o, "    }}");
    let _ = writeln!(o, "}}\n");
}

fn convection(o: &mut String, id: &str, name: &str, face: &str, h: f64) {
    let _ = writeln!(o, "    load {name} {{");
    let _ = writeln!(o, "        thermal.convection {{");
    let _ = writeln!(o, "            target = geometry.{id}.region.{face}");
    let _ = writeln!(o, "            ambient_temperature = 298.15 K");
    let _ = writeln!(o, "            coefficient = {} W/m^2/K", num(h));
    let _ = writeln!(o, "        }}");
    let _ = writeln!(o, "    }}");
}

fn study_head(o: &mut String, id: &str, model: &str) {
    let _ = writeln!(o, "    geometry = geometry.{id}");
    let _ = writeln!(o, "    mesh = mesh.{id}");
    let _ = writeln!(o, "    model = model.{model}");
}

/// Whether a net name is a ground (`GND`, `AGND`, `/ch[0]/PGND`, `VSS`, `0V`).
fn is_ground(name: &str) -> bool {
    let n = name.rsplit('/').next().unwrap_or(name).to_ascii_uppercase();
    n.starts_with("GND") || n.ends_with("GND") || matches!(n.as_str(), "VSS" | "0V")
}

/// The board's power nets, ground excluded: nets in a class named for power or supply, else nets
/// named as rails (`VBAT`, `+5V`, `3V3`, `VCC`).
fn power_nets(board: &Board) -> Vec<String> {
    let by_class: Vec<String> = board
        .nets
        .iter()
        .filter(|n| n.class.as_deref().is_some_and(|c| {
            let c = c.to_ascii_lowercase();
            c.contains("power") || c.contains("supply")
        }))
        .map(|n| n.name.clone())
        .filter(|n| !is_ground(n))
        .collect();
    if !by_class.is_empty() {
        return by_class;
    }
    board
        .nets
        .iter()
        .map(|n| n.name.clone())
        .filter(|n| {
            let leaf = n.rsplit('/').next().unwrap_or(n).to_ascii_uppercase();
            !is_ground(n)
                && (leaf.starts_with('+') || leaf.starts_with("VBAT") || leaf.starts_with("VCC") || leaf.starts_with("VIN")
                    || leaf.chars().next().is_some_and(|c| c.is_ascii_digit()) && leaf.contains('V'))
        })
        .collect()
}

/// A designator's letters: `BT` from `BT2`.
fn prefix(designator: &str) -> String {
    designator.chars().take_while(|c| c.is_ascii_alphabetic()).collect::<String>().to_ascii_uppercase()
}

/// The source and load pads of `net`: the source is a connector's or a wire pad's pin (the
/// part with the fewest pads); the load is the pin of an active part (a transistor or an IC)
/// farthest from the source, so the study finds the worst drop.
fn source_and_sink<'a>(source: &'a PcbFeaSource, net: &str) -> Option<(&'a PadRegion, &'a PadRegion)> {
    let on: Vec<&PadRegion> = source.pads.iter().filter(|p| p.net.as_deref() == Some(net)).collect();
    let pad_count = |d: &str| source.pads.iter().filter(|p| p.designator == d).count();
    let from = on
        .iter()
        .filter(|p| matches!(prefix(&p.designator).as_str(), "J" | "P" | "BT" | "CN" | "X"))
        .min_by_key(|p| (pad_count(&p.designator), p.designator.clone(), p.pin.clone()))
        .or_else(|| on.iter().min_by_key(|p| (pad_count(&p.designator), p.designator.clone(), p.pin.clone())))
        .copied()?;
    let distance = |p: &PadRegion| (p.center.x - from.center.x) as f64 * (p.center.x - from.center.x) as f64
        + (p.center.y - from.center.y) as f64 * (p.center.y - from.center.y) as f64;
    let farthest = |active: bool| {
        on.iter()
            .filter(|p| p.designator != from.designator)
            .filter(|p| !active || matches!(prefix(&p.designator).as_str(), "Q" | "U" | "IC" | "M"))
            .max_by(|a, b| distance(a).total_cmp(&distance(b)).then_with(|| b.region.cmp(&a.region)))
            .copied()
    };
    let to = farthest(true).or_else(|| farthest(false))?;
    Some((from, to))
}

fn ir_drop(id: &str, board: &Board, source: &PcbFeaSource, layout: &str) -> Result<String, String> {
    let pads_on = |n: &str| source.pads.iter().filter(|p| p.net.as_deref() == Some(n)).count();
    let net = power_nets(board)
        .into_iter()
        .max_by_key(|n| (pads_on(n), std::cmp::Reverse(n.clone())))
        .ok_or_else(|| format!("{} has no power net to study; put the rail's net in a power net class", source.board))?;
    let (from, to) = source_and_sink(source, &net)
        .ok_or_else(|| format!("{net} reaches only one part, so it has no drop to study"))?;
    let geometry = format!("{id}_{}", identifier(net.rsplit('/').next().unwrap_or(&net)));
    let mut o = String::new();
    let _ = writeln!(o, "// Board IR drop: the DC voltage drop on {net} from {}.{} to {}.{}, the load pin farthest from it.", from.designator, from.pin, to.designator, to.pin);
    let _ = writeln!(o, "//");
    let _ = writeln!(o, "// The net's copper is read from {layout} through the pcb source: a conducting sheet per");
    let _ = writeln!(o, "// copper layer (sheet resistance from the layer's copper thickness) and its vias as barrels");
    let _ = writeln!(o, "// between them. {}.{} holds the rail at 0 V, so each point's potential is its drop. Set the", from.designator, from.pin);
    let _ = writeln!(o, "// load's current and the rail's budget for the design; 1 A and 100 mV are placeholders.");
    let _ = writeln!(o);
    let _ = writeln!(o, "geometry {geometry} {{");
    let _ = writeln!(o, "    pcb = \"./{layout}\"");
    let _ = writeln!(o, "    body = copper");
    let _ = writeln!(o, "    nets = [\"{net}\"]");
    region(&mut o, "source", "face", &from.region, "one_or_more");
    region(&mut o, "load", "face", &to.region, "one_or_more");
    let _ = writeln!(o, "}}\n");
    mesh(&mut o, &geometry, "triangles", (mesh_size(source) / 2.0).max(0.3));
    let _ = writeln!(o, "model rail {{");
    let _ = writeln!(o, "    electromagnetic {{ equation = steady_conduction }}");
    let _ = writeln!(o, "    assign stackup from geometry.{geometry} {{ treatment = layered }}");
    let _ = writeln!(o, "    constraint supply {{ electrical.potential {{ target = geometry.{geometry}.region.source; value = 0 V }} }}");
    let _ = writeln!(o, "    load draw {{ electrical.integral_current {{ target = geometry.{geometry}.region.load; value = 1 A }} }}");
    let _ = writeln!(o, "}}\n");
    let _ = writeln!(o, "study {geometry}_ir_drop {{");
    study_head(&mut o, &geometry, "rail");
    let _ = writeln!(o, "    analysis drop {{ electromagnetic static {{}} }}");
    let _ = writeln!(o, "    output potential = analysis.drop.field.node.electric_potential");
    let _ = writeln!(o, "    output current_density = analysis.drop.field.element.current_density");
    let _ = writeln!(o, "    output worst_drop = max(abs(analysis.drop.field.node.electric_potential))");
    let _ = writeln!(o, "    assessment within_rail_budget {{");
    let _ = writeln!(o, "        value = output.worst_drop");
    let _ = writeln!(o, "        maximum = 100 mV");
    let _ = writeln!(o, "        evidence = authored_samples");
    let _ = writeln!(o, "    }}");
    let _ = writeln!(o, "}}");
    Ok(o)
}

fn warpage(id: &str, source: &PcbFeaSource, layout: &str) -> String {
    let mut o = String::new();
    let _ = writeln!(o, "// Board warpage: the bow and twist of {} as it cools from the reflow peak (245 °C) to 20 °C.", source.board);
    let _ = writeln!(o, "//");
    let _ = writeln!(o, "// The board is read from {layout} through the pcb source as a layered shell: a ply per");
    let _ = writeln!(o, "// stackup layer with its own thickness, stiffness and expansion, so copper against glass-epoxy");
    let _ = writeln!(o, "// bends the board as it cools. A 3-2-1 support holds it against rigid motion only, so it bows");
    let _ = writeln!(o, "// freely. IPC-6012 allows 0.75 % of the diagonal for boards with surface-mount parts.");
    let _ = writeln!(o);
    let _ = writeln!(o, "geometry {id} {{");
    let _ = writeln!(o, "    pcb = \"./{layout}\"");
    let _ = writeln!(o, "    body = board");
    let _ = writeln!(o, "    components = none");
    region(&mut o, "board", "face", &source.region, "exactly_one");
    let _ = writeln!(o, "}}\n");
    mesh(&mut o, id, "triangles", mesh_size(source));
    let _ = writeln!(o, "model reflow {{");
    let _ = writeln!(o, "    thermomechanical {{ reference_temperature = 518.15 K }}");
    let _ = writeln!(o, "    assign stackup from geometry.{id} {{ treatment = layered }}");
    let _ = writeln!(o, "    constraint support {{ structural.three_two_one {{ target = geometry.{id}.region.board }} }}");
    let _ = writeln!(o, "    load cooled {{ temperature_field on geometry.{id}.region.board = 293.15 K }}");
    let _ = writeln!(o, "    load_case reflow {{ include load.cooled }}");
    let _ = writeln!(o, "}}\n");
    let _ = writeln!(o, "study {id}_warpage {{");
    study_head(&mut o, id, "reflow");
    let _ = writeln!(o, "    analysis cooled {{ thermomechanical steady {{}} }}");
    let _ = writeln!(o, "    output displacement = analysis.cooled.field.node.displacement");
    let _ = writeln!(o, "    // Bow and twist as fractions of the board's diagonal.");
    let _ = writeln!(o, "    output bow = analysis.cooled.pcb.bow");
    let _ = writeln!(o, "    output twist = analysis.cooled.pcb.twist");
    for what in ["bow", "twist"] {
        let _ = writeln!(o, "    assessment {what}_within_ipc_6012 {{");
        let _ = writeln!(o, "        value = output.{what}");
        let _ = writeln!(o, "        maximum = 0.0075");
        let _ = writeln!(o, "        evidence = authored_samples");
        let _ = writeln!(o, "    }}");
    }
    let _ = writeln!(o, "}}");
    o
}
