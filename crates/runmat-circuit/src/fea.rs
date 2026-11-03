//! The board → FEA contract: the `pcb` geometry source.
//!
//! [`pcb_source`] turns a board (and, when available, its netlist and pool) into a
//! [`PcbFeaSource`]: a layered description with **named regions**, a material per layer and
//! a power map. FEA builds layered solids from it and matches regions by their canonical IDs.
//!
//! # Region naming (the contract)
//!
//! Every region name is a path under `pcb/<board>/`. Segments are names authored in the
//! design (layer names from the stackup, net names, designators); `/`, `%` and whitespace
//! inside a segment are percent-encoded, nothing else is changed. The canonical ID of a
//! region is [`canonical_id`] of its full name (a name-derived UUID), so the same authored
//! name always gives the same ID and no alias layer is needed.
//!
//! | Region | Name |
//! |---|---|
//! | the whole board | `pcb/<board>/board` |
//! | all copper of a layer | `pcb/<board>/copper/<layer>` (e.g. `copper/In1.Cu`) |
//! | copper of one net on one layer | `pcb/<board>/copper/<layer>/net/<net>` |
//! | a dielectric | `pcb/<board>/dielectric/<stackup layer name>` |
//! | solder mask | `pcb/<board>/mask/<stackup layer name>` |
//! | via barrels of a net | `pcb/<board>/via/<net>` |
//! | a component body | `pcb/<board>/component/<designator>` |
//!
//! Units are SI in the data (metres, watts, K/W, W/(m·K)); geometry polygons stay in board
//! nanometres with z measured down from the top surface.

use crate::elab::Netlist;
use crate::geom::poly::{boolean, BoolOp, PolySet, Polygon, Ring};
use crate::geom::{Coord, Point};
use crate::ids::{fnv128, Uuid};
use crate::layout::{Board, Material, Side, StackKind};
use crate::library::Pool;
use std::collections::BTreeMap;

/// Percent-encodes `/`, `%` and whitespace in one path segment.
pub fn segment(s: &str) -> String {
    let mut o = String::new();
    for c in s.chars() {
        match c {
            '/' => o.push_str("%2F"),
            '%' => o.push_str("%25"),
            c if c.is_whitespace() => o.push_str(&format!("%{:02X}", c as u32)),
            c => o.push(c),
        }
    }
    o
}

/// Canonical region ID of a region name.
pub fn canonical_id(name: &str) -> Uuid {
    Uuid(fnv128(format!("runmat-region|{name}").as_bytes())).with_version(8)
}

#[derive(Clone, Debug, PartialEq)]
pub enum LayerKind {
    Copper,
    Dielectric,
    Mask,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FeaLayer {
    pub region: String,
    pub id: Uuid,
    pub name: String,
    pub kind: LayerKind,
    /// Depth of the layer's top and bottom faces below the board's top surface, m.
    pub z_top: f64,
    pub z_bottom: f64,
    pub material: Material,
    /// Copper layers: fraction of the board area covered by copper (for effective
    /// orthotropic conductivity when copper is not meshed explicitly).
    pub copper_fraction: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CopperRegion {
    pub region: String,
    pub id: Uuid,
    pub layer: String,
    pub net: Option<String>,
    pub area: PolySet,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ViaRegion {
    pub region: String,
    pub id: Uuid,
    pub net: String,
    /// `(x, y, drill)` in nm, and the barrel span as depths in m.
    pub barrels: Vec<(Point, Coord, f64, f64)>,
    /// Plating thickness, m.
    pub plating: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PowerSource {
    /// Authored on the component (`power`).
    Authored,
    /// From a SPICE operating point (I·V summed over the part's pins).
    OperatingPoint,
    /// Not known; the study must supply it.
    Unknown,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ComponentRegion {
    pub region: String,
    pub id: Uuid,
    pub designator: String,
    pub side: Side,
    /// Footprint outline on the board (courtyard, else pad bounding box), nm.
    pub outline: Ring,
    /// Body height, m.
    pub height: f64,
    pub power: Option<f64>,
    pub power_source: PowerSource,
    /// Junction-to-case and junction-to-board thermal resistance, K/W.
    pub theta_jc: Option<f64>,
    pub theta_jb: Option<f64>,
    pub tj_max: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PcbFeaSource {
    pub board: String,
    pub region: String,
    pub id: Uuid,
    pub outline: Polygon,
    pub thickness: f64,
    pub layers: Vec<FeaLayer>,
    pub copper: Vec<CopperRegion>,
    pub vias: Vec<ViaRegion>,
    pub components: Vec<ComponentRegion>,
    /// Region name → dissipated power, W (components with known power only).
    pub power_map: BTreeMap<String, f64>,
}

/// Builds the FEA source. `op_point` gives per-designator dissipation from a SPICE operating
/// point (used when no power is authored).
pub fn pcb_source(
    board: &Board,
    nl: Option<&Netlist>,
    pool: Option<&Pool>,
    op_point: &BTreeMap<String, f64>,
) -> PcbFeaSource {
    let base = format!("pcb/{}", segment(&board.name));
    let reg = |p: &str| format!("{base}/{p}");
    let n = board.copper_count();
    let board_area = board.outline.area().max(1.0);
    let mut layers = Vec::new();
    let mut z = 0.0;
    let mut copper_by_layer: BTreeMap<u8, BTreeMap<Option<String>, Vec<Ring>>> = BTreeMap::new();
    let tol = crate::geom::mm(0.01);
    for p in board.all_pads() {
        for l in &p.layers {
            copper_by_layer
                .entry(*l)
                .or_default()
                .entry(p.net.map(|x| board.net_name(x).to_string()))
                .or_default()
                .extend(p.prim.outline(0, tol));
        }
    }
    for t in board.tracks.values() {
        for pr in t.prims() {
            copper_by_layer
                .entry(t.layer)
                .or_default()
                .entry(t.net.map(|x| board.net_name(x).to_string()))
                .or_default()
                .extend(pr.outline(0, tol));
        }
    }
    for v in board.vias.values() {
        for l in v.layers() {
            copper_by_layer
                .entry(l)
                .or_default()
                .entry(v.net.map(|x| board.net_name(x).to_string()))
                .or_default()
                .push(crate::geom::poly::circle_ring(v.at, v.diameter / 2, tol));
        }
    }
    for zn in board.zones.values() {
        for (l, f) in &zn.fill {
            copper_by_layer
                .entry(*l)
                .or_default()
                .entry(zn.net.map(|x| board.net_name(x).to_string()))
                .or_default()
                .extend(f.rings());
        }
    }
    let mut copper = Vec::new();
    let mut ci = 0u8;
    for sl in &board.stackup.layers {
        let t = sl.thickness as f64 * 1e-9;
        let (kind, region) = match sl.kind {
            StackKind::Copper { .. } => (
                LayerKind::Copper,
                reg(&format!("copper/{}", segment(&sl.name))),
            ),
            StackKind::SolderMask => (LayerKind::Mask, reg(&format!("mask/{}", segment(&sl.name)))),
            _ => (
                LayerKind::Dielectric,
                reg(&format!("dielectric/{}", segment(&sl.name))),
            ),
        };
        let mut fraction = None;
        if kind == LayerKind::Copper {
            let mut total = 0.0;
            if let Some(by_net) = copper_by_layer.get(&ci) {
                for (net, rings) in by_net {
                    let area = boolean(rings, &[], BoolOp::Union).intersection(&PolySet {
                        polys: vec![board.outline.clone()],
                    });
                    total += area.area();
                    let name = match net {
                        Some(nn) => format!("{region}/net/{}", segment(nn)),
                        None => format!("{region}/net/%00unconnected"),
                    };
                    copper.push(CopperRegion {
                        id: canonical_id(&name),
                        region: name,
                        layer: sl.name.clone(),
                        net: net.clone(),
                        area,
                    });
                }
            }
            fraction = Some((total / board_area).clamp(0.0, 1.0));
            ci += 1;
        }
        layers.push(FeaLayer {
            id: canonical_id(&region),
            region,
            name: sl.name.clone(),
            kind,
            z_top: z,
            z_bottom: z + t,
            material: sl.material.clone(),
            copper_fraction: fraction,
        });
        z += t;
    }
    let depth = |l: u8| {
        board
            .stackup
            .copper_z(l)
            .map(|v| v as f64 * 1e-9)
            .unwrap_or(0.0)
    };
    let mut vias: BTreeMap<String, ViaRegion> = BTreeMap::new();
    for v in board.vias.values() {
        let Some(net) = v.net else { continue };
        let nn = board.net_name(net).to_string();
        let name = reg(&format!("via/{}", segment(&nn)));
        vias.entry(nn.clone())
            .or_insert_with(|| ViaRegion {
                id: canonical_id(&name),
                region: name,
                net: nn,
                barrels: vec![],
                plating: 25e-6,
            })
            .barrels
            .push((v.at, v.drill, depth(v.from), depth(v.to)));
    }
    let mut components = Vec::new();
    let mut power_map = BTreeMap::new();
    for f in board.footprints.values().filter(|f| f.fitted) {
        let name = reg(&format!("component/{}", segment(&f.reference)));
        let comp = nl.and_then(|nl| nl.components.iter().find(|c| c.designator == f.reference));
        let part = comp
            .and_then(|c| c.part)
            .and_then(|u| pool.and_then(|p| p.parts.get(&u)))
            .map(|p| pool.unwrap().effective_part(p));
        let (power, source) = match (comp.and_then(|c| c.power), op_point.get(&f.reference)) {
            (Some(q), _) => (Some(q.value), PowerSource::Authored),
            (None, Some(w)) => (Some(*w), PowerSource::OperatingPoint),
            _ => (None, PowerSource::Unknown),
        };
        let mut outline = f
            .courtyard()
            .into_iter()
            .next()
            .unwrap_or_else(|| crate::geom::poly::rect_ring(f.bbox()));
        if crate::geom::poly::area2(&outline) < 0 {
            outline.reverse();
        }
        if let Some(w) = power {
            power_map.insert(name.clone(), w);
        }
        components.push(ComponentRegion {
            id: canonical_id(&name),
            region: name,
            designator: f.reference.clone(),
            side: f.side,
            outline,
            height: f
                .package
                .body
                .map(|b| b.height as f64 * 1e-9)
                .unwrap_or(1e-3),
            power,
            power_source: source,
            theta_jc: part.as_ref().and_then(|p| p.thermal.theta_jc),
            theta_jb: part.as_ref().and_then(|p| p.thermal.theta_jb),
            tj_max: part.as_ref().and_then(|p| p.thermal.tj_max),
        });
    }
    let _ = n;
    let region = reg("board");
    PcbFeaSource {
        board: board.name.clone(),
        id: canonical_id(&region),
        region,
        outline: board.outline.clone(),
        thickness: board.stackup.thickness() as f64 * 1e-9,
        layers,
        copper,
        vias: vias.into_values().collect(),
        components,
        power_map,
    }
}
