//! A board's mass properties: the stackup's layers and copper, and the placed components.
//!
//! - Dielectrics and solder mask: the outline's area times the layer's thickness at its
//!   material's density.
//! - Copper: each copper layer's copper area (pads, tracks, vias, pours, as
//!   [`crate::fea::pcb_source`] collects it) times the layer's thickness at copper's density.
//! - Components: the mass the design gives (a `mass` property on the placed component, else a
//!   `mass` param on its library part, such as `mass = 0.35 g`). Without one, the package's
//!   body volume at [`PACKAGE_DENSITY`]; a package with no body takes its footprint outline at
//!   1 mm.
//!
//! Each item is a box for its inertia: a layer over the outline's bounding box, a component over
//! its body. The frame is the board's: x and y in metres from the board origin, z = 0 at the
//! board's bottom face (the 3D model's frame, [`crate::board3d`]).

use crate::fea::{pcb_source, LayerKind};
use crate::geom::poly::Ring;
use crate::layout::{Board, Side, StackKind};
use crate::elab::Netlist;
use crate::library::Pool;
use std::collections::BTreeMap;

/// Density of a component without its own mass: a molded package with its lead frame, kg/m³.
pub const PACKAGE_DENSITY: f64 = 2000.0;
/// Copper, kg/m³.
pub const COPPER_DENSITY: f64 = 8960.0;

/// Where an item's mass comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MassSource {
    /// The stackup: area, thickness and material density.
    Stackup,
    /// The placed component's `mass` property.
    Component,
    /// The library part's `mass` param.
    Library,
    /// The package body at [`PACKAGE_DENSITY`].
    PackageDefault,
}

impl MassSource {
    pub fn as_str(self) -> &'static str {
        match self {
            MassSource::Stackup => "stackup",
            MassSource::Component => "component",
            MassSource::Library => "library",
            MassSource::PackageDefault => "package default",
        }
    }
}

/// One item's share: kg, its centre (m, board frame) and its box's inertia about that centre.
#[derive(Clone, Debug, PartialEq)]
pub struct MassItem {
    /// `dielectric/<layer>`, `mask/<layer>`, `copper/<layer>` or `component/<designator>`.
    pub name: String,
    pub mass: f64,
    pub center: [f64; 3],
    pub inertia: [[f64; 3]; 3],
    pub source: MassSource,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct BoardMass {
    pub mass: f64,
    pub center: [f64; 3],
    /// About `center`, board axes, kg·m².
    pub inertia: [[f64; 3]; 3],
    pub items: Vec<MassItem>,
}

/// Area (m²) and centroid (m) of rings in board nanometres; holes wind clockwise and subtract.
fn area_centroid<'a>(rings: impl IntoIterator<Item = &'a Ring>) -> (f64, [f64; 2]) {
    let (mut a, mut cx, mut cy) = (0.0, 0.0, 0.0);
    for r in rings {
        for i in 0..r.len() {
            let (p, q) = (r[i], r[(i + 1) % r.len()]);
            let (px, py, qx, qy) = (p.x as f64 * 1e-9, p.y as f64 * 1e-9, q.x as f64 * 1e-9, q.y as f64 * 1e-9);
            let c = px * qy - qx * py;
            a += c;
            cx += (px + qx) * c;
            cy += (py + qy) * c;
        }
    }
    if a.abs() < 1e-30 {
        return (0.0, [0.0; 2]);
    }
    (a / 2.0, [cx / (3.0 * a), cy / (3.0 * a)])
}

fn box_inertia(m: f64, [a, b, c]: [f64; 3]) -> [[f64; 3]; 3] {
    let k = m / 12.0;
    [[k * (b * b + c * c), 0.0, 0.0], [0.0, k * (a * a + c * c), 0.0], [0.0, 0.0, k * (a * a + b * b)]]
}

/// A mass written as a quantity's text: `0.35 g`, `2 mg`, `0.01 kg`.
fn mass_text(t: &str) -> Option<f64> {
    let t = t.trim();
    let split = t.find(|c: char| !(c.is_ascii_digit() || c == '.' || c == 'e' || c == '-' || c == '+')).unwrap_or(t.len());
    let v: f64 = t[..split].trim().parse().ok()?;
    let scale = match t[split..].trim() {
        "kg" => 1.0,
        "g" => 1e-3,
        "mg" => 1e-6,
        _ => return None,
    };
    Some(v * scale)
}

/// Combines items into the board's total about its centre of mass.
pub fn combine(items: Vec<MassItem>) -> BoardMass {
    let mass: f64 = items.iter().map(|i| i.mass).sum();
    if mass <= 0.0 {
        return BoardMass { items, ..Default::default() };
    }
    let mut center = [0.0; 3];
    for i in &items {
        for k in 0..3 {
            center[k] += i.mass * i.center[k] / mass;
        }
    }
    let mut inertia = [[0.0; 3]; 3];
    for i in &items {
        let d = [i.center[0] - center[0], i.center[1] - center[1], i.center[2] - center[2]];
        let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        for r in 0..3 {
            for c in 0..3 {
                let delta = if r == c { 1.0 } else { 0.0 };
                inertia[r][c] += i.inertia[r][c] + i.mass * (d2 * delta - d[r] * d[c]);
            }
        }
    }
    BoardMass { mass, center, inertia, items }
}

/// The board's mass properties. `netlist` and `pool` give library parts' `mass` params; without
/// them a component takes its own property or its package default.
pub fn board_mass(board: &Board, netlist: Option<&Netlist>, pool: Option<&Pool>) -> BoardMass {
    let source = pcb_source(board, netlist, pool, &BTreeMap::new());
    let thickness = source.thickness;
    let (outline_area, c) = area_centroid(board.outline.rings());
    let bb = board.outline.bbox();
    let (w, l) = ((bb.max.x - bb.min.x) as f64 * 1e-9, (bb.max.y - bb.min.y) as f64 * 1e-9);
    let mut items = Vec::new();
    let mut copper: BTreeMap<&str, Vec<Ring>> = BTreeMap::new();
    for r in &source.copper {
        copper.entry(r.layer.as_str()).or_default().extend(r.area.rings());
    }
    // Silkscreen and paste are art and process layers, not laminate.
    let art: Vec<&str> = board
        .stackup
        .layers
        .iter()
        .filter(|l| matches!(l.kind, StackKind::Silkscreen | StackKind::Paste))
        .map(|l| l.name.as_str())
        .collect();
    for layer in source.layers.iter().filter(|l| !art.contains(&l.name.as_str())) {
        let t = (layer.z_bottom - layer.z_top).abs();
        let z = thickness - (layer.z_top + layer.z_bottom) / 2.0;
        let (name, area, xy, density) = match layer.kind {
            LayerKind::Copper => {
                let (a, xy) = copper.get(layer.name.as_str()).map(|r| area_centroid(r)).unwrap_or((0.0, c));
                (format!("copper/{}", layer.name), a, xy, COPPER_DENSITY)
            }
            LayerKind::Dielectric => (format!("dielectric/{}", layer.name), outline_area, c, layer.material.density),
            LayerKind::Mask => (format!("mask/{}", layer.name), outline_area, c, layer.material.density),
        };
        let m = area * t * density;
        if m > 0.0 {
            items.push(MassItem { name, mass: m, center: [xy[0], xy[1], z], inertia: box_inertia(m, [w, l, t]), source: MassSource::Stackup });
        }
    }
    let part_of = |designator: &str| {
        let comp = netlist?.components.iter().find(|c| c.designator == designator)?;
        let pool = pool?;
        Some(pool.effective_part(pool.parts.get(&comp.part?)?))
    };
    for f in board.footprints.values().filter(|f| f.fitted) {
        let body = f.package.body;
        let (bw, bl, h) = body.map(|b| (b.width as f64 * 1e-9, b.length as f64 * 1e-9, b.height as f64 * 1e-9)).unwrap_or((0.0, 0.0, 1e-3));
        let library = part_of(&f.reference).and_then(|p| match p.params.get("mass").map(|q| &q.value) {
            Some(crate::ir::ParamValue::Quantity { nominal, .. }) if nominal.dim == runmat_rules::Dim::MASS => Some(nominal.value),
            Some(crate::ir::ParamValue::Text(t)) => mass_text(t),
            _ => None,
        });
        let (mass, src) = match (f.properties.get("mass").and_then(|t| mass_text(t)), library) {
            (Some(m), _) => (m, MassSource::Component),
            (None, Some(m)) => (m, MassSource::Library),
            (None, None) => {
                let volume = if body.is_some() {
                    bw * bl * h
                } else {
                    let ring = f.courtyard().into_iter().next().unwrap_or_else(|| crate::geom::poly::rect_ring(f.bbox()));
                    area_centroid([&ring]).0.abs() * h
                };
                (volume * PACKAGE_DENSITY, MassSource::PackageDefault)
            }
        };
        let z = match f.side {
            Side::Top => thickness + h / 2.0,
            Side::Bottom => -h / 2.0,
        };
        let size = if body.is_some() { [bw, bl, h] } else { [0.0, 0.0, h] };
        items.push(MassItem {
            name: format!("component/{}", f.reference),
            mass,
            center: [f.at.x as f64 * 1e-9, f.at.y as f64 * 1e-9, z],
            inertia: box_inertia(mass, size),
            source: src,
        });
    }
    combine(items)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn masses_read_their_units() {
        assert_eq!(mass_text("0.35 g"), Some(0.35e-3));
        assert_eq!(mass_text("2mg"), Some(2e-6));
        assert_eq!(mass_text("1 kg"), Some(1.0));
        assert_eq!(mass_text("3 oz"), None);
    }
}
