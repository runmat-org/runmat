//! The board in 3D for the 3D tab and renders: the substrate at the stack-up's thickness with
//! solder mask on both sides, copper under the mask (tracks, vias), exposed pads in the
//! finish colour, drills, silkscreen (package art and reference designators) and component
//! bodies with their terminals and leads.
//!
//! The model is written as the part display the 3D canvas already draws (bodies of faces with
//! positions and triangles), one body per material with its colour, so the renderer needs no
//! board-specific path. Within a material, each component's pads, body, terminals and leads are
//! their own face referenced `component:<designator>`, the ID the schematic and board tabs
//! select it by, so a selection highlights the part in 3D and a pick in 3D selects it. Outer-layer
//! tracks and vias are a face per net, referenced `net:<name>`, so a selected net lights in 3D;
//! the rest of a material is one face referenced by the material's name. Millimetres; z = 0 is the
//! board's bottom surface.

use crate::geom::shape::Prim;
use crate::geom::{to_mm, Point};
use crate::layer_set::CopperIndex;
use crate::item_ref::ItemRef;
use crate::layout::{Board, Footprint, Side};
use crate::library::PadKind;
use serde::Serialize;
use std::collections::BTreeMap;

mod solid;
pub use solid::*;

/// One face of a body: flat triangle soup.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Face3d {
    pub index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference: Option<String>,
    pub features: Vec<String>,
    pub positions: Vec<f64>,
    pub triangles: Vec<u32>,
}

/// One material of the board, in its colour.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Body3d {
    pub index: usize,
    pub reference: String,
    /// Linear RGBA.
    pub color: [f64; 4],
    pub faces: Vec<Face3d>,
    pub edges: Vec<()>,
    pub vertices: Vec<()>,
    pub triangle_count: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Bounds3d {
    pub min: [f64; 3],
    pub max: [f64; 3],
}

/// The board as a part display.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BoardModel {
    pub bodies: Vec<Body3d>,
    pub bounds: Bounds3d,
}

/// Board colours, as a fabricator's green-mask, white-silk, ENIG board looks.
pub mod colors {
    pub const MASK: [f64; 4] = [0.035, 0.26, 0.13, 1.0];
    pub const COPPER_UNDER_MASK: [f64; 4] = [0.10, 0.40, 0.19, 1.0];
    pub const SUBSTRATE: [f64; 4] = [0.72, 0.64, 0.40, 1.0];
    pub const FINISH: [f64; 4] = [0.86, 0.69, 0.32, 1.0];
    pub const DRILL: [f64; 4] = [0.04, 0.04, 0.04, 1.0];
    pub const SILK: [f64; 4] = [0.95, 0.95, 0.92, 1.0];
    pub const IC: [f64; 4] = [0.11, 0.11, 0.12, 1.0];
    pub const CAPACITOR: [f64; 4] = [0.66, 0.54, 0.36, 1.0];
    pub const RESISTOR: [f64; 4] = [0.09, 0.09, 0.09, 1.0];
    pub const INDUCTOR: [f64; 4] = [0.26, 0.26, 0.28, 1.0];
    pub const LED: [f64; 4] = [0.75, 0.93, 0.78, 1.0];
    pub const METAL: [f64; 4] = [0.78, 0.79, 0.80, 1.0];
    pub const PLASTIC: [f64; 4] = [0.08, 0.08, 0.08, 1.0];
}

/// Mask and silk thicknesses as drawn (they lift each layer just off the one below).
pub(crate) const MASK_T: f64 = 0.02;
const SILK_T: f64 = 0.01;
const TOL: i64 = 20_000;

#[derive(Default)]
struct Mesh {
    positions: Vec<f64>,
    triangles: Vec<u32>,
}

impl Mesh {
    fn vertex(&mut self, x: f64, y: f64, z: f64) -> u32 {
        let k = (self.positions.len() / 3) as u32;
        self.positions.extend([x, y, z]);
        k
    }
    /// A flat convex ring at height `z`, facing up (or down when `down`).
    fn cap(&mut self, ring: &[(f64, f64)], z: f64, down: bool) {
        if ring.len() < 3 {
            return;
        }
        let ccw = signed_area(ring) > 0.0;
        let base = self.vertex(ring[0].0, ring[0].1, z);
        let mut prev = self.vertex(ring[1].0, ring[1].1, z);
        for p in &ring[2..] {
            let next = self.vertex(p.0, p.1, z);
            if ccw != down {
                self.triangles.extend([base, prev, next]);
            } else {
                self.triangles.extend([base, next, prev]);
            }
            prev = next;
        }
    }
    /// A simple (possibly concave) ring at height `z`, by ear clipping.
    fn polygon(&mut self, ring: &[(f64, f64)], z: f64, down: bool) {
        let mut idx: Vec<usize> = (0..ring.len()).collect();
        if signed_area(ring) < 0.0 {
            idx.reverse();
        }
        let verts: Vec<u32> = ring.iter().map(|p| self.vertex(p.0, p.1, z)).collect();
        let mut guard = 0;
        while idx.len() > 3 && guard < 10_000 {
            guard += 1;
            let n = idx.len();
            let mut clipped = false;
            for i in 0..n {
                let (a, b, c) = (idx[(i + n - 1) % n], idx[i], idx[(i + 1) % n]);
                let (pa, pb, pc) = (ring[a], ring[b], ring[c]);
                if cross(pa, pb, pc) <= 0.0 {
                    continue;
                }
                let inside = idx
                    .iter()
                    .any(|&k| k != a && k != b && k != c && in_triangle(ring[k], pa, pb, pc));
                if inside {
                    continue;
                }
                self.tri(verts[a], verts[b], verts[c], down);
                idx.remove(i);
                clipped = true;
                break;
            }
            if !clipped {
                break;
            }
        }
        if idx.len() == 3 {
            self.tri(verts[idx[0]], verts[idx[1]], verts[idx[2]], down);
        }
    }
    fn tri(&mut self, a: u32, b: u32, c: u32, down: bool) {
        if down {
            self.triangles.extend([a, c, b]);
        } else {
            self.triangles.extend([a, b, c]);
        }
    }
    /// Vertical walls of a ring from `z0` up to `z1`.
    fn walls(&mut self, ring: &[(f64, f64)], z0: f64, z1: f64) {
        let ccw = signed_area(ring) > 0.0;
        for k in 0..ring.len() {
            let (p, q) = (ring[k], ring[(k + 1) % ring.len()]);
            let a = self.vertex(p.0, p.1, z0);
            let b = self.vertex(q.0, q.1, z0);
            let c = self.vertex(q.0, q.1, z1);
            let d = self.vertex(p.0, p.1, z1);
            if ccw {
                self.triangles.extend([a, b, c, a, c, d]);
            } else {
                self.triangles.extend([a, c, b, a, d, c]);
            }
        }
    }
    /// A convex prism: ring from `z0` to `z1` with both caps.
    fn prism(&mut self, ring: &[(f64, f64)], z0: f64, z1: f64) {
        self.cap(ring, z1, false);
        self.cap(ring, z0, true);
        self.walls(ring, z0, z1);
    }
}

fn signed_area(r: &[(f64, f64)]) -> f64 {
    let mut a = 0.0;
    for k in 0..r.len() {
        let (p, q) = (r[k], r[(k + 1) % r.len()]);
        a += p.0 * q.1 - q.0 * p.1;
    }
    a / 2.0
}

fn cross(a: (f64, f64), b: (f64, f64), c: (f64, f64)) -> f64 {
    (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)
}

fn in_triangle(p: (f64, f64), a: (f64, f64), b: (f64, f64), c: (f64, f64)) -> bool {
    cross(a, b, p) >= 0.0 && cross(b, c, p) >= 0.0 && cross(c, a, p) >= 0.0
}

fn mm2(p: Point) -> (f64, f64) {
    (to_mm(p.x), to_mm(p.y))
}

fn ring_mm(r: &[Point]) -> Vec<(f64, f64)> {
    r.iter().map(|p| mm2(*p)).collect()
}

/// A rectangle `w × l` centred at `c`, turned by `deg`.
fn rect(c: (f64, f64), w: f64, l: f64, deg: f64) -> Vec<(f64, f64)> {
    let (s, k) = deg.to_radians().sin_cos();
    [(-w, -l), (w, -l), (w, l), (-w, l)]
        .iter()
        .map(|(x, y)| {
            (
                c.0 + (x / 2.0) * k - (y / 2.0) * s,
                c.1 + (x / 2.0) * s + (y / 2.0) * k,
            )
        })
        .collect()
}

struct Model {
    /// By material: its meshes by owner (a component's designator; empty for the board's own
    /// geometry) and its colour.
    meshes: BTreeMap<&'static str, (BTreeMap<String, Mesh>, [f64; 4])>,
    /// The component whose geometry is being written, or empty.
    owner: String,
}

impl Model {
    fn mesh(&mut self, name: &'static str, color: [f64; 4]) -> &mut Mesh {
        let owner = self.owner.clone();
        self.meshes
            .entry(name)
            .or_insert_with(|| (BTreeMap::new(), color))
            .0
            .entry(owner)
            .or_default()
    }
}

/// Body colour and body name by the reference designator's prefix.
fn body_material(f: &Footprint) -> (&'static str, [f64; 4]) {
    let prefix: String = f
        .reference
        .chars()
        .take_while(|c| c.is_ascii_alphabetic())
        .collect();
    match prefix.as_str() {
        "C" => ("Capacitors", colors::CAPACITOR),
        "R" => ("Resistors", colors::RESISTOR),
        "L" => ("Inductors", colors::INDUCTOR),
        "D" | "LED" => ("LEDs", colors::LED),
        "J" | "P" if f.package.name.starts_with("USB") => ("Shells", colors::METAL),
        "J" | "P" => ("Connector bodies", colors::PLASTIC),
        _ => ("Packages", colors::IC),
    }
}

/// The board model.
pub fn board_model(board: &Board) -> BoardModel {
    let top = thickness_mm(board);
    let n = board.copper_count();
    let mut m = Model {
        meshes: BTreeMap::new(),
        owner: String::new(),
    };
    // Substrate: the outline extruded, mask on both faces.
    let outline = ring_mm(&board.outline.outer);
    m.mesh("Board edge", colors::SUBSTRATE)
        .walls(&outline, 0.0, top);
    m.mesh("Solder mask", colors::MASK)
        .polygon(&outline, top, false);
    m.mesh("Solder mask", colors::MASK)
        .polygon(&outline, 0.0, true);
    let lift = |side_top: bool, h: f64| if side_top { top + h } else { -h };
    // Copper under the mask: tracks and vias on the outer layers.
    for t in board.tracks.values() {
        let outer = if t.layer == 0 {
            Some(true)
        } else if t.layer + 1 == n {
            Some(false)
        } else {
            None
        };
        let Some(up) = outer else { continue };
        m.owner = net_owner(board, t.net);
        for prim in t.prims() {
            for r in prim.outline(0, TOL) {
                let z = lift(up, MASK_T * 0.5);
                m.mesh("Copper", colors::COPPER_UNDER_MASK)
                    .cap(&ring_mm(&r), z, !up);
            }
        }
    }
    for v in board.vias.values() {
        m.owner = net_owner(board, v.net);
        for up in [true, false] {
            let ring = Prim::circle(v.at, v.diameter / 2).outline(0, TOL);
            m.mesh("Copper", colors::COPPER_UNDER_MASK).cap(
                &ring_mm(&ring[0]),
                lift(up, MASK_T * 0.6),
                !up,
            );
            let hole = Prim::circle(v.at, v.drill / 2).outline(0, TOL);
            m.mesh("Drills", colors::DRILL)
                .cap(&ring_mm(&hole[0]), lift(up, MASK_T * 0.7), !up);
        }
    }
    m.owner.clear();
    // Pads in the finish; through holes on both sides with their drill.
    for f in board.footprints.values() {
        m.owner = ItemRef::Component(f.reference.clone()).to_string();
        for p in f.pads(n) {
            let both = p.layers.len() > 1;
            let sides: Vec<bool> = if both {
                vec![true, false]
            } else {
                vec![p.layers.contains(&0)]
            };
            for up in sides {
                for r in p.prim.outline(0, TOL) {
                    m.mesh("Pads", colors::FINISH)
                        .cap(&ring_mm(&r), lift(up, MASK_T * 1.1), !up);
                }
                if let Some(d) = p.drill {
                    let hole = Prim::circle(p.center, d / 2).outline(0, TOL);
                    m.mesh("Drills", colors::DRILL).cap(
                        &ring_mm(&hole[0]),
                        lift(up, MASK_T * 1.2),
                        !up,
                    );
                }
            }
        }
    }
    m.owner.clear();
    // Silkscreen: package art and the placed designators.
    let silk_z = |side: Side| lift(side == Side::Top, MASK_T + SILK_T);
    let designators = crate::designators::designators(board);
    for f in board.footprints.values() {
        let geometry = crate::display::footprint(f);
        let up = f.side == Side::Top;
        for s in &geometry.silkscreen {
            stroke(
                m.mesh("Silkscreen", colors::SILK),
                &s.points,
                s.width_mm,
                silk_z(f.side),
                !up,
            );
        }
        if let Some(d) = designators.get(&f.id) {
            let width = to_mm(crate::font::default_thickness(d.size));
            for line in crate::font::strokes(&f.reference, d.at, d.size, 0.0, !up) {
                let pts: Vec<[f64; 2]> = line.iter().map(|p| [to_mm(p.x), to_mm(p.y)]).collect();
                stroke(
                    m.mesh("Silkscreen", colors::SILK),
                    &pts,
                    width,
                    silk_z(f.side),
                    !up,
                );
            }
        }
        m.owner = ItemRef::Component(f.reference.clone()).to_string();
        component(&mut m, f, n, top);
        m.owner.clear();
    }
    for t in board.texts.values() {
        let up = !matches!(t.layer, crate::layout::Layer::Silk(Side::Bottom));
        if !matches!(t.layer, crate::layout::Layer::Silk(_)) {
            continue;
        }
        let width = to_mm(t.thickness);
        for line in crate::font::strokes(&t.text, t.at, t.size, t.angle, !up) {
            let pts: Vec<[f64; 2]> = line.iter().map(|p| [to_mm(p.x), to_mm(p.y)]).collect();
            stroke(
                m.mesh("Silkscreen", colors::SILK),
                &pts,
                width,
                lift(up, MASK_T + SILK_T),
                !up,
            );
        }
    }
    let bb = board.outline.bbox();
    let tallest = board
        .footprints
        .values()
        .filter_map(|f| f.package.body.as_ref().map(|b| to_mm(b.height)))
        .fold(0.0f64, f64::max);
    let bodies = m
        .meshes
        .into_iter()
        .enumerate()
        .map(|(index, (name, (meshes, color)))| {
            let triangle_count = meshes.values().map(|mesh| mesh.triangles.len() / 3).sum();
            Body3d {
                index,
                reference: name.to_owned(),
                color,
                faces: meshes
                    .into_iter()
                    .enumerate()
                    .map(|(i, (owner, mesh))| Face3d {
                        index: i,
                        reference: Some(if owner.is_empty() {
                            name.to_owned()
                        } else {
                            owner
                        }),
                        features: Vec::new(),
                        positions: mesh
                            .positions
                            .iter()
                            .map(|v| (v * 1e4).round() / 1e4)
                            .collect(),
                        triangles: mesh.triangles,
                    })
                    .collect(),
                edges: Vec::new(),
                vertices: Vec::new(),
                triangle_count,
            }
        })
        .collect();
    BoardModel {
        bodies,
        bounds: Bounds3d {
            min: [to_mm(bb.min.x), to_mm(bb.min.y), -tallest.max(0.1)],
            max: [to_mm(bb.max.x), to_mm(bb.max.y), top + tallest.max(0.1)],
        },
    }
}

/// The face owner of a net's copper (`net:GND`), so selecting the net lights its tracks and vias.
fn net_owner(board: &Board, net: Option<crate::layout::NetId>) -> String {
    net.and_then(|n| board.nets.get(n.0 as usize))
        .map(|n| ItemRef::Net(n.name.clone()).to_string())
        .unwrap_or_default()
}

/// A stroked polyline as capsules at height `z`.
fn stroke(mesh: &mut Mesh, pts: &[[f64; 2]], width: f64, z: f64, down: bool) {
    let w = Point::from_f(width.max(0.05) * 1e6, 0.0).x;
    let p = |q: &[f64; 2]| Point::from_f(q[0] * 1e6, q[1] * 1e6);
    if pts.len() == 1 {
        for r in Prim::circle(p(&pts[0]), w / 2).outline(0, TOL) {
            mesh.cap(&ring_mm(&r), z, down);
        }
    }
    for s in pts.windows(2) {
        for r in Prim::segment(p(&s[0]), p(&s[1]), w).outline(0, TOL) {
            mesh.cap(&ring_mm(&r), z, down);
        }
    }
}

/// A two-terminal chip part's body: centred between its pads, along the line through them.
struct Chip {
    mid: (f64, f64),
    along: f64,
    across: f64,
    deg: f64,
}

fn chip(pads: &[crate::layout::PlacedPad], bw: f64, bl: f64) -> Option<Chip> {
    if !(pads.len() == 2 && pads.iter().all(|p| p.kind == PadKind::Smd)) {
        return None;
    }
    let (a, b) = (mm2(pads[0].center), mm2(pads[1].center));
    let len = ((b.0 - a.0).powi(2) + (b.1 - a.1).powi(2)).sqrt();
    Some(Chip {
        mid: ((a.0 + b.0) / 2.0, (a.1 + b.1) / 2.0),
        along: len.max(bl.max(bw)),
        across: bl.min(bw),
        deg: (b.1 - a.1).atan2(b.0 - a.0).to_degrees(),
    })
}

/// A component body's extent as the 3D view draws it: its outline on the board (mm) and the
/// heights it spans (`z0 < z1`, mm, z = 0 at the board's bottom face). A chip part's end caps
/// are inside the outline.
pub type BodyExtent = (Vec<(f64, f64)>, f64, f64);

/// The component outline and lower/upper heights, in millimetres; `None` for a package with no body.
pub fn body_extent(f: &Footprint, n: CopperIndex, thickness: f64) -> Option<BodyExtent> {
    let body = f.package.body.as_ref()?;
    let (bw, bl, h) = (to_mm(body.width), to_mm(body.length), to_mm(body.height));
    if h <= 0.0 {
        return None;
    }
    let (z0, z1) = if f.side == Side::Top {
        (thickness + MASK_T, thickness + MASK_T + h)
    } else {
        (-MASK_T - h, -MASK_T)
    };
    let ring = match chip(&f.pads(n), bw, bl) {
        Some(c) => rect(c.mid, c.along, c.across, c.deg),
        None => rect(mm2(f.at), bw, bl, f.rotation),
    };
    Some((ring, z0, z1))
}

/// The board's thickness as the 3D view draws it (mm).
pub fn thickness_mm(board: &Board) -> f64 {
    to_mm(board.stackup.thickness()).max(0.2)
}

/// A component's body, with metal terminals for chip parts and leads for leaded ICs.
fn component(m: &mut Model, f: &Footprint, n: CopperIndex, top: f64) {
    let Some(body) = &f.package.body else { return };
    let up = f.side == Side::Top;
    let (bw, bl, h) = (to_mm(body.width), to_mm(body.length), to_mm(body.height));
    if h <= 0.0 {
        return;
    }
    let (z0, z1) = if up {
        (top + MASK_T, top + MASK_T + h)
    } else {
        (-MASK_T - h, -MASK_T)
    };
    let c = mm2(f.at);
    let (name, color) = body_material(f);
    let pads = f.pads(n);
    if let Some(Chip {
        mid,
        along,
        across,
        deg,
    }) = chip(&pads, bw, bl)
    {
        // Ceramic body between two metal end caps over the pads.
        let cap = (along * 0.22).max(0.15);
        m.mesh(name, color)
            .prism(&rect(mid, along - 2.0 * cap, across, deg), z0, z1);
        let (s, k) = deg.to_radians().sin_cos();
        for sign in [-1.0, 1.0] {
            let off = sign * (along - cap) / 2.0;
            let at = (mid.0 + off * k, mid.1 + off * s);
            m.mesh("Terminals", colors::METAL).prism(
                &rect(at, cap, across * 1.02, deg),
                z0,
                z1 + 0.005,
            );
        }
        return;
    }
    m.mesh(name, color)
        .prism(&rect(c, bw, bl, f.rotation), z0, z1);
    // Leads from the body edge out to each SMD pad (gull wings drawn as flat strips).
    let bga = f.package.name.starts_with("BGA");
    let lead_h = 0.15f64.min(h * 0.5);
    for p in &pads {
        if bga || p.thermal_pad || p.kind != PadKind::Smd {
            continue;
        }
        let pc = mm2(p.center);
        let rel = (pc.0 - c.0, pc.1 - c.1);
        let (s, k) = (-f.rotation).to_radians().sin_cos();
        let local = (rel.0 * k - rel.1 * s, rel.0 * s + rel.1 * k);
        // Only pads outside the body get a lead.
        if local.0.abs() <= bw / 2.0 && local.1.abs() <= bl / 2.0 {
            continue;
        }
        let bb = p.prim.bbox();
        let (pw, ph) = (to_mm(bb.width()), to_mm(bb.height()));
        let lead = rect(pc, pw * 0.8, ph * 0.8, 0.0);
        let (lz0, lz1) = if up {
            (z0, z0 + lead_h)
        } else {
            (z1 - lead_h, z1)
        };
        m.mesh("Leads", colors::METAL).prism(&lead, lz0, lz1);
    }
    // Header pins: square posts through the plastic.
    if f.package.pads.iter().all(|p| p.kind == PadKind::Tht) && name == "Connector bodies" {
        for p in &pads {
            let pc = mm2(p.center);
            let (pz0, pz1) = if up { (z0, z1 + 3.0) } else { (z0 - 3.0, z1) };
            m.mesh("Pins", colors::FINISH)
                .prism(&rect(pc, 0.64, 0.64, f.rotation), pz0, pz1);
        }
    }
}
