//! The board layout model.
//!
//! - **Outline:** a board outline polygon with cutouts (holes), in nanometres, y up.
//! - **Stackup:** ordered top-to-bottom layers (mask, copper, dielectric, ...) with thickness,
//!   copper weight, material (Dk, Df, conductivity, in-plane and through-plane thermal
//!   conductivity, density, heat capacity, CTE). Named stackups plus **regions** give
//!   rigid-flex: each region of the outline uses one stackup (a subset of the rigid copper
//!   layers), and flex regions carry **bend lines** with radius and angle.
//! - **Placement:** footprints with side (top/bottom), rotation, lock flag, pad nets and an
//!   embedded package copy (embed-on-use, so a project is self-contained).
//! - **Copper:** track segments and arcs, vias (through, blind, buried, micro) by layer span,
//!   pads, zones (copper pours) with priority, clearance, minimum width, thermal relief and
//!   island policy. Filled areas are derived data, computed by [`crate::zone`].
//! - **Other:** keepouts (per layer, per object class, height limit), rooms, text, graphics
//!   on any layer, dimensions.
//!
//! Every item has an [`ItemId`] unique within the board, which the DRC kernel and the
//! spatial index key on.

use crate::geom::poly::{Polygon, Ring};
use crate::geom::shape::Prim;
use crate::geom::{mm, Coord, Point, Rect, Transform};
use crate::library::{CopperSpan, Graphic, Package, PadDef, PadKind, PadShape};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ItemId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NetId(pub u32);

/// Every board layer. Copper layers are numbered top (0) to bottom (n − 1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Layer {
    Copper(u8),
    Mask(Side),
    Paste(Side),
    Silk(Side),
    Fab(Side),
    Courtyard(Side),
    Edge,
    User(u8),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum Side {
    #[default]
    Top,
    Bottom,
}

impl Side {
    pub fn flip(self) -> Side {
        match self {
            Side::Top => Side::Bottom,
            Side::Bottom => Side::Top,
        }
    }
}

/// Material properties, SI units.
#[derive(Clone, Debug, PartialEq)]
pub struct Material {
    pub name: String,
    /// Relative permittivity (Dk).
    pub dk: f64,
    /// Loss tangent (Df).
    pub df: f64,
    /// Electrical conductivity, S/m (copper ≈ 5.8e7).
    pub conductivity: f64,
    /// In-plane thermal conductivity, W/(m·K).
    pub k_inplane: f64,
    /// Through-plane thermal conductivity, W/(m·K).
    pub k_through: f64,
    /// Density, kg/m³.
    pub density: f64,
    /// Specific heat capacity, J/(kg·K).
    pub cp: f64,
    /// Coefficient of thermal expansion, ppm/K (in-plane).
    pub cte_ppm: f64,
}

impl Material {
    pub fn copper() -> Self {
        Material {
            name: "copper".into(),
            dk: 1.0,
            df: 0.0,
            conductivity: 5.8e7,
            k_inplane: 385.0,
            k_through: 385.0,
            density: 8960.0,
            cp: 385.0,
            cte_ppm: 17.0,
        }
    }
    pub fn fr4() -> Self {
        Material {
            name: "FR-4".into(),
            dk: 4.3,
            df: 0.02,
            conductivity: 0.0,
            k_inplane: 0.8,
            k_through: 0.3,
            density: 1850.0,
            cp: 1100.0,
            cte_ppm: 14.0,
        }
    }
    pub fn polyimide() -> Self {
        Material {
            name: "polyimide".into(),
            dk: 3.4,
            df: 0.004,
            conductivity: 0.0,
            k_inplane: 0.2,
            k_through: 0.12,
            density: 1420.0,
            cp: 1090.0,
            cte_ppm: 20.0,
        }
    }
    pub fn solder_mask() -> Self {
        Material {
            name: "LPI solder mask".into(),
            dk: 3.8,
            df: 0.025,
            conductivity: 0.0,
            k_inplane: 0.25,
            k_through: 0.25,
            density: 1500.0,
            cp: 1000.0,
            cte_ppm: 60.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum StackKind {
    /// Copper with its weight in oz/ft² (thickness is authoritative; weight is informative).
    Copper {
        weight_oz: f64,
    },
    Dielectric {
        core: bool,
    },
    SolderMask,
    Silkscreen,
    Paste,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StackLayer {
    pub name: String,
    pub kind: StackKind,
    pub thickness: Coord,
    pub material: Material,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Stackup {
    pub name: String,
    /// Top to bottom.
    pub layers: Vec<StackLayer>,
    pub finish: String,
}

/// Copper layer names in KiCad convention: `F.Cu`, `In1.Cu` … `B.Cu`.
pub fn copper_name(i: u8, count: u8) -> String {
    if i == 0 {
        "F.Cu".into()
    } else if i + 1 == count {
        "B.Cu".into()
    } else {
        format!("In{i}.Cu")
    }
}

impl Stackup {
    /// A symmetric FR-4 stack of `n` copper layers (n even, ≥ 2) of `total` thickness, with
    /// `outer_oz` / `inner_oz` copper, alternating core and prepreg, 20 µm mask.
    pub fn standard(n: u8, total: Coord, outer_oz: f64, inner_oz: f64) -> Stackup {
        let oz = |w: f64| (w * 34.79e3) as Coord;
        let mask = mm(0.02);
        let mut copper_total = 0;
        for i in 0..n {
            copper_total += oz(if i == 0 || i + 1 == n {
                outer_oz
            } else {
                inner_oz
            });
        }
        let gaps = n.saturating_sub(1).max(1) as Coord;
        let diel = ((total - copper_total - 2 * mask) / gaps).max(mm(0.05));
        let mut layers = vec![StackLayer {
            name: "F.Mask".into(),
            kind: StackKind::SolderMask,
            thickness: mask,
            material: Material::solder_mask(),
        }];
        for i in 0..n {
            let w = if i == 0 || i + 1 == n {
                outer_oz
            } else {
                inner_oz
            };
            layers.push(StackLayer {
                name: copper_name(i, n),
                kind: StackKind::Copper { weight_oz: w },
                thickness: oz(w),
                material: Material::copper(),
            });
            if i + 1 < n {
                let core = i % 2 == 1 || n == 2;
                let mut m = Material::fr4();
                if !core {
                    m.name = "FR-4 prepreg".into();
                    m.dk = 4.1;
                }
                layers.push(StackLayer {
                    name: format!("dielectric {}", i + 1),
                    kind: StackKind::Dielectric { core },
                    thickness: diel,
                    material: m,
                });
            }
        }
        layers.push(StackLayer {
            name: "B.Mask".into(),
            kind: StackKind::SolderMask,
            thickness: mask,
            material: Material::solder_mask(),
        });
        Stackup {
            name: "rigid".into(),
            layers,
            finish: "ENIG".into(),
        }
    }

    /// A flex stack: `n` copper layers on polyimide with coverlay.
    pub fn flex(n: u8, copper_oz: f64, core: Coord) -> Stackup {
        let mut s = Stackup::standard(n.max(1), core * n as Coord, copper_oz, copper_oz);
        s.name = "flex".into();
        for l in &mut s.layers {
            match l.kind {
                StackKind::Dielectric { .. } => {
                    l.material = Material::polyimide();
                    l.thickness = core;
                }
                StackKind::SolderMask => {
                    l.name = l.name.replace("Mask", "Coverlay");
                    l.material = Material::polyimide();
                    l.thickness = mm(0.025);
                }
                _ => {}
            }
        }
        s
    }

    pub fn copper_count(&self) -> u8 {
        self.layers
            .iter()
            .filter(|l| matches!(l.kind, StackKind::Copper { .. }))
            .count() as u8
    }
    pub fn copper_names(&self) -> Vec<String> {
        self.layers
            .iter()
            .filter(|l| matches!(l.kind, StackKind::Copper { .. }))
            .map(|l| l.name.clone())
            .collect()
    }
    pub fn thickness(&self) -> Coord {
        self.layers.iter().map(|l| l.thickness).sum()
    }
    /// Index into `layers` of copper layer `i`.
    pub fn copper_index(&self, i: u8) -> Option<usize> {
        self.layers
            .iter()
            .enumerate()
            .filter(|(_, l)| matches!(l.kind, StackKind::Copper { .. }))
            .nth(i as usize)
            .map(|(k, _)| k)
    }
    /// Depth from the top surface to the middle of copper layer `i`.
    pub fn copper_z(&self, i: u8) -> Option<Coord> {
        let k = self.copper_index(i)?;
        let above: Coord = self.layers[..k].iter().map(|l| l.thickness).sum();
        Some(above + self.layers[k].thickness / 2)
    }
    /// Vertical distance a via travels between copper layers `a` and `b` (centre to centre).
    pub fn span_length(&self, a: u8, b: u8) -> Coord {
        match (self.copper_z(a), self.copper_z(b)) {
            (Some(x), Some(y)) => (x - y).abs(),
            _ => 0,
        }
    }
    /// Dielectric directly between adjacent copper layers `i` and `i + 1`.
    pub fn dielectric_below(&self, i: u8) -> Option<&StackLayer> {
        let k = self.copper_index(i)?;
        self.layers[k + 1..]
            .iter()
            .take_while(|l| !matches!(l.kind, StackKind::Copper { .. }))
            .find(|l| matches!(l.kind, StackKind::Dielectric { .. }))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Bend {
    pub a: Point,
    pub b: Point,
    pub radius: Coord,
    /// Fold angle in degrees (positive folds the far side toward the top).
    pub angle: f64,
    /// Width of the bend area each side of the line where vias and pads are not allowed.
    pub keepout: Coord,
}

/// A region of the board using one stackup (rigid-flex).
#[derive(Clone, Debug, PartialEq)]
pub struct StackRegion {
    pub name: String,
    pub outline: Polygon,
    pub stackup: String,
    pub bends: Vec<Bend>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ViaKind {
    Through,
    Blind,
    Buried,
    Micro,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Via {
    pub id: ItemId,
    pub net: Option<NetId>,
    pub at: Point,
    pub diameter: Coord,
    pub drill: Coord,
    /// Top-most and bottom-most copper layers (`from < to`).
    pub from: u8,
    pub to: u8,
    pub micro: bool,
    pub tented: bool,
    pub locked: bool,
}

impl Via {
    pub fn kind(&self, copper_count: u8) -> ViaKind {
        if self.micro {
            ViaKind::Micro
        } else if self.from == 0 && self.to + 1 == copper_count {
            ViaKind::Through
        } else if self.from == 0 || self.to + 1 == copper_count {
            ViaKind::Blind
        } else {
            ViaKind::Buried
        }
    }
    pub fn layers(&self) -> impl Iterator<Item = u8> {
        self.from..=self.to
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum TrackGeom {
    Segment {
        a: Point,
        b: Point,
    },
    /// Three-point arc.
    Arc {
        start: Point,
        mid: Point,
        end: Point,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Track {
    pub id: ItemId,
    pub net: Option<NetId>,
    pub layer: u8,
    pub width: Coord,
    pub geom: TrackGeom,
    pub locked: bool,
}

impl Track {
    pub fn endpoints(&self) -> (Point, Point) {
        match self.geom {
            TrackGeom::Segment { a, b } => (a, b),
            TrackGeom::Arc { start, end, .. } => (start, end),
        }
    }
    /// The track as primitives (an arc becomes chained segments within tolerance).
    pub fn prims(&self) -> Vec<Prim> {
        match self.geom {
            TrackGeom::Segment { a, b } => vec![Prim::segment(a, b, self.width)],
            TrackGeom::Arc { start, mid, end } => {
                match crate::geom::shape::arc_from_three(start, mid, end) {
                    Some((c, sw)) => {
                        let pts =
                            crate::geom::shape::arc_points(c, start, sw, crate::geom::ARC_TOL);
                        pts.windows(2)
                            .map(|w| Prim::segment(w[0], w[1], self.width))
                            .collect()
                    }
                    None => vec![Prim::segment(start, end, self.width)],
                }
            }
        }
    }
    /// Centreline length, nm.
    pub fn length(&self) -> f64 {
        match self.geom {
            TrackGeom::Segment { a, b } => a.dist(b),
            TrackGeom::Arc { start, mid, end } => {
                match crate::geom::shape::arc_from_three(start, mid, end) {
                    Some((c, sw)) => c.dist(start) * sw.abs().to_radians(),
                    None => start.dist(end),
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum PadConnect {
    Solid,
    #[default]
    Thermal,
    /// Thermal reliefs for through-hole pads only; SMD pads solid.
    ThermalThtOnly,
    None,
}

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum IslandPolicy {
    #[default]
    Remove,
    Keep,
    /// Remove islands smaller than this area (nm²).
    RemoveBelow(f64),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Zone {
    pub id: ItemId,
    pub name: String,
    pub net: Option<NetId>,
    pub layers: Vec<u8>,
    pub outline: Polygon,
    /// Higher priority fills first; lower-priority zones keep clear of it.
    pub priority: i32,
    /// Overrides the rule clearance when larger.
    pub clearance: Option<Coord>,
    pub min_width: Coord,
    pub connect: PadConnect,
    pub thermal_gap: Coord,
    pub spoke_width: Coord,
    pub islands: IslandPolicy,
    pub locked: bool,
    /// Derived: filled area per copper layer. Not source data.
    pub fill: BTreeMap<u8, crate::geom::poly::PolySet>,
}

impl Zone {
    pub fn new(
        id: ItemId,
        name: &str,
        net: Option<NetId>,
        layers: Vec<u8>,
        outline: Polygon,
    ) -> Self {
        Zone {
            id,
            name: name.into(),
            net,
            layers,
            outline,
            priority: 0,
            clearance: None,
            min_width: mm(0.2),
            connect: PadConnect::Thermal,
            thermal_gap: mm(0.3),
            spoke_width: mm(0.3),
            islands: IslandPolicy::Remove,
            locked: false,
            fill: BTreeMap::new(),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct KeepoutRules {
    pub tracks: bool,
    pub vias: bool,
    pub pads: bool,
    pub pours: bool,
    pub footprints: bool,
    /// Maximum component height inside (mechanical keepouts from the enclosure).
    pub max_height: Option<Coord>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Keepout {
    pub id: ItemId,
    pub name: String,
    pub outline: Polygon,
    /// Copper layers affected (empty = all).
    pub layers: Vec<u8>,
    pub rules: KeepoutRules,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Room {
    pub name: String,
    pub outline: Polygon,
    /// Components (by path or designator) that belong in the room.
    pub members: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BoardText {
    pub id: ItemId,
    pub layer: Layer,
    pub at: Point,
    pub text: String,
    pub size: Coord,
    pub thickness: Coord,
    pub angle: f64,
    pub mirrored: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BoardGraphic {
    pub id: ItemId,
    pub layer: Layer,
    pub graphic: Graphic,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Dimension {
    pub id: ItemId,
    pub layer: Layer,
    pub a: Point,
    pub b: Point,
    pub offset: Coord,
    pub precision: u8,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Footprint {
    pub id: ItemId,
    pub reference: String,
    /// The schematic component this footprint realises (path text), for ECO.
    pub component: Option<String>,
    pub value: String,
    pub package: Package,
    pub at: Point,
    pub rotation: f64,
    pub side: Side,
    pub locked: bool,
    /// Net of each pad by pad name.
    pub pad_nets: BTreeMap<String, NetId>,
    pub fitted: bool,
    pub properties: BTreeMap<String, String>,
}

/// A pad in board coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct PlacedPad {
    pub footprint: ItemId,
    pub index: usize,
    pub name: String,
    pub net: Option<NetId>,
    pub kind: PadKind,
    pub prim: Prim,
    pub center: Point,
    /// Copper layers the pad occupies.
    pub layers: BTreeSet<u8>,
    pub drill: Option<Coord>,
    pub plated: bool,
    pub thermal_pad: bool,
    pub side: Side,
}

/// Pad shape as a primitive, placed by `t`.
pub fn pad_prim(p: &PadDef, t: &Transform) -> Prim {
    let local = Transform::new(p.at, p.rotation, false);
    let t = t.compose(&local);
    match &p.shape {
        PadShape::Circle { d } => Prim::circle(t.apply(Point::ORIGIN), d / 2),
        PadShape::Rect { w, h } => Prim::rect(*w, *h, &t),
        PadShape::RoundRect { w, h, ratio } => Prim::round_rect(
            *w,
            *h,
            ((*w).min(*h) as f64 * ratio.clamp(0.0, 0.5)) as Coord,
            &t,
        ),
        PadShape::Oval { w, h } => Prim::oval(*w, *h, &t),
        PadShape::Trapezoid { w, h, delta } => {
            let (hw, hh, hd) = (w / 2, h / 2, delta / 2);
            Prim::local_polygon(
                &[
                    Point::new(-hw + hd, -hh),
                    Point::new(hw - hd, -hh),
                    Point::new(hw + hd, hh),
                    Point::new(-hw - hd, hh),
                ],
                &t,
            )
        }
        PadShape::Chamfered {
            w,
            h,
            chamfer,
            corners,
        } => {
            let (hw, hh, c) = (w / 2, h / 2, *chamfer);
            let mut pts = Vec::new();
            // Corners counter-clockwise from bottom-left: BL(4), BR(8), TR(2), TL(1).
            let cs = [
                (
                    Point::new(-hw, -hh),
                    4u8,
                    Point::new(0, c),
                    Point::new(c, 0),
                ),
                (Point::new(hw, -hh), 8, Point::new(-c, 0), Point::new(0, c)),
                (Point::new(hw, hh), 2, Point::new(0, -c), Point::new(-c, 0)),
                (Point::new(-hw, hh), 1, Point::new(c, 0), Point::new(0, -c)),
            ];
            for (p0, bit, d1, d2) in cs {
                if corners & bit != 0 {
                    pts.push(p0 + d1);
                    pts.push(p0 + d2);
                } else {
                    pts.push(p0);
                }
            }
            Prim::local_polygon(&pts, &t)
        }
        PadShape::Custom { points, holes } => Prim::local_polygon_with_holes(points, holes, &t),
    }
}

impl Footprint {
    pub fn transform(&self) -> Transform {
        Transform::new(self.at, self.rotation, self.side == Side::Bottom)
    }
    pub fn pads(&self, copper_count: u8) -> Vec<PlacedPad> {
        let t = self.transform();
        (0..self.package.pads.len())
            .map(|i| self.placed(i, copper_count, &t))
            .collect()
    }
    /// Pad `i` in board coordinates (without placing the others).
    pub fn pad(&self, i: usize, copper_count: u8) -> Option<PlacedPad> {
        (i < self.package.pads.len()).then(|| self.placed(i, copper_count, &self.transform()))
    }
    fn placed(&self, i: usize, copper_count: u8, t: &Transform) -> PlacedPad {
        let p = &self.package.pads[i];
        let last = copper_count.saturating_sub(1);
        let layers: BTreeSet<u8> = match (p.copper, self.side) {
            (CopperSpan::All, _) if p.kind != PadKind::Npth => (0..copper_count).collect(),
            (CopperSpan::All, _) => BTreeSet::new(),
            (CopperSpan::Front, Side::Top) | (CopperSpan::Back, Side::Bottom) => [0].into(),
            (CopperSpan::Front, Side::Bottom) | (CopperSpan::Back, Side::Top) => [last].into(),
        };
        PlacedPad {
            footprint: self.id,
            index: i,
            name: p.name.clone(),
            net: self.pad_nets.get(&p.name).copied(),
            kind: p.kind,
            prim: pad_prim(p, t),
            center: t.apply(p.at),
            layers,
            drill: p.drill.as_ref().map(|d| d.diameter),
            plated: p.drill.as_ref().is_some_and(|d| d.plated),
            thermal_pad: p.thermal_pad,
            side: self.side,
        }
    }
    /// Courtyard rings in board coordinates.
    pub fn courtyard(&self) -> Vec<Ring> {
        let t = self.transform();
        self.package
            .courtyard
            .iter()
            .map(|r| {
                let mut v: Ring = r.iter().map(|p| t.apply(*p)).collect();
                if crate::geom::poly::area2(&v) < 0 {
                    v.reverse();
                }
                v
            })
            .collect()
    }
    pub fn bbox(&self) -> Rect {
        Rect::from_points(self.courtyard().into_iter().flatten()).union(
            self.pads(2)
                .iter()
                .fold(Rect::EMPTY, |r, p| r.union(p.prim.bbox())),
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BoardNet {
    pub name: String,
    pub class: Option<String>,
}

/// Board-wide fabrication minimums; they are the defaults under every rule.
#[derive(Clone, Debug, PartialEq)]
pub struct FabLimits {
    pub min_clearance: Coord,
    pub min_track: Coord,
    pub min_via_diameter: Coord,
    pub min_drill: Coord,
    pub min_annular: Coord,
    pub min_hole_to_hole: Coord,
    pub min_copper_to_edge: Coord,
    pub min_silk_to_pad: Coord,
    pub min_microvia_drill: Coord,
}

impl Default for FabLimits {
    fn default() -> Self {
        FabLimits {
            min_clearance: mm(0.15),
            min_track: mm(0.15),
            min_via_diameter: mm(0.45),
            min_drill: mm(0.2),
            min_annular: mm(0.1),
            min_hole_to_hole: mm(0.25),
            min_copper_to_edge: mm(0.3),
            min_silk_to_pad: mm(0.1),
            min_microvia_drill: mm(0.1),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Board {
    pub name: String,
    pub outline: Polygon,
    pub stackup: Stackup,
    /// Additional stackups for rigid-flex regions, by name.
    pub stackups: BTreeMap<String, Stackup>,
    pub regions: Vec<StackRegion>,
    pub nets: Vec<BoardNet>,
    pub footprints: BTreeMap<ItemId, Footprint>,
    pub tracks: BTreeMap<ItemId, Track>,
    pub vias: BTreeMap<ItemId, Via>,
    pub zones: BTreeMap<ItemId, Zone>,
    pub keepouts: BTreeMap<ItemId, Keepout>,
    pub rooms: Vec<Room>,
    pub texts: BTreeMap<ItemId, BoardText>,
    pub graphics: BTreeMap<ItemId, BoardGraphic>,
    pub dimensions: BTreeMap<ItemId, Dimension>,
    pub limits: FabLimits,
    pub next_id: u64,
    /// What the board last took from the schematic: the ECO's base for three-way compares.
    pub sync: crate::eco::BoardSync,
}

impl Board {
    pub fn new(name: &str, outline: Polygon, stackup: Stackup) -> Self {
        Board {
            name: name.into(),
            outline,
            stackup,
            stackups: BTreeMap::new(),
            regions: Vec::new(),
            nets: Vec::new(),
            footprints: BTreeMap::new(),
            tracks: BTreeMap::new(),
            vias: BTreeMap::new(),
            zones: BTreeMap::new(),
            keepouts: BTreeMap::new(),
            rooms: Vec::new(),
            texts: BTreeMap::new(),
            graphics: BTreeMap::new(),
            dimensions: BTreeMap::new(),
            limits: FabLimits::default(),
            next_id: 1,
            sync: Default::default(),
        }
    }
    pub fn alloc(&mut self) -> ItemId {
        let id = ItemId(self.next_id);
        self.next_id += 1;
        id
    }
    pub fn copper_count(&self) -> u8 {
        self.stackup.copper_count()
    }
    pub fn add_net(&mut self, name: &str, class: Option<&str>) -> NetId {
        if let Some(i) = self.nets.iter().position(|n| n.name == name) {
            return NetId(i as u32);
        }
        self.nets.push(BoardNet {
            name: name.into(),
            class: class.map(|c| c.to_string()),
        });
        NetId(self.nets.len() as u32 - 1)
    }
    pub fn net(&self, name: &str) -> Option<NetId> {
        self.nets
            .iter()
            .position(|n| n.name == name)
            .map(|i| NetId(i as u32))
    }
    pub fn net_name(&self, n: NetId) -> &str {
        &self.nets[n.0 as usize].name
    }
    pub fn add_track(
        &mut self,
        net: Option<NetId>,
        layer: u8,
        width: Coord,
        a: Point,
        b: Point,
    ) -> ItemId {
        let id = self.alloc();
        self.tracks.insert(
            id,
            Track {
                id,
                net,
                layer,
                width,
                geom: TrackGeom::Segment { a, b },
                locked: false,
            },
        );
        id
    }
    pub fn add_via(
        &mut self,
        net: Option<NetId>,
        at: Point,
        diameter: Coord,
        drill: Coord,
    ) -> ItemId {
        let id = self.alloc();
        let to = self.copper_count().saturating_sub(1);
        self.vias.insert(
            id,
            Via {
                id,
                net,
                at,
                diameter,
                drill,
                from: 0,
                to,
                micro: false,
                tented: true,
                locked: false,
            },
        );
        id
    }
    pub fn place(
        &mut self,
        reference: &str,
        package: Package,
        at: Point,
        rotation: f64,
        side: Side,
    ) -> ItemId {
        let id = self.alloc();
        self.footprints.insert(
            id,
            Footprint {
                id,
                reference: reference.into(),
                component: None,
                value: String::new(),
                package,
                at,
                rotation,
                side,
                locked: false,
                pad_nets: BTreeMap::new(),
                fitted: true,
                properties: BTreeMap::new(),
            },
        );
        id
    }
    pub fn footprint_by_ref(&self, r: &str) -> Option<&Footprint> {
        self.footprints.values().find(|f| f.reference == r)
    }
    pub fn all_pads(&self) -> Vec<PlacedPad> {
        let n = self.copper_count();
        self.footprints.values().flat_map(|f| f.pads(n)).collect()
    }
    /// The stackup in force at `p` (a region's, else the rigid one).
    pub fn stackup_at(&self, p: Point) -> &Stackup {
        for r in &self.regions {
            if r.outline.contains(p) != Some(false) {
                if let Some(s) = self.stackups.get(&r.stackup) {
                    return s;
                }
            }
        }
        &self.stackup
    }
    /// Copper layer indices (of the rigid stack) present at `p`.
    pub fn copper_at(&self, p: Point) -> BTreeSet<u8> {
        let s = self.stackup_at(p);
        let names = s.copper_names();
        let all = self.stackup.copper_names();
        (0..all.len() as u8)
            .filter(|i| names.contains(&all[*i as usize]))
            .collect()
    }
    /// Assigns pad nets from an elaborated netlist by designator and pad name.
    pub fn assign_nets(&mut self, nl: &crate::elab::Netlist) {
        let mut ids: BTreeMap<usize, NetId> = BTreeMap::new();
        for (i, n) in nl.nets.iter().enumerate() {
            ids.insert(i, self.add_net(&n.name, n.class.as_deref()));
        }
        let by_ref: BTreeMap<String, ItemId> = self
            .footprints
            .values()
            .map(|f| (f.reference.clone(), f.id))
            .collect();
        for c in &nl.components {
            let Some(fid) = by_ref.get(&c.designator) else {
                continue;
            };
            let fp = self.footprints.get_mut(fid).unwrap();
            fp.fitted = c.fitted();
            fp.component = Some(c.path.to_string());
            for p in &c.pins {
                if let Some(n) = p.net {
                    for pad in &p.pads {
                        fp.pad_nets.insert(pad.clone(), ids[&n]);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stackup_numbers() {
        let s = Stackup::standard(8, mm(1.6), 1.0, 0.5);
        assert_eq!(s.copper_count(), 8);
        assert!((s.thickness() - mm(1.6)).abs() < mm(0.01));
        assert_eq!(s.copper_names()[0], "F.Cu");
        assert_eq!(s.copper_names()[7], "B.Cu");
        assert_eq!(s.copper_names()[3], "In3.Cu");
        assert!(s.span_length(0, 7) > mm(1.4));
        assert!(s.dielectric_below(0).is_some());
        let v = Via {
            id: ItemId(1),
            net: None,
            at: Point::ORIGIN,
            diameter: mm(0.3),
            drill: mm(0.15),
            from: 1,
            to: 3,
            micro: false,
            tented: true,
            locked: false,
        };
        assert_eq!(v.kind(8), ViaKind::Buried);
        let f = Stackup::flex(2, 0.5, mm(0.05));
        assert_eq!(f.copper_count(), 2);
    }
}
