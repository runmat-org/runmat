//! The pool library: unit, entity, symbol, package and part are separate items.
//!
//! - A [`Unit`] is the pin list of one logical gate, with electrical pin types and swap groups.
//! - An [`Entity`] is a logical device made of gates, each an instance of a unit (a quad op-amp
//!   is four `opamp` gates plus a `power` gate).
//! - A [`Symbol`] is the schematic graphic for one unit.
//! - A [`Package`] is the physical footprint: pads with padstacks, courtyard, silk, fab
//!   drawing and a 3D model reference.
//! - A [`Part`] is orderable: an entity on a package with a pin→pad map, MPN, manufacturer,
//!   parametrics, lifecycle, supply data, thermal data (θJC, θJB, Pmax) and a SPICE model.
//!   Parts can inherit from a base part.
//!
//! Every item has a UUID and a name; lookups accept either. [`footprint`] generates packages
//! for standard families from IPC-7351 parameters.

pub mod footprint;

use crate::geom::poly::Ring;
use crate::geom::{Coord, Point};
use crate::ids::Uuid;
use crate::ir::Param;
use std::collections::BTreeMap;

/// Electrical pin type, as used by the ERC matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PinType {
    Input,
    Output,
    Bidirectional,
    TriState,
    Passive,
    PowerIn,
    PowerOut,
    OpenCollector,
    OpenEmitter,
    NotConnected,
    Unspecified,
    /// Free pins connect to anything without complaint (mounting pads, test points).
    Free,
}

impl PinType {
    pub const ALL: [PinType; 12] = [
        PinType::Input,
        PinType::Output,
        PinType::Bidirectional,
        PinType::TriState,
        PinType::Passive,
        PinType::PowerIn,
        PinType::PowerOut,
        PinType::OpenCollector,
        PinType::OpenEmitter,
        PinType::NotConnected,
        PinType::Unspecified,
        PinType::Free,
    ];
    pub fn index(self) -> usize {
        PinType::ALL.iter().position(|p| *p == self).unwrap()
    }
    pub fn name(self) -> &'static str {
        match self {
            PinType::Input => "input",
            PinType::Output => "output",
            PinType::Bidirectional => "bidirectional",
            PinType::TriState => "tri_state",
            PinType::Passive => "passive",
            PinType::PowerIn => "power_in",
            PinType::PowerOut => "power_out",
            PinType::OpenCollector => "open_collector",
            PinType::OpenEmitter => "open_emitter",
            PinType::NotConnected => "no_connect",
            PinType::Unspecified => "unspecified",
            PinType::Free => "free",
        }
    }
    pub fn from_name(s: &str) -> Option<PinType> {
        PinType::ALL.iter().copied().find(|p| p.name() == s)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct UnitPin {
    pub name: String,
    pub kind: PinType,
    /// Pins with the same non-zero swap group can be swapped during layout.
    pub swap_group: u32,
    pub alternates: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Unit {
    pub uuid: Uuid,
    pub name: String,
    pub pins: Vec<UnitPin>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Gate {
    pub name: String,
    pub unit: Uuid,
    /// Gates with the same non-zero swap group are interchangeable.
    pub swap_group: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Entity {
    pub uuid: Uuid,
    pub name: String,
    /// Designator prefix (`R`, `C`, `U`, `J`).
    pub prefix: String,
    pub gates: Vec<Gate>,
    pub tags: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Graphic {
    Line {
        points: Vec<Point>,
        width: Coord,
    },
    Rect {
        a: Point,
        b: Point,
        width: Coord,
        fill: bool,
    },
    Circle {
        center: Point,
        radius: Coord,
        width: Coord,
        fill: bool,
    },
    Arc {
        start: Point,
        mid: Point,
        end: Point,
        width: Coord,
    },
    Polygon {
        points: Vec<Point>,
        width: Coord,
        fill: bool,
    },
    Text {
        at: Point,
        text: String,
        size: Coord,
        angle: f64,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct SymbolPin {
    pub pin: String,
    pub at: Point,
    pub angle: f64,
    pub length: Coord,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Symbol {
    pub uuid: Uuid,
    pub unit: Uuid,
    pub name: String,
    pub pins: Vec<SymbolPin>,
    pub graphics: Vec<Graphic>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum PadShape {
    Circle {
        d: Coord,
    },
    Rect {
        w: Coord,
        h: Coord,
    },
    /// `ratio` of the smaller side used as corner radius (KiCad convention, 0..=0.5).
    RoundRect {
        w: Coord,
        h: Coord,
        ratio: f64,
    },
    Oval {
        w: Coord,
        h: Coord,
    },
    /// Trapezoid: `delta` widens the +y edge and narrows the −y edge in x.
    Trapezoid {
        w: Coord,
        h: Coord,
        delta: Coord,
    },
    /// Chamfered rectangle; `corners` bit mask: 1 top-left, 2 top-right, 4 bottom-left, 8 bottom-right.
    Chamfered {
        w: Coord,
        h: Coord,
        chamfer: Coord,
        corners: u8,
    },
    /// Arbitrary polygon in pad-local coordinates.
    Custom {
        points: Vec<Point>,
    },
}

impl PadShape {
    pub fn size(&self) -> (Coord, Coord) {
        match self {
            PadShape::Circle { d } => (*d, *d),
            PadShape::Rect { w, h }
            | PadShape::RoundRect { w, h, .. }
            | PadShape::Oval { w, h }
            | PadShape::Trapezoid { w, h, .. }
            | PadShape::Chamfered { w, h, .. } => (*w, *h),
            PadShape::Custom { points } => {
                let r = crate::geom::Rect::from_points(points.iter().copied());
                (r.width(), r.height())
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PadKind {
    Smd,
    /// Plated through hole.
    Tht,
    /// Non-plated hole (mechanical).
    Npth,
    /// Edge connector / fingers (SMD without paste).
    Connector,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Drill {
    pub diameter: Coord,
    /// Slot length along the pad's x axis, when slotted.
    pub slot: Option<Coord>,
    pub plated: bool,
}

/// Which copper a pad occupies, relative to the footprint's side.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CopperSpan {
    /// The footprint's own side only (SMD).
    Front,
    /// The opposite side only.
    Back,
    /// Every copper layer (through hole).
    All,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PadDef {
    pub name: String,
    pub kind: PadKind,
    pub shape: PadShape,
    pub at: Point,
    pub rotation: f64,
    pub drill: Option<Drill>,
    pub copper: CopperSpan,
    pub mask: bool,
    pub paste: bool,
    pub mask_margin: Option<Coord>,
    pub paste_margin: Option<Coord>,
    /// Electrical role for thermal connection: thermal pads under ICs get solid connections.
    pub thermal_pad: bool,
}

impl PadDef {
    pub fn smd(name: &str, at: Point, shape: PadShape) -> Self {
        PadDef {
            name: name.into(),
            kind: PadKind::Smd,
            shape,
            at,
            rotation: 0.0,
            drill: None,
            copper: CopperSpan::Front,
            mask: true,
            paste: true,
            mask_margin: None,
            paste_margin: None,
            thermal_pad: false,
        }
    }
    pub fn tht(name: &str, at: Point, shape: PadShape, drill: Coord) -> Self {
        PadDef {
            name: name.into(),
            kind: PadKind::Tht,
            shape,
            at,
            rotation: 0.0,
            drill: Some(Drill {
                diameter: drill,
                slot: None,
                plated: true,
            }),
            copper: CopperSpan::All,
            mask: true,
            paste: false,
            mask_margin: None,
            paste_margin: None,
            thermal_pad: false,
        }
    }
    pub fn npth(name: &str, at: Point, drill: Coord) -> Self {
        PadDef {
            name: name.into(),
            kind: PadKind::Npth,
            shape: PadShape::Circle { d: drill },
            at,
            rotation: 0.0,
            drill: Some(Drill {
                diameter: drill,
                slot: None,
                plated: false,
            }),
            copper: CopperSpan::All,
            mask: true,
            paste: false,
            mask_margin: None,
            paste_margin: None,
            thermal_pad: false,
        }
    }
}

/// How a 3D model is placed relative to the footprint (circuit-json `cad_component`
/// conventions: origin, board normal, unit scale).
#[derive(Clone, Debug, PartialEq)]
pub struct Model3d {
    pub path: String,
    /// Millimetres.
    pub offset: (f64, f64, f64),
    /// Degrees about x, y, z.
    pub rotation: (f64, f64, f64),
    pub scale: (f64, f64, f64),
    /// `"footprint"` (model origin at the footprint origin) or `"center"` (package centre).
    pub origin: String,
}

/// Package body for 3D placeholders and height keepouts (mm).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Body {
    pub width: Coord,
    pub length: Coord,
    pub height: Coord,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Package {
    pub uuid: Uuid,
    pub name: String,
    pub description: String,
    pub pads: Vec<PadDef>,
    pub courtyard: Vec<Ring>,
    pub silk: Vec<Graphic>,
    pub fab: Vec<Graphic>,
    pub model: Option<Model3d>,
    pub body: Option<Body>,
    /// Through-hole packages need wave or selective soldering; SMD ones go in pick-and-place.
    pub smd: bool,
}

impl Package {
    pub fn pad(&self, name: &str) -> Option<&PadDef> {
        self.pads.iter().find(|p| p.name == name)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum Lifecycle {
    Preview,
    Active,
    /// Not recommended for new designs.
    Nrnd,
    LastTimeBuy,
    Obsolete,
    #[default]
    Unknown,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SupplyRecord {
    pub supplier: String,
    pub sku: String,
    pub stock: Option<u64>,
    /// `(quantity, unit price)` breaks, ascending by quantity.
    pub price_breaks: Vec<(u64, f64)>,
    pub currency: String,
}

/// Compact thermal model fields used by the board → FEA power map.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ThermalData {
    /// Junction to case, K/W.
    pub theta_jc: Option<f64>,
    /// Junction to board, K/W.
    pub theta_jb: Option<f64>,
    /// Junction to ambient (JEDEC board), K/W; informative only.
    pub theta_ja: Option<f64>,
    /// Maximum dissipation, W.
    pub p_max: Option<f64>,
    /// Maximum junction temperature, °C.
    pub tj_max: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SpiceModel {
    /// Model or subcircuit name.
    pub model: String,
    /// Library file or inline text reference.
    pub library: Option<String>,
    /// Pin order of the subcircuit, as `(gate, pin)`.
    pub pin_order: Vec<(String, String)>,
    /// Primitive device letter when not a subcircuit (`R`, `C`, `L`, `D`, `Q`, `M`).
    pub device: Option<char>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PadMapEntry {
    pub gate: String,
    pub pin: String,
    pub pads: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Part {
    pub uuid: Uuid,
    pub mpn: String,
    pub manufacturer: String,
    pub description: String,
    pub entity: Uuid,
    pub package: Uuid,
    pub pad_map: Vec<PadMapEntry>,
    pub params: BTreeMap<String, Param>,
    pub lifecycle: Lifecycle,
    pub datasheet: Option<String>,
    pub supply: Vec<SupplyRecord>,
    pub thermal: ThermalData,
    pub spice: Option<SpiceModel>,
    /// Inherit unset fields (params, thermal, spice, supply) from this part.
    pub base: Option<Uuid>,
}

/// One resolved pin of a part: gate, pin, electrical type, pads.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedPin {
    pub gate: String,
    pub pin: String,
    pub kind: PinType,
    pub pads: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct LibraryIssue {
    pub item: String,
    pub message: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Pool {
    pub units: BTreeMap<Uuid, Unit>,
    pub entities: BTreeMap<Uuid, Entity>,
    pub symbols: BTreeMap<Uuid, Symbol>,
    pub packages: BTreeMap<Uuid, Package>,
    pub parts: BTreeMap<Uuid, Part>,
}

fn find<'a, T>(map: &'a BTreeMap<Uuid, T>, key: &str, name: impl Fn(&T) -> &str) -> Option<&'a T> {
    if let Some(u) = Uuid::parse(key) {
        if let Some(t) = map.get(&u) {
            return Some(t);
        }
    }
    map.values().find(|t| name(t) == key)
}

impl Pool {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn add_unit(&mut self, u: Unit) -> Uuid {
        let id = u.uuid;
        self.units.insert(id, u);
        id
    }
    pub fn add_entity(&mut self, e: Entity) -> Uuid {
        let id = e.uuid;
        self.entities.insert(id, e);
        id
    }
    pub fn add_symbol(&mut self, s: Symbol) -> Uuid {
        let id = s.uuid;
        self.symbols.insert(id, s);
        id
    }
    pub fn add_package(&mut self, p: Package) -> Uuid {
        let id = p.uuid;
        self.packages.insert(id, p);
        id
    }
    pub fn add_part(&mut self, p: Part) -> Uuid {
        let id = p.uuid;
        self.parts.insert(id, p);
        id
    }
    pub fn unit(&self, key: &str) -> Option<&Unit> {
        find(&self.units, key, |u| &u.name)
    }
    pub fn entity(&self, key: &str) -> Option<&Entity> {
        find(&self.entities, key, |e| &e.name)
    }
    pub fn package(&self, key: &str) -> Option<&Package> {
        find(&self.packages, key, |p| &p.name)
    }
    pub fn part(&self, key: &str) -> Option<&Part> {
        find(&self.parts, key, |p| &p.mpn)
    }

    /// The part with inherited fields filled from its base chain.
    pub fn effective_part(&self, part: &Part) -> Part {
        let mut p = part.clone();
        let mut seen = vec![p.uuid];
        let mut base = p.base;
        while let Some(b) = base {
            if seen.contains(&b) {
                break;
            }
            seen.push(b);
            let Some(bp) = self.parts.get(&b) else { break };
            for (k, v) in &bp.params {
                p.params.entry(k.clone()).or_insert_with(|| v.clone());
            }
            let t = &mut p.thermal;
            t.theta_jc = t.theta_jc.or(bp.thermal.theta_jc);
            t.theta_jb = t.theta_jb.or(bp.thermal.theta_jb);
            t.theta_ja = t.theta_ja.or(bp.thermal.theta_ja);
            t.p_max = t.p_max.or(bp.thermal.p_max);
            t.tj_max = t.tj_max.or(bp.thermal.tj_max);
            if p.spice.is_none() {
                p.spice = bp.spice.clone();
            }
            if p.supply.is_empty() {
                p.supply = bp.supply.clone();
            }
            if p.datasheet.is_none() {
                p.datasheet = bp.datasheet.clone();
            }
            if p.pad_map.is_empty() {
                p.pad_map = bp.pad_map.clone();
            }
            base = bp.base;
        }
        p
    }

    /// Every pin of an entity with its type, and its pads under `part` (if any).
    pub fn pins(&self, entity: &Entity, part: Option<&Part>) -> Vec<ResolvedPin> {
        let mut out = Vec::new();
        for g in &entity.gates {
            let Some(u) = self.units.get(&g.unit) else {
                continue;
            };
            for pin in &u.pins {
                let pads = part
                    .and_then(|p| {
                        p.pad_map
                            .iter()
                            .find(|m| m.gate == g.name && m.pin == pin.name)
                            .map(|m| m.pads.clone())
                    })
                    .unwrap_or_default();
                out.push(ResolvedPin {
                    gate: g.name.clone(),
                    pin: pin.name.clone(),
                    kind: pin.kind,
                    pads,
                });
            }
        }
        out
    }

    /// Structural checks: dangling references, unmapped pins, pads mapped twice, unknown pads.
    pub fn validate(&self) -> Vec<LibraryIssue> {
        let mut out = Vec::new();
        for e in self.entities.values() {
            for g in &e.gates {
                if !self.units.contains_key(&g.unit) {
                    out.push(LibraryIssue {
                        item: e.name.clone(),
                        message: format!("gate `{}` refers to a missing unit", g.name),
                    });
                }
            }
        }
        for s in self.symbols.values() {
            match self.units.get(&s.unit) {
                None => out.push(LibraryIssue {
                    item: s.name.clone(),
                    message: "symbol refers to a missing unit".into(),
                }),
                Some(u) => {
                    for p in &u.pins {
                        if !s.pins.iter().any(|sp| sp.pin == p.name) {
                            out.push(LibraryIssue {
                                item: s.name.clone(),
                                message: format!("unit pin `{}` has no symbol pin", p.name),
                            });
                        }
                    }
                }
            }
        }
        for part in self.parts.values() {
            let part = self.effective_part(part);
            let (Some(e), Some(pkg)) = (
                self.entities.get(&part.entity),
                self.packages.get(&part.package),
            ) else {
                out.push(LibraryIssue {
                    item: part.mpn.clone(),
                    message: "part refers to a missing entity or package".into(),
                });
                continue;
            };
            let mut used: BTreeMap<&str, &str> = BTreeMap::new();
            for rp in self.pins(e, Some(&part)) {
                if rp.pads.is_empty() && rp.kind != PinType::NotConnected {
                    out.push(LibraryIssue {
                        item: part.mpn.clone(),
                        message: format!("pin `{}.{}` has no pad", rp.gate, rp.pin),
                    });
                }
            }
            for m in &part.pad_map {
                for pad in &m.pads {
                    if pkg.pad(pad).is_none() {
                        out.push(LibraryIssue {
                            item: part.mpn.clone(),
                            message: format!("pad `{pad}` is not in package `{}`", pkg.name),
                        });
                    }
                    if let Some(prev) = used.insert(pad, &m.pin) {
                        out.push(LibraryIssue {
                            item: part.mpn.clone(),
                            message: format!(
                                "pad `{pad}` is mapped to both `{prev}` and `{}`",
                                m.pin
                            ),
                        });
                    }
                }
            }
        }
        out.sort();
        out
    }
}

/// Builders for common logical items, so fixtures and importers do not repeat themselves.
pub mod make {
    use super::*;
    use crate::ids::{fnv128, Uuid};

    pub fn uuid(kind: &str, name: &str) -> Uuid {
        Uuid(fnv128(format!("pool|{kind}|{name}").as_bytes())).with_version(8)
    }

    /// A single-gate entity whose unit has the given pins.
    pub fn simple_entity(
        pool: &mut Pool,
        name: &str,
        prefix: &str,
        pins: &[(&str, PinType)],
    ) -> Uuid {
        let unit = Unit {
            uuid: uuid("unit", name),
            name: name.into(),
            pins: pins
                .iter()
                .map(|(n, k)| UnitPin {
                    name: n.to_string(),
                    kind: *k,
                    swap_group: 0,
                    alternates: vec![],
                })
                .collect(),
        };
        let u = pool.add_unit(unit);
        pool.add_entity(Entity {
            uuid: uuid("entity", name),
            name: name.into(),
            prefix: prefix.into(),
            gates: vec![Gate {
                name: "A".into(),
                unit: u,
                swap_group: 0,
            }],
            tags: vec![],
        })
    }

    /// A part mapping pins to pads by an explicit list `(pin, [pads])`.
    pub fn part(
        pool: &mut Pool,
        mpn: &str,
        manufacturer: &str,
        entity: Uuid,
        package: Uuid,
        map: &[(&str, &[&str])],
    ) -> Uuid {
        let gate = pool.entities[&entity].gates[0].name.clone();
        pool.add_part(Part {
            uuid: uuid("part", mpn),
            mpn: mpn.into(),
            manufacturer: manufacturer.into(),
            description: String::new(),
            entity,
            package,
            pad_map: map
                .iter()
                .map(|(pin, pads)| PadMapEntry {
                    gate: gate.clone(),
                    pin: pin.to_string(),
                    pads: pads.iter().map(|p| p.to_string()).collect(),
                })
                .collect(),
            params: BTreeMap::new(),
            lifecycle: Lifecycle::Active,
            datasheet: None,
            supply: vec![],
            thermal: ThermalData::default(),
            spice: None,
            base: None,
        })
    }

    /// A part whose pins map one-to-one to pads of the same name.
    pub fn part_by_name(
        pool: &mut Pool,
        mpn: &str,
        manufacturer: &str,
        entity: Uuid,
        package: Uuid,
    ) -> Uuid {
        let e = pool.entities[&entity].clone();
        let mut map = Vec::new();
        for g in &e.gates {
            for p in &pool.units[&g.unit].pins {
                map.push(PadMapEntry {
                    gate: g.name.clone(),
                    pin: p.name.clone(),
                    pads: vec![p.name.clone()],
                });
            }
        }
        pool.add_part(Part {
            uuid: uuid("part", mpn),
            mpn: mpn.into(),
            manufacturer: manufacturer.into(),
            description: String::new(),
            entity,
            package,
            pad_map: map,
            params: BTreeMap::new(),
            lifecycle: Lifecycle::Active,
            datasheet: None,
            supply: vec![],
            thermal: ThermalData::default(),
            spice: None,
            base: None,
        })
    }
}
