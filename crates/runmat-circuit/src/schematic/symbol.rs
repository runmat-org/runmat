//! Symbol geometry: one unit's body, pins and texts in library coordinates (y up, nm), and the
//! placed transform (mirror, quarter turns, origin) that puts it on a page.
//!
//! - **Boxes** (ICs, connectors, split units): positive supply pins on the top edge, ground
//!   pins on the bottom edge, inputs on the left and outputs on the right, so VCC ports point up
//!   and GND ports point down. The top-left and bottom-left corners stay clear for the
//!   designator and the value.
//! - **Two-pin parts** (R, C, L, D) are drawn with their standard art, 7.62 mm tip to tip, pin
//!   names and numbers hidden, as KiCad and Altium draw them. They turn a quarter for shunts
//!   and decoupling.
//! - **Mounting holes and other one-pin parts:** a circle with its pin below.

use super::units::Unit;
use super::{ceil_div, number_text, text_width, up_to_grid, GRID, NAME_OFFSET, NO_CONNECT_HALF};
use super::{PIN_LEN, PIN_PITCH, PIN_TEXT};
use crate::geom::{Coord, Point};
use crate::library::PinType;

/// The side of the body a pin leaves from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PinSide {
    Left,
    Right,
    Top,
    Bottom,
}

impl PinSide {
    pub fn name(self) -> &'static str {
        match self {
            PinSide::Left => "left",
            PinSide::Right => "right",
            PinSide::Top => "top",
            PinSide::Bottom => "bottom",
        }
    }
    /// The unit step away from the body.
    pub fn out(self) -> (Coord, Coord) {
        match self {
            PinSide::Left => (-1, 0),
            PinSide::Right => (1, 0),
            PinSide::Top => (0, 1),
            PinSide::Bottom => (0, -1),
        }
    }
    pub fn from_out(d: (Coord, Coord)) -> PinSide {
        match d {
            (x, _) if x < 0 => PinSide::Left,
            (x, _) if x > 0 => PinSide::Right,
            (_, y) if y > 0 => PinSide::Top,
            _ => PinSide::Bottom,
        }
    }
    pub fn horizontal(self) -> bool {
        matches!(self, PinSide::Left | PinSide::Right)
    }
}

/// How a symbol's body is drawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Art {
    Box,
    Resistor,
    Capacitor,
    Inductor,
    Diode,
    Hole,
}

impl Art {
    pub fn name(self) -> &'static str {
        match self {
            Art::Box => "box",
            Art::Resistor => "resistor",
            Art::Capacitor => "capacitor",
            Art::Inductor => "inductor",
            Art::Diode => "diode",
            Art::Hole => "hole",
        }
    }
    /// Small parts hide pin names and numbers.
    pub fn small(self) -> bool {
        self != Art::Box
    }
}

/// One pin of a unit layout, in library coordinates.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LayoutPin {
    pub(crate) name: String,
    pub(crate) pads: Vec<String>,
    pub(crate) kind: PinType,
    pub(crate) side: PinSide,
    /// y of the pin row for a left or right pin, x of the pin column for a top or bottom pin.
    pub(crate) at: Coord,
}

/// One generated unit: body half-size, pins on four sides and the art.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct UnitLayout {
    pub(crate) pins: Vec<LayoutPin>,
    pub(crate) half_w: Coord,
    pub(crate) half_h: Coord,
    /// Every pin's length: long enough for the widest number beside its line.
    pub(crate) pin_len: Coord,
    pub(crate) art: Art,
}

/// The least gap between the left and right names across the body.
const NAME_GAP: Coord = 1_270_000;
/// Width kept clear at the body's top-left and bottom-left for the designator and value.
const CORNER: Coord = 8 * GRID;
/// Most pins on the top or bottom edge.
const EDGE_PINS: usize = 8;

/// Whether a pin name is a ground or negative supply (`GND`, `AGND`, `VSS`, `V-`, `VEE`).
pub fn is_ground_pin(name: &str) -> bool {
    let u = name.to_ascii_uppercase();
    u.contains("GND") || u.starts_with("VSS") || u.starts_with("VEE") || u == "V-"
}

/// A pin list: (name, pads, type).
pub(crate) type PinList = [(String, Vec<String>, PinType)];

impl UnitLayout {
    /// The layout a component's unit is drawn with.
    pub(crate) fn of(c: &crate::elab::CompInst, unit: &Unit, units: usize) -> UnitLayout {
        let pins = super::units::unit_pins(c, unit);
        let prefix: String = c
            .designator
            .chars()
            .take_while(|ch| ch.is_ascii_alphabetic())
            .collect::<String>()
            .to_ascii_uppercase();
        // The corners keep room for the designator (with its unit) and the value.
        let unit_text = match &unit.label {
            Some(l) => format!("{prefix}99{} · {l}", unit.name),
            None => format!("{prefix}99{}", if units > 1 { unit.name.as_str() } else { "" }),
        };
        let value = crate::index::component_value(c).unwrap_or_default();
        let corners = (
            text_width(&unit_text, super::DESIGNATOR_TEXT) + 2 * GRID,
            text_width(&value, super::VALUE_TEXT) + 2 * GRID,
        );
        UnitLayout::new(&pins, art_of(&prefix, &pins, units), corners)
    }

    /// `corners`: the widths kept clear at the top-left and bottom-left of a box.
    pub(crate) fn new(pins: &PinList, art: Art, corners: (Coord, Coord)) -> UnitLayout {
        match art {
            Art::Box => UnitLayout::boxed(pins, corners),
            Art::Hole => UnitLayout {
                pins: pins
                    .iter()
                    .map(|p| LayoutPin {
                        name: p.0.clone(),
                        pads: p.1.clone(),
                        kind: p.2,
                        side: PinSide::Bottom,
                        at: 0,
                    })
                    .collect(),
                half_w: GRID,
                half_h: GRID,
                pin_len: PIN_LEN,
                art,
            },
            _ => {
                // Two pins, 7.62 mm tip to tip; a diode's anode on the left.
                let mut order: Vec<usize> = vec![0, 1];
                if art == Art::Diode {
                    let k = |n: &str| matches!(n.to_ascii_uppercase().as_str(), "K" | "CATHODE");
                    if k(&pins[0].0) {
                        order = vec![1, 0];
                    }
                }
                let half_h = match art {
                    Art::Capacitor => 1_016_000,
                    Art::Diode => 889_000,
                    _ => 635_000,
                };
                UnitLayout {
                    pins: order
                        .iter()
                        .zip([PinSide::Left, PinSide::Right])
                        .map(|(&i, side)| LayoutPin {
                            name: pins[i].0.clone(),
                            pads: pins[i].1.clone(),
                            kind: pins[i].2,
                            side,
                            at: 0,
                        })
                        .collect(),
                    half_w: GRID,
                    half_h,
                    pin_len: PIN_LEN,
                    art,
                }
            }
        }
    }

    fn boxed(pins: &PinList, corners: (Coord, Coord)) -> UnitLayout {
        let ground = |i: usize| {
            is_ground_pin(&pins[i].0) && matches!(pins[i].2, PinType::PowerIn | PinType::Passive)
        };
        let supply = |i: usize| pins[i].2 == PinType::PowerIn && !is_ground_pin(&pins[i].0);
        let mut top: Vec<usize> = (0..pins.len()).filter(|&i| supply(i)).collect();
        let mut bottom: Vec<usize> = (0..pins.len()).filter(|&i| ground(i)).collect();
        let overflow: Vec<usize> = top
            .split_off(top.len().min(EDGE_PINS))
            .into_iter()
            .chain(bottom.split_off(bottom.len().min(EDGE_PINS)))
            .collect();
        let edge: Vec<usize> = top.iter().chain(&bottom).copied().collect();
        let rest: Vec<usize> = (0..pins.len())
            .filter(|i| !edge.contains(i) || overflow.contains(i))
            .collect();
        let passive: Vec<usize> = rest
            .iter()
            .copied()
            .filter(|&i| pins[i].2 == PinType::Passive)
            .collect();
        let left_of = |i: usize| match pins[i].2 {
            PinType::Input | PinType::PowerIn => true,
            PinType::Passive => {
                passive.iter().position(|&p| p == i).unwrap() < passive.len().div_ceil(2)
            }
            _ => false,
        };
        let (mut left, mut right): (Vec<usize>, Vec<usize>) = (Vec::new(), Vec::new());
        for &i in &rest {
            if left_of(i) {
                left.push(i)
            } else {
                right.push(i)
            }
        }
        if right.is_empty() && left.len() > 1 {
            right = left.split_off(left.len().div_ceil(2));
        }
        // A large part keeps both sides in use: the longer side gives half its excess to the
        // other, from its end, so the box is half as tall.
        if right.len() > left.len() + 8 {
            let n = (right.len() - left.len()) / 2;
            left.extend(right.split_off(right.len() - n));
        } else if left.len() > right.len() + 8 {
            let n = (left.len() - right.len()) / 2;
            right.extend(left.split_off(left.len() - n));
        }
        let rows = left.len().max(right.len()).max(1) as Coord;
        // Room above and below the rows for the top and bottom pins' names.
        let edge_room = if top.is_empty() && bottom.is_empty() {
            0
        } else {
            2 * GRID
        };
        let half_h = (rows * PIN_PITCH) / 2 + GRID + edge_room;
        // The body holds the longest name on each side, inset by the name offset, with a gap
        // between them; the top and bottom pins sit right of the corner kept for the texts,
        // each column as wide as its name. Nothing can overlap.
        let widest = |list: &[usize]| {
            list.iter()
                .map(|&i| text_width(&pins[i].0, PIN_TEXT))
                .max()
                .unwrap_or(0)
        };
        let names = widest(&left) + widest(&right) + 2 * NAME_OFFSET + NAME_GAP;
        let column = up_to_pitch(
            top.iter()
                .chain(&bottom)
                .map(|&i| text_width(&pins[i].0, PIN_TEXT) + NAME_GAP)
                .max()
                .unwrap_or(0),
        );
        let edge_span = |n: usize, corner: Coord| {
            if n == 0 {
                0
            } else {
                corner.max(CORNER) + n as Coord * column
            }
        };
        let width = names
            .max(edge_span(top.len(), corners.0))
            .max(edge_span(bottom.len(), corners.1));
        let half_w = up_to_pitch(ceil_div(width, 2)).max(2 * PIN_PITCH);
        let number = pins
            .iter()
            .map(|p| text_width(&number_text(&p.1), PIN_TEXT))
            .max()
            .unwrap_or(0);
        // The number sits centred on the pin, clear of the body and of a no-connect cross at the tip.
        let pin_len = up_to_grid(number + 2 * NO_CONNECT_HALF + NAME_OFFSET).max(PIN_LEN);
        let mut out = Vec::new();
        for (side, list) in [(PinSide::Left, &left), (PinSide::Right, &right)] {
            let first = (list.len() as Coord - 1) * PIN_PITCH / 2;
            for (k, &i) in list.iter().enumerate() {
                // Rows on the 2.54 mm grid from the top, centred.
                let y = (first - k as Coord * PIN_PITCH).div_euclid(GRID) * GRID;
                out.push(pin(pins, i, side, y));
            }
        }
        for (side, list) in [(PinSide::Top, &top), (PinSide::Bottom, &bottom)] {
            // Columns from the right edge, a column apart, the first half a column in.
            for (k, &i) in list.iter().enumerate() {
                let x = half_w - column / 2 - k as Coord * column;
                out.push(pin(pins, i, side, x.div_euclid(GRID) * GRID));
            }
        }
        UnitLayout {
            pins: out,
            half_w,
            half_h,
            pin_len,
            art: Art::Box,
        }
    }

    /// Pin tip in library coordinates.
    pub(crate) fn tip(&self, side: PinSide, at: Coord) -> Point {
        let (dx, dy) = side.out();
        if side.horizontal() {
            Point::new(dx * (self.half_w + self.pin_len), at)
        } else {
            Point::new(at, dy * (self.half_h + self.pin_len))
        }
    }
    /// Where the pin meets the body.
    pub(crate) fn root(&self, side: PinSide, at: Coord) -> Point {
        let (dx, dy) = side.out();
        if side.horizontal() {
            Point::new(dx * self.half_w, at)
        } else {
            Point::new(at, dy * self.half_h)
        }
    }
    /// Where a pin's name sits (inside the body) and its anchor.
    pub(crate) fn name_at(&self, side: PinSide, at: Coord) -> (Point, &'static str) {
        match side {
            PinSide::Left => (
                Point::new(-self.half_w + NAME_OFFSET, centred_baseline(at)),
                "start",
            ),
            PinSide::Right => (
                Point::new(self.half_w - NAME_OFFSET, centred_baseline(at)),
                "end",
            ),
            PinSide::Top => (
                Point::new(at, self.half_h - NAME_OFFSET - PIN_TEXT * 4 / 5),
                "middle",
            ),
            PinSide::Bottom => (
                Point::new(at, -self.half_h + NAME_OFFSET + PIN_TEXT / 5),
                "middle",
            ),
        }
    }
    /// Where a pin's number sits (baseline centre): above a horizontal pin's line, right of a
    /// vertical one's, outside the body.
    pub(crate) fn number_at(&self, side: PinSide, at: Coord, number: &str) -> Point {
        let mid = self.half_w + self.pin_len / 2;
        match side {
            PinSide::Left => Point::new(-mid, at + NUMBER_LIFT + PIN_TEXT / 5),
            PinSide::Right => Point::new(mid, at + NUMBER_LIFT + PIN_TEXT / 5),
            PinSide::Top | PinSide::Bottom => {
                let (_, dy) = side.out();
                let y = dy * (self.half_h + self.pin_len / 2);
                Point::new(
                    at + NUMBER_LIFT + text_width(number, PIN_TEXT) / 2,
                    centred_baseline(y),
                )
            }
        }
    }
    /// The art's strokes (open polylines) in library coordinates; a box or resistor draws its
    /// body outline instead.
    pub(crate) fn strokes(&self) -> Vec<Vec<Point>> {
        let p = Point::new;
        let w = self.half_w;
        match self.art {
            Art::Box | Art::Resistor => Vec::new(),
            Art::Capacitor => {
                let (g, h) = (381_000, self.half_h);
                vec![
                    vec![p(-w, 0), p(-g, 0)],
                    vec![p(-g, -h), p(-g, h)],
                    vec![p(g, -h), p(g, h)],
                    vec![p(g, 0), p(w, 0)],
                ]
            }
            Art::Inductor => {
                // Four half-turns along the axis, six segments each.
                let r = w / 4;
                let mut arc = Vec::new();
                for k in 0..4 {
                    let cx = -w + r + 2 * r * k as Coord;
                    for s in 0..=6 {
                        let a = std::f64::consts::PI * (1.0 - s as f64 / 6.0);
                        arc.push(p(
                            cx + (r as f64 * a.cos()).round() as Coord,
                            (r as f64 * a.sin()).round() as Coord,
                        ));
                    }
                }
                vec![arc]
            }
            Art::Diode => {
                let (t, h) = (762_000, self.half_h);
                vec![
                    vec![p(-w, 0), p(-t, 0)],
                    vec![p(-t, -h), p(-t, h), p(t, 0), p(-t, -h)],
                    vec![p(t, -h), p(t, h)],
                    vec![p(t, 0), p(w, 0)],
                ]
            }
            Art::Hole => {
                let r = self.half_w as f64;
                vec![(0..=16)
                    .map(|k| {
                        let a = std::f64::consts::TAU * k as f64 / 16.0;
                        p((r * a.cos()).round() as Coord, (r * a.sin()).round() as Coord)
                    })
                    .collect()]
            }
        }
    }
    /// The extent of body and pins, `[x0, y0, x1, y1]` in library coordinates.
    pub(crate) fn extent(&self) -> [Coord; 4] {
        let mut e = [-self.half_w, -self.half_h, self.half_w, self.half_h];
        for pin in &self.pins {
            let t = self.tip(pin.side, pin.at);
            e = [e[0].min(t.x), e[1].min(t.y), e[2].max(t.x), e[3].max(t.y)];
        }
        e
    }
}

/// The gap between a pin line and the number above it.
const NUMBER_LIFT: Coord = 254_000;

/// The baseline that centres capitals of height [`PIN_TEXT`] on `y`.
pub(crate) fn centred_baseline(y: Coord) -> Coord {
    y - PIN_TEXT * 3 / 10
}

fn up_to_pitch(v: Coord) -> Coord {
    ceil_div(v, PIN_PITCH) * PIN_PITCH
}

fn pin(pins: &PinList, i: usize, side: PinSide, at: Coord) -> LayoutPin {
    LayoutPin {
        name: pins[i].0.clone(),
        pads: pins[i].1.clone(),
        kind: pins[i].2,
        side,
        at,
    }
}

/// The art a component's unit is drawn with: by designator prefix for a two-pin passive or
/// diode, a circle for a one-pin part, else a box.
pub(crate) fn art_of(prefix: &str, pins: &PinList, units: usize) -> Art {
    if units != 1 {
        return Art::Box;
    }
    let passive = pins
        .iter()
        .all(|p| matches!(p.2, PinType::Passive | PinType::Unspecified | PinType::Free));
    match (pins.len(), prefix) {
        (1, _) if passive => Art::Hole,
        (2, "R" | "RN") if passive => Art::Resistor,
        (2, "C") if passive => Art::Capacitor,
        (2, "L" | "FB") if passive => Art::Inductor,
        (2, "D" | "LED") => Art::Diode,
        _ => Art::Box,
    }
}

/// A placed symbol's transform: mirror (x → −x), then a counter-clockwise quarter turn
/// count, then the origin. KiCad's `(mirror y)` and `(at x y angle)` are the same.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Place {
    pub origin: Point,
    /// Degrees, 0, 90, 180 or 270.
    pub rotation: u16,
    pub mirror: bool,
}

impl Place {
    pub fn at(origin: Point) -> Place {
        Place {
            origin,
            ..Default::default()
        }
    }
    pub fn apply(&self, p: Point) -> Point {
        let v = self.vector(p);
        Point::new(self.origin.x + v.x, self.origin.y + v.y)
    }
    /// The transform without the translation (a direction or offset).
    pub fn vector(&self, p: Point) -> Point {
        let x = if self.mirror { -p.x } else { p.x };
        match self.rotation % 360 {
            90 => Point::new(-p.y, x),
            180 => Point::new(-x, -p.y),
            270 => Point::new(p.y, -x),
            _ => Point::new(x, p.y),
        }
    }
    pub fn side(&self, s: PinSide) -> PinSide {
        let (dx, dy) = s.out();
        let v = self.vector(Point::new(dx, dy));
        PinSide::from_out((v.x, v.y))
    }
    /// A box `[x0, y0, x1, y1]` in library coordinates, placed.
    pub fn rect(&self, r: [Coord; 4]) -> [Coord; 4] {
        let a = self.apply(Point::new(r[0], r[1]));
        let b = self.apply(Point::new(r[2], r[3]));
        [a.x.min(b.x), a.y.min(b.y), a.x.max(b.x), a.y.max(b.y)]
    }
    /// Whether the transform turns a horizontal line vertical.
    pub fn upright(&self) -> bool {
        self.rotation % 180 == 90
    }
}
