//! Board display geometry for the circuit projection: where every pad, track, via, pour,
//! keepout, room, text and dimension sits, in board millimetres (y up, angles in degrees),
//! so a canvas can draw the board from the projection alone.
//!
//! Curves are flattened to polylines within [`DISPLAY_TOL`]; pads whose shape the canvas
//! draws from its parameters (circle, rectangle, rounded rectangle, oval) carry the parameters,
//! and the others carry their exact outline. Every list follows the board's item order, so
//! identical boards give identical bytes.

use crate::geom::poly::Ring;
use crate::geom::shape::{arc_from_three, arc_points};
use crate::geom::{Coord, Point, Transform};
use crate::layout::{copper_name, Board, Footprint, Layer, Side, TrackGeom};
use crate::library::{Graphic, PadShape};
use serde::{Deserialize, Serialize};

/// Chord tolerance for flattened curves: 5 µm, below anything a screen or a plot shows.
pub const DISPLAY_TOL: Coord = 5_000;

/// A point in board millimetres.
pub type Xy = [f64; 2];

fn mm(nm: Coord) -> f64 {
    (nm as f64 / 1.0e6 * 1.0e6).round() / 1.0e6
}

pub fn xy(p: Point) -> Xy {
    [mm(p.x), mm(p.y)]
}

fn ring(r: &Ring) -> Vec<Xy> {
    r.iter().map(|p| xy(*p)).collect()
}

/// A layer's display name (`F.Cu`, `In2.Cu`, `B.SilkS`, `Edge.Cuts`).
pub fn layer_name(l: Layer, copper: u8) -> String {
    let f =
        |s: Side, front: &str, back: &str| (if s == Side::Top { front } else { back }).to_owned();
    match l {
        Layer::Copper(i) => copper_name(i, copper),
        Layer::Mask(s) => f(s, "F.Mask", "B.Mask"),
        Layer::Paste(s) => f(s, "F.Paste", "B.Paste"),
        Layer::Silk(s) => f(s, "F.SilkS", "B.SilkS"),
        Layer::Fab(s) => f(s, "F.Fab", "B.Fab"),
        Layer::Courtyard(s) => f(s, "F.CrtYd", "B.CrtYd"),
        Layer::Edge => "Edge.Cuts".into(),
        Layer::User(0) => "Dwgs.User".into(),
        Layer::User(k) => format!("User.{k}"),
    }
}

/// The board outline: the outer boundary first, then each cut-out.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct OutlineGeometry {
    pub polygons: Vec<Vec<Xy>>,
    pub thickness_mm: f64,
}

pub fn outline(board: &Board) -> OutlineGeometry {
    let mut polygons = Vec::new();
    if board.outline.outer.len() >= 3 {
        polygons.push(ring(&board.outline.outer));
        polygons.extend(board.outline.holes.iter().map(ring));
    }
    OutlineGeometry {
        polygons,
        thickness_mm: mm(board.stackup.thickness()),
    }
}

/// A graphic stroke in board coordinates: a polyline (closed when the first point repeats
/// at the end), its pen width, and whether its interior is filled.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Stroke {
    pub points: Vec<Xy>,
    pub width_mm: f64,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub filled: bool,
}

/// Text placed on a layer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TextGeometry {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<u64>,
    pub layer: String,
    pub x_mm: f64,
    pub y_mm: f64,
    pub text: String,
    pub size_mm: f64,
    pub rotation: f64,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub mirrored: bool,
}

fn closed(mut pts: Vec<Xy>) -> Vec<Xy> {
    if let (Some(first), Some(last)) = (pts.first().copied(), pts.last().copied()) {
        if first != last {
            pts.push(first);
        }
    }
    pts
}

/// A library or board graphic placed by `t`: strokes, or a text item.
pub fn graphic(g: &Graphic, t: &Transform, layer: &str) -> (Option<Stroke>, Option<TextGeometry>) {
    let place = |p: Point| xy(t.apply(p));
    let stroke = |points: Vec<Xy>, width: Coord, filled: bool| {
        Some(Stroke {
            points,
            width_mm: mm(width),
            filled,
        })
    };
    match g {
        Graphic::Line { points, width } => (
            stroke(points.iter().map(|p| place(*p)).collect(), *width, false),
            None,
        ),
        Graphic::Rect { a, b, width, fill } => {
            let pts = [*a, Point::new(b.x, a.y), *b, Point::new(a.x, b.y)];
            (
                stroke(
                    closed(pts.iter().map(|p| place(*p)).collect()),
                    *width,
                    *fill,
                ),
                None,
            )
        }
        Graphic::Circle {
            center,
            radius,
            width,
            fill,
        } => {
            let r = crate::geom::poly::circle_ring(t.apply(*center), (*radius).max(1), DISPLAY_TOL);
            (stroke(closed(ring(&r)), *width, *fill), None)
        }
        Graphic::Arc {
            start,
            mid,
            end,
            width,
        } => {
            let (s, m, e) = (t.apply(*start), t.apply(*mid), t.apply(*end));
            let pts = match arc_from_three(s, m, e) {
                Some((c, sweep)) => arc_points(c, s, sweep, DISPLAY_TOL)
                    .into_iter()
                    .map(xy)
                    .collect(),
                None => vec![xy(s), xy(e)],
            };
            (stroke(pts, *width, false), None)
        }
        Graphic::Polygon {
            points,
            width,
            fill,
        } => (
            stroke(
                closed(points.iter().map(|p| place(*p)).collect()),
                *width,
                *fill,
            ),
            None,
        ),
        Graphic::Text {
            at,
            text,
            size,
            angle,
        } => {
            let p = t.apply(*at);
            (
                None,
                Some(TextGeometry {
                    id: None,
                    layer: layer.to_owned(),
                    x_mm: mm(p.x),
                    y_mm: mm(p.y),
                    text: text.clone(),
                    size_mm: mm(*size),
                    rotation: t.apply_angle(*angle),
                    mirrored: t.mirror,
                }),
            )
        }
    }
}

/// A footprint's courtyard and silkscreen in board coordinates.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FootprintGeometry {
    pub courtyard: Vec<Vec<Xy>>,
    pub silkscreen: Vec<Stroke>,
    pub fab: Vec<Stroke>,
    pub texts: Vec<TextGeometry>,
}

pub fn footprint(f: &Footprint) -> FootprintGeometry {
    let t = f.transform();
    let silk_layer = layer_name(Layer::Silk(f.side), 2);
    let fab_layer = layer_name(Layer::Fab(f.side), 2);
    let mut out = FootprintGeometry {
        courtyard: f.courtyard().iter().map(|r| closed(ring(r))).collect(),
        ..Default::default()
    };
    for (graphics, layer, fab) in [
        (&f.package.silk, &silk_layer, false),
        (&f.package.fab, &fab_layer, true),
    ] {
        for g in graphics {
            let (stroke, text) = graphic(g, &t, layer);
            if let Some(s) = stroke {
                if fab {
                    out.fab.push(s)
                } else {
                    out.silkscreen.push(s)
                }
            }
            out.texts.extend(text);
        }
    }
    out
}

/// A pad's placement: centre, absolute rotation, size, drill and, for shapes the canvas can't
/// draw from parameters, its outline.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PadGeometry {
    pub x_mm: f64,
    pub y_mm: f64,
    pub rotation: f64,
    pub size_x_mm: f64,
    pub size_y_mm: f64,
    pub drill_mm: Option<f64>,
    pub slot_mm: Option<f64>,
    pub corner_radius_mm: Option<f64>,
    pub outline: Option<Vec<Vec<Xy>>>,
}

pub fn pad(f: &Footprint, index: usize, copper: u8) -> Option<PadGeometry> {
    let def = f.package.pads.get(index)?;
    let placed = f.pad(index, copper)?;
    let t = f.transform();
    let (w, h, corner, exact) = match &def.shape {
        PadShape::Circle { d } => (*d, *d, None, false),
        PadShape::Rect { w, h } => (*w, *h, None, false),
        PadShape::RoundRect { w, h, ratio } => (
            *w,
            *h,
            Some(((*w).min(*h) as f64 * ratio.clamp(0.0, 0.5)) as Coord),
            false,
        ),
        PadShape::Oval { w, h } => (*w, *h, Some((*w).min(*h) / 2), false),
        PadShape::Trapezoid { w, h, .. } | PadShape::Chamfered { w, h, .. } => (*w, *h, None, true),
        PadShape::Custom { .. } => {
            let b = placed.prim.bbox();
            (b.max.x - b.min.x, b.max.y - b.min.y, None, true)
        }
    };
    Some(PadGeometry {
        x_mm: mm(placed.center.x),
        y_mm: mm(placed.center.y),
        rotation: t.apply_angle(def.rotation),
        size_x_mm: mm(w),
        size_y_mm: mm(h),
        drill_mm: def.drill.as_ref().map(|d| mm(d.diameter)),
        slot_mm: def.drill.as_ref().and_then(|d| d.slot).map(mm),
        corner_radius_mm: corner.map(mm),
        outline: exact.then(|| {
            placed
                .prim
                .outline(0, DISPLAY_TOL)
                .iter()
                .map(|r| closed(ring(r)))
                .collect()
        }),
    })
}

/// A track's centreline as a polyline, and the arc it was authored as.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ArcGeometry {
    pub center: Xy,
    pub start: Xy,
    pub mid: Xy,
    pub end: Xy,
    /// Counter-clockwise positive.
    pub sweep: f64,
}

pub fn track_points(geom: &TrackGeom) -> (Vec<Xy>, Option<ArcGeometry>) {
    match *geom {
        TrackGeom::Segment { a, b } => (vec![xy(a), xy(b)], None),
        TrackGeom::Arc { start, mid, end } => match arc_from_three(start, mid, end) {
            Some((c, sweep)) => (
                arc_points(c, start, sweep, DISPLAY_TOL)
                    .into_iter()
                    .map(xy)
                    .collect(),
                Some(ArcGeometry {
                    center: xy(c),
                    start: xy(start),
                    mid: xy(mid),
                    end: xy(end),
                    sweep: (sweep * 1.0e6).round() / 1.0e6,
                }),
            ),
            None => (vec![xy(start), xy(end)], None),
        },
    }
}

/// One copper layer of a pour: the authored outline and the copper left after clearances,
/// each filled polygon as its outer ring then its holes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PourLayer {
    pub layer: String,
    pub outline: Vec<Xy>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub filled: Vec<Vec<Vec<Xy>>>,
}

pub fn pour_layers(board: &Board, zone: &crate::layout::Zone) -> Vec<PourLayer> {
    let copper = board.copper_count();
    zone.layers
        .iter()
        .map(|l| PourLayer {
            layer: copper_name(*l, copper),
            outline: closed(ring(&zone.outline.outer)),
            filled: zone
                .fill
                .get(l)
                .map(|set| {
                    set.polys
                        .iter()
                        .map(|p| {
                            std::iter::once(&p.outer)
                                .chain(p.holes.iter())
                                .map(|r| closed(ring(r)))
                                .collect()
                        })
                        .collect()
                })
                .unwrap_or_default(),
        })
        .collect()
}

/// A keepout: where it applies and what it forbids.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KeepoutRow {
    pub id: u64,
    pub name: String,
    /// Copper layers it applies to; every copper layer when the keepout names none.
    pub layers: Vec<String>,
    pub outline: Vec<Xy>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub holes: Vec<Vec<Xy>>,
    /// `tracks`, `vias`, `pads`, `pours`, `components`.
    pub forbids: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_height_mm: Option<f64>,
}

pub fn keepouts(board: &Board) -> Vec<KeepoutRow> {
    let copper = board.copper_count();
    board
        .keepouts
        .values()
        .map(|k| {
            let r = &k.rules;
            let forbids = [
                (r.tracks, "tracks"),
                (r.vias, "vias"),
                (r.pads, "pads"),
                (r.pours, "pours"),
                (r.footprints, "components"),
            ]
            .into_iter()
            .filter(|(on, _)| *on)
            .map(|(_, s)| s.to_owned())
            .collect();
            let layers = if k.layers.is_empty() {
                (0..copper).map(|l| copper_name(l, copper)).collect()
            } else {
                k.layers.iter().map(|l| copper_name(*l, copper)).collect()
            };
            KeepoutRow {
                id: k.id.0,
                name: k.name.clone(),
                layers,
                outline: closed(ring(&k.outline.outer)),
                holes: k.outline.holes.iter().map(|h| closed(ring(h))).collect(),
                forbids,
                max_height_mm: r.max_height.map(mm),
            }
        })
        .collect()
}

/// A board graphic (silkscreen art, fabrication notes, user drawings).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraphicRow {
    pub id: u64,
    pub layer: String,
    #[serde(flatten)]
    pub stroke: Stroke,
}

/// A dimension: the two measured points, the offset of its line and the text it shows.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DimensionRow {
    pub id: u64,
    pub layer: String,
    pub a: Xy,
    pub b: Xy,
    /// Offset of the dimension line from the measured points, perpendicular to a → b
    /// (positive to the left of a → b).
    pub offset_mm: f64,
    pub value_mm: f64,
    pub text: String,
}

pub fn dimensions(board: &Board) -> Vec<DimensionRow> {
    let copper = board.copper_count();
    board
        .dimensions
        .values()
        .map(|d| {
            let value = d.a.dist(d.b) / 1.0e6;
            let precision = d.precision.min(6) as usize;
            DimensionRow {
                id: d.id.0,
                layer: layer_name(d.layer, copper),
                a: xy(d.a),
                b: xy(d.b),
                offset_mm: mm(d.offset),
                value_mm: (value * 1.0e6).round() / 1.0e6,
                text: format!("{value:.precision$} mm"),
            }
        })
        .collect()
}

/// Texts and graphics placed on the board itself (not inside a footprint).
pub fn board_texts(board: &Board) -> Vec<TextGeometry> {
    let copper = board.copper_count();
    board
        .texts
        .values()
        .map(|t| TextGeometry {
            id: Some(t.id.0),
            layer: layer_name(t.layer, copper),
            x_mm: mm(t.at.x),
            y_mm: mm(t.at.y),
            text: t.text.clone(),
            size_mm: mm(t.size),
            rotation: t.angle,
            mirrored: t.mirrored,
        })
        .collect()
}

pub fn board_graphics(board: &Board) -> Vec<GraphicRow> {
    let copper = board.copper_count();
    board
        .graphics
        .values()
        .filter_map(|g| {
            let layer = layer_name(g.layer, copper);
            let (stroke, _) = graphic(&g.graphic, &Transform::IDENTITY, &layer);
            stroke.map(|stroke| GraphicRow {
                id: g.id.0,
                layer,
                stroke,
            })
        })
        .collect()
}

/// An unrouted connection: the shortest link still needed between two copper islands of a net.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AirwireRow {
    pub net: String,
    pub from: Xy,
    pub to: Xy,
}

pub fn unrouted(board: &Board) -> Vec<AirwireRow> {
    let mut rows: Vec<AirwireRow> = crate::route::ratsnest(board)
        .into_iter()
        .map(|a| AirwireRow {
            net: board.net_name(a.net).to_owned(),
            from: xy(a.a),
            to: xy(a.b),
        })
        .collect();
    rows.sort_by(|a, b| {
        (&a.net, a.from[0], a.from[1], a.to[0], a.to[1])
            .partial_cmp(&(&b.net, b.from[0], b.from[1], b.to[0], b.to[1]))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    rows
}

/// A fiducial: a bare copper dot and its mask opening.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FiducialRow {
    pub name: String,
    pub x_mm: f64,
    pub y_mm: f64,
    pub side: String,
    pub diameter_mm: f64,
    pub opening_mm: f64,
}

pub fn fiducials(board: &Board) -> Vec<FiducialRow> {
    board
        .fiducials
        .values()
        .map(|f| FiducialRow {
            name: f.name.clone(),
            x_mm: mm(f.at.x),
            y_mm: mm(f.at.y),
            side: match f.side {
                Side::Top => "top",
                Side::Bottom => "bottom",
            }
            .to_owned(),
            diameter_mm: mm(f.diameter),
            opening_mm: mm(f.opening),
        })
        .collect()
}
