//! A catalog part's previews for the library panel: its schematic symbol as the schematic draws
//! it, its footprint (pads, courtyard, silk and fab outline) and its body box for the 3D view.
//!
//! The symbol comes from the same placer the schematic uses (a one-component design laid out
//! on its own), so the preview matches what placing the part draws.

use super::catalog::{add_to_pool_with, CatalogPart};
use super::{Graphic, PadKind, PadShape, Pool};
use crate::elab::{elaborate, ElabOptions};
use crate::geom::{to_mm, Point};
use crate::ir::{ComponentDecl, Design, PartRef};
use crate::schematic::{layout, SchematicSymbol};
use serde::Serialize;

pub type Xy = [f64; 2];

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PadPreview {
    pub name: String,
    /// `smd`, `tht`, `npth` or `connector`.
    pub kind: &'static str,
    /// `circle`, `rect`, `roundrect`, `oval` or `polygon`.
    pub shape: &'static str,
    pub x_mm: f64,
    pub y_mm: f64,
    pub w_mm: f64,
    pub h_mm: f64,
    pub rotation: f64,
    /// Corner radius over the smaller side, for `roundrect`.
    pub ratio: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub drill_mm: Option<f64>,
    /// The outline, for `polygon` (trapezoid, chamfered and custom pads).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub outline: Vec<Xy>,
    /// The pin the pad carries (`VIN`), when the part maps one to it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pin: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct FootprintPreview {
    pub name: String,
    pub pads: Vec<PadPreview>,
    pub courtyard: Vec<Vec<Xy>>,
    /// Silk and fab lines (closed outlines repeat their first point).
    pub silk: Vec<Vec<Xy>>,
    pub fab: Vec<Vec<Xy>>,
    /// Where silk circles mark pin 1: `[x, y, r]`.
    pub dots: Vec<[f64; 3]>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct BodyPreview {
    pub width_mm: f64,
    pub length_mm: f64,
    pub height_mm: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PartPreview {
    pub mpn: String,
    pub symbol: SchematicSymbol,
    pub footprint: FootprintPreview,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<BodyPreview>,
}

fn xy(p: Point) -> Xy {
    [to_mm(p.x), to_mm(p.y)]
}

fn rect_ring(a: Point, b: Point) -> Vec<Xy> {
    let (a, b) = (xy(a), xy(b));
    vec![[a[0], a[1]], [b[0], a[1]], [b[0], b[1]], [a[0], b[1]], [a[0], a[1]]]
}

fn lines(graphics: &[Graphic], dots: &mut Vec<[f64; 3]>) -> Vec<Vec<Xy>> {
    let mut out = Vec::new();
    for g in graphics {
        match g {
            Graphic::Line { points, .. } => out.push(points.iter().map(|p| xy(*p)).collect()),
            Graphic::Polygon { points, .. } => {
                let mut ring: Vec<Xy> = points.iter().map(|p| xy(*p)).collect();
                if let Some(first) = ring.first().copied() {
                    ring.push(first);
                }
                out.push(ring);
            }
            Graphic::Rect { a, b, .. } => out.push(rect_ring(*a, *b)),
            Graphic::Arc { start, mid, end, .. } => out.push(vec![xy(*start), xy(*mid), xy(*end)]),
            Graphic::Circle { center, radius, .. } => {
                let c = xy(*center);
                dots.push([c[0], c[1], to_mm(*radius)]);
            }
            Graphic::Text { .. } => {}
        }
    }
    out
}

/// The previews of `part`, or why the record cannot be drawn.
pub fn preview(part: &CatalogPart) -> Result<PartPreview, String> {
    preview_with(part, &|_| None)
}

/// [`preview`], reading a footprint-file package through `file` (by the path as written).
pub fn preview_with(part: &CatalogPart, file: &dyn Fn(&str) -> Option<String>) -> Result<PartPreview, String> {
    let mut pool = Pool::new();
    let id = add_to_pool_with(&mut pool, part, file)?;
    let stored = pool.parts[&id].clone();
    let prefix = pool.entities[&stored.entity].prefix.clone();
    let mut design = Design::new("preview");
    design
        .sheet_mut("main")
        .components
        .push(ComponentDecl::new(&format!("{prefix}1"), PartRef::Part(part.mpn.clone())));
    let netlist = elaborate(&design, &pool, &ElabOptions::default(), None);
    let symbol = layout(&netlist)
        .sheets
        .into_iter()
        .flat_map(|s| s.symbols)
        .next()
        .ok_or_else(|| format!("part `{}` lays out no symbol", part.mpn))?;
    let package = &pool.packages[&stored.package];
    let pin_of = |pad: &str| stored.pad_map.iter().find(|m| m.pads.iter().any(|p| p == pad)).map(|m| m.pin.clone());
    let pads = package
        .pads
        .iter()
        .map(|p| {
            let (w, h) = p.shape.size();
            let (shape, ratio, outline) = match &p.shape {
                PadShape::Circle { .. } => ("circle", 0.0, vec![]),
                PadShape::Rect { .. } => ("rect", 0.0, vec![]),
                PadShape::RoundRect { ratio, .. } => ("roundrect", *ratio, vec![]),
                PadShape::Oval { .. } => ("oval", 0.0, vec![]),
                PadShape::Trapezoid { w, h, delta } => {
                    let (w, h, d) = (to_mm(*w) / 2.0, to_mm(*h) / 2.0, to_mm(*delta) / 2.0);
                    ("polygon", 0.0, vec![[-w + d, -h], [w - d, -h], [w + d, h], [-w - d, h]])
                }
                PadShape::Chamfered { .. } => ("rect", 0.0, vec![]),
                PadShape::Custom { points, .. } => ("polygon", 0.0, points.iter().map(|q| xy(*q)).collect()),
            };
            PadPreview {
                name: p.name.clone(),
                kind: match p.kind {
                    PadKind::Smd => "smd",
                    PadKind::Tht => "tht",
                    PadKind::Npth => "npth",
                    PadKind::Connector => "connector",
                },
                shape,
                x_mm: to_mm(p.at.x),
                y_mm: to_mm(p.at.y),
                w_mm: to_mm(w),
                h_mm: to_mm(h),
                rotation: p.rotation,
                ratio,
                drill_mm: p.drill.as_ref().map(|d| to_mm(d.diameter)),
                outline,
                pin: pin_of(&p.name),
            }
        })
        .collect();
    let mut dots = Vec::new();
    let silk = lines(&package.silk, &mut dots);
    let fab = lines(&package.fab, &mut Vec::new());
    let footprint = FootprintPreview {
        name: package.name.clone(),
        pads,
        courtyard: package
            .courtyard
            .iter()
            .map(|r| {
                let mut ring: Vec<Xy> = r.iter().map(|p| xy(*p)).collect();
                if let Some(first) = ring.first().copied() {
                    ring.push(first);
                }
                ring
            })
            .collect(),
        silk,
        fab,
        dots,
    };
    let body = package.body.as_ref().map(|b| BodyPreview {
        width_mm: to_mm(b.width),
        length_mm: to_mm(b.length),
        height_mm: to_mm(b.height),
    });
    Ok(PartPreview { mpn: part.mpn.clone(), symbol, footprint, body })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::library::catalog::CatalogPin;

    fn buck() -> CatalogPart {
        let pin = |name: &str, pad: &str, kind: &str| CatalogPin { name: name.into(), pads: vec![pad.into()], kind: kind.into() };
        CatalogPart {
            mpn: "BUCK-36V-2A".into(),
            manufacturer: "Generic".into(),
            category: "regulator".into(),
            package: "SOT23-6".into(),
            pins: vec![
                pin("BST", "1", "passive"),
                pin("GND", "2", "power_in"),
                pin("FB", "3", "input"),
                pin("EN", "4", "input"),
                pin("VIN", "5", "power_in"),
                pin("SW", "6", "power_out"),
            ],
            ..Default::default()
        }
    }

    #[test]
    fn a_regulator_previews_as_a_box_symbol_six_pads_and_a_body() {
        let p = preview(&buck()).unwrap();
        assert_eq!(p.symbol.designator, "U1");
        assert_eq!(p.symbol.art, "box");
        assert_eq!(p.symbol.pins.len(), 6);
        assert_eq!(p.footprint.pads.len(), 6);
        assert_eq!(p.footprint.pads[4].pin.as_deref(), Some("VIN"));
        assert!(!p.footprint.courtyard.is_empty() && !p.footprint.silk.is_empty());
        let body = p.body.unwrap();
        assert!(body.height_mm > 0.5 && body.height_mm < 2.0, "{body:?}");
    }

    #[test]
    fn a_resistor_previews_with_its_standard_art() {
        let r = CatalogPart {
            mpn: "R0402-10K".into(),
            manufacturer: "Generic".into(),
            category: "resistor".into(),
            package: "R0402".into(),
            pins: vec![
                CatalogPin { name: "1".into(), pads: vec!["1".into()], kind: "passive".into() },
                CatalogPin { name: "2".into(), pads: vec!["2".into()], kind: "passive".into() },
            ],
            ..Default::default()
        };
        let p = preview(&r).unwrap();
        assert_eq!((p.symbol.designator.as_str(), p.symbol.art.as_str()), ("R1", "resistor"));
        assert_eq!(p.footprint.pads.len(), 2);
    }
}
