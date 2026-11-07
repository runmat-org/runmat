//! Board texts, graphics, dimensions and fiducials, read from and written to KiCad nodes, and
//! KiCad layer names.

use crate::layer_set::CopperIndex;
use super::*;
use super::super::footprint;
use super::super::sexpr::SExpr;
use super::super::{at_of, copper_index, copper_layer, len, mm_of, uuid_node, xy};
use crate::geom::{mm, Coord};

/// A non-copper layer by its KiCad name; copper layers by [`copper_index`].
pub(super) fn layer_of(name: &str, n: CopperIndex) -> Option<Layer> {
    if let Some(i) = copper_index(name, n) {
        return Some(Layer::Copper(i));
    }
    let side = |f: bool| if f { Side::Top } else { Side::Bottom };
    Some(match name {
        "F.Mask" | "B.Mask" => Layer::Mask(side(name.starts_with('F'))),
        "F.Paste" | "B.Paste" => Layer::Paste(side(name.starts_with('F'))),
        "F.SilkS" | "B.SilkS" => Layer::Silk(side(name.starts_with('F'))),
        "F.Fab" | "B.Fab" => Layer::Fab(side(name.starts_with('F'))),
        "F.CrtYd" | "B.CrtYd" => Layer::Courtyard(side(name.starts_with('F'))),
        "Edge.Cuts" => Layer::Edge,
        "Dwgs.User" => Layer::User(0),
        _ => Layer::User(
            name.strip_prefix("User.")?
                .parse()
                .ok()
                .filter(|k| *k > 0)?,
        ),
    })
}

/// The KiCad name of a layer.
pub(super) fn layer_name(l: Layer, n: CopperIndex) -> String {
    let f =
        |s: Side, front: &str, back: &str| (if s == Side::Top { front } else { back }).to_owned();
    match l {
        Layer::Copper(i) => copper_layer(i, n),
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

pub(super) fn font_size(node: &SExpr) -> (Option<Coord>, Option<Coord>) {
    let font = node.child("effects").and_then(|e| e.child("font"));
    (
        font.and_then(|f| mm_of(f.child("size"))),
        font.and_then(|f| mm_of(f.child("thickness"))),
    )
}

pub(super) fn text_of(node: &SExpr, id: ItemId, n: CopperIndex) -> Option<BoardText> {
    let layer = layer_of(node.child_arg("layer")?, n)?;
    let (at, angle) = at_of(node.child("at"));
    let (size, thickness) = font_size(node);
    let size = size.unwrap_or(mm(1.0));
    let mirrored = node
        .child("effects")
        .and_then(|e| e.child("justify"))
        .is_some_and(|j| j.items().iter().skip(1).any(|x| x.atom() == Some("mirror")));
    Some(BoardText {
        id,
        layer,
        at,
        text: node.arg(0)?.to_owned(),
        size,
        thickness: thickness.unwrap_or(size / 7),
        angle,
        mirrored,
    })
}

pub(super) fn text_node(t: &BoardText, n: CopperIndex) -> SExpr {
    let mut effects = vec![SExpr::list(
        "font",
        vec![
            SExpr::list("size", vec![len(t.size), len(t.size)]),
            SExpr::list("thickness", vec![len(t.thickness)]),
        ],
    )];
    if t.mirrored {
        effects.push(SExpr::list("justify", vec![SExpr::sym("mirror")]));
    }
    SExpr::list(
        "gr_text",
        vec![
            SExpr::str(&t.text),
            SExpr::list("at", vec![len(t.at.x), len(-t.at.y), SExpr::num(t.angle)]),
            SExpr::list("layer", vec![SExpr::str(&layer_name(t.layer, n))]),
            uuid_node(&format!("x{}", t.id.0)),
            SExpr::list("effects", effects),
        ],
    )
}

pub(super) fn graphic_of(node: &SExpr, id: ItemId, n: CopperIndex) -> Option<BoardGraphic> {
    let (layer, graphic) = footprint::graphic(node, &|p| p)?;
    Some(BoardGraphic {
        id,
        layer: layer_of(&layer, n)?,
        graphic,
    })
}

pub(super) fn board_graphic_node(g: &BoardGraphic, n: CopperIndex) -> SExpr {
    let mut node = footprint::graphic_node(
        &g.graphic,
        &layer_name(g.layer, n),
        &|p| p,
        &format!("g{}", g.id.0),
    );
    if let Some(head) = node.items_mut().and_then(|v| v.first_mut()) {
        if let Some(name) = head.atom() {
            *head = SExpr::sym(&name.replacen("fp_", "gr_", 1));
        }
    }
    node
}

/// An aligned dimension (`(dimension (type aligned) … (pts a b) (height h))`); other types stay
/// in the tree.
pub(super) fn dimension_of(node: &SExpr, id: ItemId, n: CopperIndex) -> Option<Dimension> {
    if node.child_arg("type") != Some("aligned") {
        return None;
    }
    let pts = super::super::pts_of(node.child("pts"));
    let (a, b) = (*pts.first()?, *pts.get(1)?);
    Some(Dimension {
        id,
        layer: layer_of(node.child_arg("layer")?, n)?,
        a,
        b,
        // KiCad's height is measured to the left of a → b with y down: RunMat's offset sign.
        offset: -mm_of(node.child("height")).unwrap_or(0),
        precision: node
            .child("format")
            .and_then(|f| f.child_f("precision"))
            .map_or(2, |p| p as u8),
    })
}

pub(super) fn dimension_node(d: &Dimension, n: CopperIndex) -> SExpr {
    SExpr::list(
        "dimension",
        vec![
            SExpr::list("type", vec![SExpr::sym("aligned")]),
            SExpr::list("layer", vec![SExpr::str(&layer_name(d.layer, n))]),
            uuid_node(&format!("d{}", d.id.0)),
            SExpr::list("pts", vec![xy("xy", d.a), xy("xy", d.b)]),
            SExpr::list("height", vec![len(-d.offset)]),
            SExpr::list(
                "format",
                vec![
                    SExpr::list("units", vec![SExpr::sym("2")]),
                    SExpr::list("units_format", vec![SExpr::sym("1")]),
                    SExpr::list("precision", vec![SExpr::sym(&d.precision.to_string())]),
                ],
            ),
            SExpr::list(
                "style",
                vec![
                    SExpr::list("thickness", vec![SExpr::num(0.15)]),
                    SExpr::list("arrow_length", vec![SExpr::num(1.27)]),
                    SExpr::list("text_position_mode", vec![SExpr::sym("0")]),
                    SExpr::list("extension_height", vec![SExpr::num(0.58642)]),
                    SExpr::list("keep_text_aligned", vec![SExpr::sym("yes")]),
                ],
            ),
        ],
    )
}

/// A fiducial: a footprint from a `Fiducial` library entry with one round pad and no net. The
/// mask opening is the pad plus twice its `solder_mask_margin` (the pad itself without one).
pub(super) fn fiducial_of(node: &SExpr, id: ItemId) -> Option<Fiducial> {
    let lib = node.arg(0)?;
    if !lib.to_ascii_lowercase().contains("fiducial") {
        return None;
    }
    let pads: Vec<&SExpr> = node.children("pad").collect();
    let [pad] = pads.as_slice() else { return None };
    if pad.child("net").is_some() || pad.arg(2) != Some("circle") {
        return None;
    }
    let diameter = mm_of(pad.child("size"))?;
    let margin = mm_of(pad.child("solder_mask_margin")).unwrap_or(0);
    let side = if node.child_arg("layer") == Some("B.Cu") {
        Side::Bottom
    } else {
        Side::Top
    };
    let name = node
        .children("property")
        .find(|p| p.arg(0) == Some("Reference"))
        .and_then(|p| p.arg(1))
        .or_else(|| {
            node.children("fp_text")
                .find(|t| t.arg(0) == Some("reference"))
                .and_then(|t| t.arg(1))
        })
        .unwrap_or("FID")
        .to_owned();
    Some(Fiducial {
        id,
        name,
        at: at_of(node.child("at")).0,
        side,
        diameter,
        opening: diameter + 2 * margin,
        locked: node.has_flag("locked") || node.child("locked").is_some(),
    })
}

pub(super) fn fiducial_node(f: &Fiducial) -> SExpr {
    let (cu, mask, silk) = if f.side == Side::Top {
        ("F.Cu", "F.Mask", "F.SilkS")
    } else {
        ("B.Cu", "B.Mask", "B.SilkS")
    };
    let mm_text = crate::syntax::values::mm_text;
    let mut items = vec![
        SExpr::str(&format!(
            "Fiducial:Fiducial_{}mm_Mask{}mm",
            mm_text(f.diameter),
            mm_text(f.opening)
        )),
        SExpr::list("layer", vec![SExpr::str(cu)]),
        uuid_node(&format!("f{}", f.id.0)),
        SExpr::list("at", vec![len(f.at.x), len(-f.at.y)]),
        SExpr::list(
            "property",
            vec![
                SExpr::str("Reference"),
                SExpr::str(&f.name),
                SExpr::list("at", vec![SExpr::num(0.0), SExpr::num(-1.5)]),
                SExpr::list("layer", vec![SExpr::str(silk)]),
                SExpr::list("hide", vec![SExpr::sym("yes")]),
            ],
        ),
        SExpr::list(
            "attr",
            vec![SExpr::sym("smd"), SExpr::sym("exclude_from_bom")],
        ),
        SExpr::list(
            "pad",
            vec![
                SExpr::str("1"),
                SExpr::sym("smd"),
                SExpr::sym("circle"),
                SExpr::list("at", vec![SExpr::num(0.0), SExpr::num(0.0)]),
                SExpr::list("size", vec![len(f.diameter), len(f.diameter)]),
                SExpr::list("layers", vec![SExpr::str(cu), SExpr::str(mask)]),
                SExpr::list(
                    "solder_mask_margin",
                    vec![len((f.opening - f.diameter) / 2)],
                ),
            ],
        ),
    ];
    if f.locked {
        items.insert(1, SExpr::sym("locked"));
    }
    SExpr::list("footprint", items)
}
