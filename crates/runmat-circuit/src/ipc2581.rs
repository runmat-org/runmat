//! IPC-2581 revision C export: one XML file carrying the whole fabrication and assembly
//! data set.
//!
//! Written from the structure of the published standard (IPC-2581C, "Generic Requirements
//! for Printed Board Assembly Products Manufacturing Description Data and Transfer
//! Methodology"): `Content` (function mode, step and layer references, dictionaries of
//! standard primitives and line descriptions), `LogisticHeader`, `HistoryRecord`, `Bom`, and
//! `Ecad` with the layer list, the stackup, and one `Step` holding the profile, packages
//! (pins with their pad primitives), components, logical nets and per-layer features
//! (tracks as lines, pads, vias, zone fills as contours with cutouts, legend graphics and text
//! strokes, drill holes per span).
//!
//! Units are millimetres; coordinates are the board's (y up). Output is deterministic.
//! The file has not been validated against the consortium's XSD inside this crate (no XML
//! schema dependency); element order follows the standard's sequences.

use crate::geom::poly::{Polygon, Ring};
use crate::geom::shape::Core;
use crate::geom::{to_mm, Coord, Point};
use crate::layout::{copper_name, Board, Layer, Side, TrackGeom};
use crate::library::{Graphic, PadKind, PadShape};
use std::collections::BTreeMap;
use std::fmt::Write as _;

fn n(v: Coord) -> String {
    let s = format!("{:.6}", to_mm(v));
    let s = s.trim_end_matches('0').trim_end_matches('.').to_string();
    if s == "-0" {
        "0".into()
    } else {
        s
    }
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn polygon(out: &mut String, tag: &str, r: &Ring, ind: &str) {
    if r.len() < 3 {
        return;
    }
    let _ = writeln!(out, "{ind}<{tag}>");
    let _ = writeln!(
        out,
        "{ind}  <PolyBegin x=\"{}\" y=\"{}\"/>",
        n(r[0].x),
        n(r[0].y)
    );
    for p in r.iter().skip(1).chain(std::iter::once(&r[0])) {
        let _ = writeln!(
            out,
            "{ind}  <PolyStepSegment x=\"{}\" y=\"{}\"/>",
            n(p.x),
            n(p.y)
        );
    }
    let _ = writeln!(out, "{ind}</{tag}>");
}

fn contour(out: &mut String, p: &Polygon, ind: &str) {
    let _ = writeln!(out, "{ind}<Contour>");
    polygon(out, "Polygon", &p.outer, &format!("{ind}  "));
    for h in &p.holes {
        let mut h = h.clone();
        h.reverse();
        polygon(out, "Cutout", &h, &format!("{ind}  "));
    }
    let _ = writeln!(out, "{ind}</Contour>");
}

/// Standard-primitive dictionary entries, keyed by their XML body.
#[derive(Default)]
struct Dict {
    shapes: BTreeMap<String, String>,
    lines: BTreeMap<Coord, String>,
}

impl Dict {
    fn shape(&mut self, body: String) -> String {
        let k = self.shapes.len();
        self.shapes
            .entry(body)
            .or_insert_with(|| format!("PRIM_{k}"))
            .clone()
    }
    fn line(&mut self, w: Coord) -> String {
        let k = self.lines.len();
        self.lines
            .entry(w)
            .or_insert_with(|| format!("LINE_{k}"))
            .clone()
    }
    /// A pad shape as a standard primitive (local frame, unrotated).
    fn pad(&mut self, s: &PadShape) -> String {
        let body = match s {
            PadShape::Circle { d } => format!("<Circle diameter=\"{}\"/>", n(*d)),
            PadShape::Rect { w, h } => {
                format!("<RectCenter width=\"{}\" height=\"{}\"/>", n(*w), n(*h))
            }
            PadShape::Oval { w, h } => format!("<Oval width=\"{}\" height=\"{}\"/>", n(*w), n(*h)),
            PadShape::RoundRect { w, h, ratio } => format!(
                "<RectRound width=\"{}\" height=\"{}\" radius=\"{}\" upperRight=\"true\" upperLeft=\"true\" lowerRight=\"true\" lowerLeft=\"true\"/>",
                n(*w),
                n(*h),
                n(((*w).min(*h) as f64 * ratio.clamp(0.0, 0.5)) as Coord)
            ),
            PadShape::Chamfered { w, h, chamfer, corners } => format!(
                "<RectCham width=\"{}\" height=\"{}\" chamfer=\"{}\" upperRight=\"{}\" upperLeft=\"{}\" lowerRight=\"{}\" lowerLeft=\"{}\"/>",
                n(*w),
                n(*h),
                n(*chamfer),
                corners & 2 != 0,
                corners & 1 != 0,
                corners & 8 != 0,
                corners & 4 != 0
            ),
            PadShape::Trapezoid { w, h, delta } => {
                let (hw, hh, hd) = (w / 2, h / 2, delta / 2);
                let r = vec![
                    Point::new(-hw + hd, -hh),
                    Point::new(hw - hd, -hh),
                    Point::new(hw + hd, hh),
                    Point::new(-hw - hd, hh),
                ];
                let mut s = String::new();
                contour(&mut s, &Polygon { outer: r, holes: vec![] }, "");
                s
            }
            PadShape::Custom { points, holes } => {
                let mut s = String::new();
                contour(
                    &mut s,
                    &Polygon {
                        outer: points.clone(),
                        holes: holes.clone(),
                    },
                    "",
                );
                s
            }
        };
        self.shape(body)
    }
}

fn layer_names(board: &Board) -> Vec<(String, &'static str, &'static str)> {
    let c = board.copper_count();
    let mut v = vec![
        ("SILK_TOP".to_string(), "SILKSCREEN", "TOP"),
        ("PASTE_TOP".to_string(), "SOLDERPASTE", "TOP"),
        ("MASK_TOP".to_string(), "SOLDERMASK", "TOP"),
    ];
    for i in 0..c {
        let side = if i == 0 {
            "TOP"
        } else if i + 1 == c {
            "BOTTOM"
        } else {
            "INTERNAL"
        };
        v.push((copper_name(i, c), "SIGNAL", side));
    }
    v.push(("MASK_BOTTOM".into(), "SOLDERMASK", "BOTTOM"));
    v.push(("PASTE_BOTTOM".into(), "SOLDERPASTE", "BOTTOM"));
    v.push(("SILK_BOTTOM".into(), "SILKSCREEN", "BOTTOM"));
    v
}

/// The IPC-2581C document for a board. `bom_parts` maps a reference designator to its part
/// number (MPN) for the BOM section (components without one use their value).
pub fn export(board: &Board, bom_parts: &BTreeMap<String, String>) -> String {
    let c = board.copper_count();
    let mut dict = Dict::default();
    let mut step = String::new();
    let ind = "      ";
    // Profile.
    let _ = writeln!(step, "{ind}<Datum x=\"0\" y=\"0\"/>");
    let _ = writeln!(step, "{ind}<Profile>");
    polygon(
        &mut step,
        "Polygon",
        &board.outline.outer,
        &format!("{ind}  "),
    );
    for h in &board.outline.holes {
        let mut h = h.clone();
        h.reverse();
        polygon(&mut step, "Cutout", &h, &format!("{ind}  "));
    }
    let _ = writeln!(step, "{ind}</Profile>");
    // Packages, once per package name.
    let mut packages: BTreeMap<String, &crate::library::Package> = BTreeMap::new();
    for f in board.footprints.values() {
        packages.entry(f.package.name.clone()).or_insert(&f.package);
    }
    for (name, pkg) in &packages {
        let _ = writeln!(
            step,
            "{ind}<Package name=\"{}\" type=\"OTHER\" pinOne=\"{}\" pinOneOrientation=\"OTHER\" height=\"{}\">",
            esc(name),
            esc(pkg.pads.first().map(|p| p.name.as_str()).unwrap_or("1")),
            n(pkg.body.map(|b| b.height).unwrap_or(0))
        );
        // The outline: the courtyard, else the box around the pads.
        let pad_box = || {
            let b = pkg
                .pads
                .iter()
                .map(|p| crate::layout::pad_prim(p, &crate::geom::Transform::IDENTITY).bbox())
                .reduce(|a, b| a.union(b))
                .unwrap_or(crate::geom::Rect::new(Point::ORIGIN, Point::ORIGIN))
                .inflate(crate::geom::mm(0.05));
            vec![
                b.min,
                Point::new(b.max.x, b.min.y),
                b.max,
                Point::new(b.min.x, b.max.y),
            ]
        };
        let outline = pkg.courtyard.first().cloned().unwrap_or_else(pad_box);
        {
            let _ = writeln!(step, "{ind}  <Outline>");
            polygon(&mut step, "Polygon", &outline, &format!("{ind}    "));
            let lid = dict.line(crate::geom::mm(0.05));
            let _ = writeln!(step, "{ind}    <LineDescRef id=\"{lid}\"/>");
            let _ = writeln!(step, "{ind}  </Outline>");
        }
        for p in &pkg.pads {
            let (kind, mount) = match p.kind {
                PadKind::Tht => ("THRU", "THROUGH_HOLE_PIN"),
                PadKind::Npth => ("NONPLATED", "HOLE"),
                _ => ("SURFACE", "SURFACE_MOUNT_PAD"),
            };
            let prim = dict.pad(&p.shape);
            let _ = writeln!(
                step,
                "{ind}  <Pin number=\"{}\" type=\"{kind}\" electricalType=\"ELECTRICAL\" mountType=\"{mount}\">",
                esc(&p.name)
            );
            if p.rotation != 0.0 {
                let _ = writeln!(step, "{ind}    <Xform rotation=\"{}\"/>", p.rotation);
            }
            let _ = writeln!(
                step,
                "{ind}    <Location x=\"{}\" y=\"{}\"/>",
                n(p.at.x),
                n(p.at.y)
            );
            let _ = writeln!(step, "{ind}    <StandardPrimitiveRef id=\"{prim}\"/>");
            let _ = writeln!(step, "{ind}  </Pin>");
        }
        let _ = writeln!(step, "{ind}</Package>");
    }
    // Components.
    for f in board.footprints.values() {
        let side = if f.side == Side::Top {
            copper_name(0, c)
        } else {
            copper_name(c - 1, c)
        };
        let mount = if f.package.smd { "SMT" } else { "THMT" };
        let part = bom_parts
            .get(&f.reference)
            .cloned()
            .unwrap_or_else(|| f.value.clone());
        let _ = writeln!(
            step,
            "{ind}<Component refDes=\"{}\" packageRef=\"{}\" layerRef=\"{}\" mountType=\"{mount}\" part=\"{}\">",
            esc(&f.reference),
            esc(&f.package.name),
            esc(&side),
            esc(&part)
        );
        let _ = writeln!(
            step,
            "{ind}  <Xform rotation=\"{}\" mirror=\"{}\"/>",
            f.rotation,
            f.side == Side::Bottom
        );
        let _ = writeln!(
            step,
            "{ind}  <Location x=\"{}\" y=\"{}\"/>",
            n(f.at.x),
            n(f.at.y)
        );
        let _ = writeln!(step, "{ind}</Component>");
    }
    // Logical nets.
    let mut net_pins: BTreeMap<String, Vec<(String, String)>> = BTreeMap::new();
    for f in board.footprints.values() {
        for (pad, net) in &f.pad_nets {
            net_pins
                .entry(board.net_name(*net).to_string())
                .or_default()
                .push((f.reference.clone(), pad.clone()));
        }
    }
    for (net, pins) in &net_pins {
        let _ = writeln!(step, "{ind}<LogicalNet name=\"{}\">", esc(net));
        for (r, p) in pins {
            let _ = writeln!(
                step,
                "{ind}  <PinRef componentRef=\"{}\" pin=\"{}\"/>",
                esc(r),
                esc(p)
            );
        }
        let _ = writeln!(step, "{ind}</LogicalNet>");
    }
    // Copper features per layer, grouped by net.
    let net_of = |x: Option<crate::layout::NetId>| x.map(|k| board.net_name(k).to_string());
    for l in 0..c {
        let mut sets: BTreeMap<Option<String>, String> = BTreeMap::new();
        let fi = format!("{ind}    ");
        for t in board.tracks.values().filter(|t| t.layer == l) {
            let lid = dict.line(t.width);
            let s = sets.entry(net_of(t.net)).or_default();
            let pts: Vec<Point> = match t.geom {
                TrackGeom::Segment { a, b } => vec![a, b],
                TrackGeom::Arc { .. } => {
                    let ps = t.prims();
                    let mut v: Vec<Point> = ps
                        .iter()
                        .filter_map(|p| match p.core {
                            Core::Segment(a, _) => Some(a),
                            _ => None,
                        })
                        .collect();
                    if let Some(Core::Segment(_, b)) = ps.last().map(|p| &p.core) {
                        v.push(*b);
                    }
                    v
                }
            };
            for w in pts.windows(2) {
                let _ = writeln!(
                    s,
                    "{fi}<Features><Line startX=\"{}\" startY=\"{}\" endX=\"{}\" endY=\"{}\"><LineDescRef id=\"{lid}\"/></Line></Features>",
                    n(w[0].x), n(w[0].y), n(w[1].x), n(w[1].y)
                );
            }
        }
        for v in board.vias.values().filter(|v| v.layers().any(|x| x == l)) {
            let prim = dict.pad(&PadShape::Circle { d: v.diameter });
            let s = sets.entry(net_of(v.net)).or_default();
            let _ = writeln!(
                s,
                "{fi}<Pad><Location x=\"{}\" y=\"{}\"/><StandardPrimitiveRef id=\"{prim}\"/></Pad>",
                n(v.at.x),
                n(v.at.y)
            );
        }
        for f in board.footprints.values() {
            for (pd, pp) in f.package.pads.iter().zip(f.pads(c)) {
                if !pp.layers.contains(&l) || pd.kind == PadKind::Npth {
                    continue;
                }
                let prim = dict.pad(&pd.shape);
                let rot = f.transform().apply_angle(pd.rotation);
                let s = sets.entry(net_of(pp.net)).or_default();
                let _ = writeln!(
                    s,
                    "{fi}<Pad><Xform rotation=\"{}\" mirror=\"{}\"/><Location x=\"{}\" y=\"{}\"/><StandardPrimitiveRef id=\"{prim}\"/><PinRef componentRef=\"{}\" pin=\"{}\"/></Pad>",
                    crate::geom::normalize_deg(rot),
                    f.side == Side::Bottom,
                    n(pp.center.x),
                    n(pp.center.y),
                    esc(&f.reference),
                    esc(&pd.name)
                );
            }
        }
        for z in board.zones.values() {
            if let Some(fill) = z.fill.get(&l) {
                let s = sets.entry(net_of(z.net)).or_default();
                for p in &fill.polys {
                    let _ = writeln!(s, "{fi}<Features>");
                    contour(s, p, &format!("{fi}  "));
                    let _ = writeln!(s, "{fi}</Features>");
                }
            }
        }
        for t in board.texts.values().filter(|t| t.layer == Layer::Copper(l)) {
            text_lines(sets.entry(None).or_default(), &mut dict, t, &fi);
        }
        write_layer(&mut step, &copper_name(l, c), &sets, ind);
    }
    // Legend.
    for (side, lname) in [(Side::Top, "SILK_TOP"), (Side::Bottom, "SILK_BOTTOM")] {
        let mut s = String::new();
        let fi = format!("{ind}    ");
        for f in board.footprints.values().filter(|f| f.side == side) {
            let t = f.transform();
            for g in &f.package.silk {
                graphic_lines(&mut s, &mut dict, g, &t, &fi);
            }
        }
        for g in board
            .graphics
            .values()
            .filter(|g| g.layer == Layer::Silk(side))
        {
            graphic_lines(
                &mut s,
                &mut dict,
                &g.graphic,
                &crate::geom::Transform::IDENTITY,
                &fi,
            );
        }
        for t in board
            .texts
            .values()
            .filter(|t| t.layer == Layer::Silk(side))
        {
            text_lines(&mut s, &mut dict, t, &fi);
        }
        write_layer(&mut step, lname, &BTreeMap::from([(None, s)]), ind);
    }
    // Drills per span.
    let mut spans: BTreeMap<(u8, u8, bool), String> = BTreeMap::new();
    for v in board.vias.values() {
        let s = spans.entry((v.from, v.to, true)).or_default();
        let _ = writeln!(
            s,
            "{ind}    <Hole name=\"V{}\" diameter=\"{}\" platingStatus=\"VIA\" plusTol=\"0\" minusTol=\"0\" x=\"{}\" y=\"{}\"/>",
            v.id.0, n(v.drill), n(v.at.x), n(v.at.y)
        );
    }
    for p in board.all_pads() {
        if let Some(d) = p.drill {
            let s = spans.entry((0, c - 1, p.plated)).or_default();
            let _ = writeln!(
                s,
                "{ind}    <Hole name=\"{}\" diameter=\"{}\" platingStatus=\"{}\" plusTol=\"0\" minusTol=\"0\" x=\"{}\" y=\"{}\"/>",
                esc(&format!("{}-{}", p.footprint.0, p.name)),
                n(d),
                if p.plated { "PLATED" } else { "NONPLATED" },
                n(p.center.x),
                n(p.center.y)
            );
        }
    }
    let drill_name = |(a, b, pl): (u8, u8, bool)| {
        format!("DRILL_{}_{}{}", a + 1, b + 1, if pl { "" } else { "_NPTH" })
    };
    for (k, body) in &spans {
        let _ = writeln!(step, "{ind}<LayerFeature layerRef=\"{}\">", drill_name(*k));
        let _ = writeln!(step, "{ind}  <Set>");
        step.push_str(body);
        let _ = writeln!(step, "{ind}  </Set>");
        let _ = writeln!(step, "{ind}</LayerFeature>");
    }

    // Assemble.
    let layers = layer_names(board);
    let mut x = String::new();
    x.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    x.push_str("<IPC-2581 revision=\"C\" xmlns=\"http://webstds.ipc.org/2581\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\">\n");
    x.push_str("  <Content roleRef=\"Owner\">\n    <FunctionMode mode=\"USERDEF\"/>\n");
    let _ = writeln!(x, "    <StepRef name=\"{}\"/>", esc(&board.name));
    for (l, _, _) in &layers {
        let _ = writeln!(x, "    <LayerRef name=\"{}\"/>", esc(l));
    }
    for k in spans.keys() {
        let _ = writeln!(x, "    <LayerRef name=\"{}\"/>", drill_name(*k));
    }
    x.push_str("    <BomRef name=\"BOM\"/>\n");
    x.push_str("    <DictionaryLineDesc units=\"MILLIMETER\">\n");
    let mut lines: Vec<(&String, &Coord)> = dict.lines.iter().map(|(w, id)| (id, w)).collect();
    lines.sort();
    for (id, w) in lines {
        let _ = writeln!(
            x,
            "      <EntryLineDesc id=\"{id}\"><LineDesc lineEnd=\"ROUND\" lineWidth=\"{}\"/></EntryLineDesc>",
            n(*w)
        );
    }
    x.push_str("    </DictionaryLineDesc>\n");
    x.push_str("    <DictionaryStandard units=\"MILLIMETER\">\n");
    let mut shapes: Vec<(&String, &String)> = dict.shapes.iter().map(|(b, id)| (id, b)).collect();
    shapes.sort();
    for (id, body) in shapes {
        if body.starts_with("<Contour") {
            continue;
        }
        let _ = writeln!(x, "      <EntryStandard id=\"{id}\">{body}</EntryStandard>");
    }
    x.push_str("    </DictionaryStandard>\n    <DictionaryUser units=\"MILLIMETER\">\n");
    for (body, id) in &dict.shapes {
        if body.starts_with("<Contour") {
            let _ = writeln!(x, "      <EntryUser id=\"{id}\"><UserSpecial>\n{body}      </UserSpecial></EntryUser>");
        }
    }
    x.push_str("    </DictionaryUser>\n  </Content>\n");
    x.push_str("  <LogisticHeader>\n    <Role id=\"Owner\" roleFunction=\"SENDER\"/>\n    <Enterprise id=\"RunMat\" code=\"NONE\"/>\n    <Person name=\"RunMat\" enterpriseRef=\"RunMat\" roleRef=\"Owner\"/>\n  </LogisticHeader>\n");
    let _ = writeln!(
        x,
        "  <HistoryRecord number=\"1\" origination=\"2000-01-01T00:00:00\" software=\"RunMat\" lastChange=\"2000-01-01T00:00:00\">\n    <FileRevision fileRevisionId=\"1\" comment=\"\" label=\"\">\n      <SoftwarePackage name=\"runmat-circuit\" revision=\"{}\" vendor=\"Dystr\"><Certification certificationStatus=\"SELFTEST\"/></SoftwarePackage>\n    </FileRevision>\n  </HistoryRecord>",
        env!("CARGO_PKG_VERSION")
    );
    // BOM: one item per part, with its designators.
    let mut items: BTreeMap<(String, String), Vec<&crate::layout::Footprint>> = BTreeMap::new();
    for f in board.footprints.values().filter(|f| f.fitted) {
        let part = bom_parts
            .get(&f.reference)
            .cloned()
            .unwrap_or_else(|| f.value.clone());
        items
            .entry((part, f.package.name.clone()))
            .or_default()
            .push(f);
    }
    let _ = writeln!(x, "  <Bom name=\"BOM\">\n    <BomHeader assembly=\"{}\" revision=\"1\"><StepRef name=\"{}\"/></BomHeader>", esc(&board.name), esc(&board.name));
    for ((part, pkg), fs) in &items {
        let _ = writeln!(
            x,
            "    <BomItem OEMDesignNumberRef=\"{}\" quantity=\"{}\" pinCount=\"{}\" category=\"ELECTRICAL\">",
            esc(part),
            fs.len(),
            fs[0].package.pads.len()
        );
        for f in fs {
            let side = if f.side == Side::Top {
                copper_name(0, c)
            } else {
                copper_name(c - 1, c)
            };
            let _ = writeln!(
                x,
                "      <RefDes name=\"{}\" packageRef=\"{}\" populate=\"true\" layerRef=\"{}\"/>",
                esc(&f.reference),
                esc(pkg),
                esc(&side)
            );
        }
        x.push_str("      <Characteristics category=\"ELECTRICAL\"/>\n    </BomItem>\n");
    }
    x.push_str("  </Bom>\n");
    let _ = writeln!(
        x,
        "  <Ecad name=\"{}\">\n    <CadHeader units=\"MILLIMETER\"/>\n    <CadData>",
        esc(&board.name)
    );
    for (l, func, side) in &layers {
        let _ = writeln!(
            x,
            "      <Layer name=\"{}\" layerFunction=\"{func}\" side=\"{side}\" polarity=\"POSITIVE\"/>",
            esc(l)
        );
    }
    for (k, sl) in board.stackup.layers.iter().enumerate() {
        if let crate::layout::StackKind::Dielectric { core } = sl.kind {
            let _ = writeln!(
                x,
                "      <Layer name=\"{}\" layerFunction=\"{}\" side=\"INTERNAL\" polarity=\"POSITIVE\"/>",
                stack_layer_name(board, k),
                if core { "DIELCORE" } else { "DIELPREG" }
            );
        }
    }
    for k in spans.keys() {
        let _ = writeln!(
            x,
            "      <Layer name=\"{}\" layerFunction=\"DRILL\" side=\"ALL\" polarity=\"POSITIVE\"><Span fromLayer=\"{}\" toLayer=\"{}\"/></Layer>",
            drill_name(*k),
            esc(&copper_name(k.0, c)),
            esc(&copper_name(k.1, c))
        );
    }
    // Stackup.
    let total = board.stackup.thickness();
    let _ = writeln!(
        x,
        "      <Stackup name=\"PRIMARY\" overallThickness=\"{}\" tolPlus=\"0\" tolMinus=\"0\" whereMeasured=\"METAL\" stackupStatus=\"PROPOSED\">\n        <StackupGroup name=\"PRIMARY\" thickness=\"{}\" tolPlus=\"0\" tolMinus=\"0\">",
        n(total),
        n(total)
    );
    let mut seq = 1;
    for (k, sl) in board.stackup.layers.iter().enumerate() {
        let is_cu = |x: &crate::layout::StackLayer| {
            matches!(x.kind, crate::layout::StackKind::Copper { .. })
        };
        let name = if is_cu(sl) {
            let ci = board.stackup.layers[..k]
                .iter()
                .filter(|x| is_cu(x))
                .count() as u8;
            copper_name(ci, c)
        } else {
            stack_layer_name(board, k)
        };
        let _ = writeln!(
            x,
            "          <StackupLayer layerOrGroupRef=\"{}\" thickness=\"{}\" tolPlus=\"0\" tolMinus=\"0\" sequence=\"{seq}\"/>",
            esc(&name),
            n(sl.thickness)
        );
        seq += 1;
    }
    x.push_str("        </StackupGroup>\n      </Stackup>\n");
    let _ = writeln!(x, "      <Step name=\"{}\">", esc(&board.name));
    x.push_str(&step);
    x.push_str("      </Step>\n    </CadData>\n  </Ecad>\n</IPC-2581>\n");
    x
}

fn write_layer(out: &mut String, layer: &str, sets: &BTreeMap<Option<String>, String>, ind: &str) {
    if sets.values().all(|s| s.is_empty()) {
        return;
    }
    let _ = writeln!(out, "{ind}<LayerFeature layerRef=\"{}\">", esc(layer));
    for (net, body) in sets {
        if body.is_empty() {
            continue;
        }
        match net {
            Some(nm) => {
                let _ = writeln!(out, "{ind}  <Set net=\"{}\">", esc(nm));
            }
            None => {
                let _ = writeln!(out, "{ind}  <Set>");
            }
        }
        out.push_str(body);
        let _ = writeln!(out, "{ind}  </Set>");
    }
    let _ = writeln!(out, "{ind}</LayerFeature>");
}

fn polyline(out: &mut String, dict: &mut Dict, pts: &[Point], w: Coord, ind: &str) {
    let lid = dict.line(w.max(1));
    if pts.len() == 1 {
        let prim = dict.pad(&PadShape::Circle { d: w.max(1) });
        let _ = writeln!(
            out,
            "{ind}<Pad><Location x=\"{}\" y=\"{}\"/><StandardPrimitiveRef id=\"{prim}\"/></Pad>",
            n(pts[0].x),
            n(pts[0].y)
        );
        return;
    }
    let _ = writeln!(out, "{ind}<Features><Polyline>");
    let _ = writeln!(
        out,
        "{ind}  <PolyBegin x=\"{}\" y=\"{}\"/>",
        n(pts[0].x),
        n(pts[0].y)
    );
    for p in &pts[1..] {
        let _ = writeln!(
            out,
            "{ind}  <PolyStepSegment x=\"{}\" y=\"{}\"/>",
            n(p.x),
            n(p.y)
        );
    }
    let _ = writeln!(
        out,
        "{ind}  <LineDescRef id=\"{lid}\"/>\n{ind}</Polyline></Features>"
    );
}

fn text_lines(out: &mut String, dict: &mut Dict, t: &crate::layout::BoardText, ind: &str) {
    let th = if t.thickness > 0 {
        t.thickness
    } else {
        crate::font::default_thickness(t.size)
    };
    for pl in crate::font::strokes(&t.text, t.at, t.size, t.angle, t.mirrored) {
        polyline(out, dict, &pl, th, ind);
    }
}

fn graphic_lines(
    out: &mut String,
    dict: &mut Dict,
    g: &Graphic,
    t: &crate::geom::Transform,
    ind: &str,
) {
    let tf = |ps: &[Point]| ps.iter().map(|p| t.apply(*p)).collect::<Vec<_>>();
    match g {
        Graphic::Line { points, width } => polyline(out, dict, &tf(points), *width, ind),
        Graphic::Rect { a, b, width, .. } => {
            let c = [*a, Point::new(b.x, a.y), *b, Point::new(a.x, b.y), *a];
            polyline(out, dict, &tf(&c), *width, ind);
        }
        Graphic::Polygon { points, width, .. } => {
            let mut p = points.clone();
            if let Some(f) = p.first().copied() {
                p.push(f);
            }
            polyline(out, dict, &tf(&p), *width, ind);
        }
        Graphic::Circle {
            center,
            radius,
            width,
            ..
        } => {
            let mut r = crate::geom::poly::circle_ring(*center, *radius, crate::geom::mm(0.01));
            r.push(r[0]);
            polyline(out, dict, &tf(&r), *width, ind);
        }
        Graphic::Arc {
            start,
            mid,
            end,
            width,
        } => {
            if let Some((c, sw)) = crate::geom::shape::arc_from_three(*start, *mid, *end) {
                let p = crate::geom::shape::arc_points(c, *start, sw, crate::geom::mm(0.01));
                polyline(out, dict, &tf(&p), *width, ind);
            }
        }
        Graphic::Text {
            at,
            text,
            size,
            angle,
        } => {
            if text.starts_with("${") {
                return;
            }
            let ang = t.apply_angle(*angle) - if t.mirror { 180.0 } else { 0.0 };
            for pl in crate::font::strokes(text, t.apply(*at), *size, ang, t.mirror) {
                polyline(out, dict, &pl, crate::font::default_thickness(*size), ind);
            }
        }
    }
}

/// The name of non-copper stackup layer `k`: the mask, paste and legend layers by side (above
/// the first copper layer is the top), dielectrics as `DIELECTRIC_<k>`.
fn stack_layer_name(board: &Board, k: usize) -> String {
    use crate::layout::StackKind;
    let first_cu = board
        .stackup
        .layers
        .iter()
        .position(|l| matches!(l.kind, StackKind::Copper { .. }))
        .unwrap_or(0);
    let side = if k < first_cu { "TOP" } else { "BOTTOM" };
    match board.stackup.layers[k].kind {
        StackKind::SolderMask => format!("MASK_{side}"),
        StackKind::Paste => format!("PASTE_{side}"),
        StackKind::Silkscreen => format!("SILK_{side}"),
        _ => format!("DIELECTRIC_{k}"),
    }
}
