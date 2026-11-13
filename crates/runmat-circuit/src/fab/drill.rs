//! Excellon drill files and the drill map.

use crate::geom::{to_mm, Coord, Point};
use crate::layer_set::CopperIndex;
use crate::layout::Board;
use std::collections::BTreeMap;
use std::fmt::Write as _;

/// One drilled hole: a round hole at `at`, or a slot from `slot.0` to `slot.1`.
#[derive(Clone, Debug, PartialEq)]
pub struct Hole {
    pub at: Point,
    pub dia: Coord,
    pub slot: Option<(Point, Point)>,
}

/// A drill span: the copper layers a hole runs between (0-based, top first) and whether it is
/// plated. Non-plated holes always run through.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Span {
    pub from: CopperIndex,
    pub to: CopperIndex,
    pub plated: bool,
}

impl Span {
    /// `PTH`, `Blind`, `Buried` or `NPTH`, as the drill file's function names it.
    pub fn kind(&self, copper_count: CopperIndex) -> &'static str {
        if !self.plated {
            "NPTH"
        } else if self.from == 0 && self.to + 1 == copper_count {
            "PTH"
        } else if self.from == 0 || self.to + 1 == copper_count {
            "Blind"
        } else {
            "Buried"
        }
    }
}

/// Every footprint's drilled hole (round or slotted) with whether it is plated. Footprint holes
/// run through the board; vias are not included.
pub fn footprint_holes(board: &Board) -> Vec<(bool, Hole)> {
    let n = board.copper_count();
    let mut out = Vec::new();
    for f in board.footprints.values() {
        let t = f.transform();
        for (pd, pp) in f.package.pads.iter().zip(f.pads(n)) {
            let Some(d) = &pd.drill else { continue };
            let slot = d.slot.map(|len| {
                let half = (len - d.diameter).max(0) / 2;
                let local = crate::geom::Transform::new(pd.at, pd.rotation, false);
                let tt = t.compose(&local);
                (
                    tt.apply(Point::new(-half, 0)),
                    tt.apply(Point::new(half, 0)),
                )
            });
            out.push((
                d.plated,
                Hole {
                    at: pp.center,
                    dia: d.diameter,
                    slot,
                },
            ));
        }
    }
    out
}

/// Every hole of the board by span, sorted by diameter then position.
pub fn holes(board: &Board) -> BTreeMap<Span, Vec<Hole>> {
    let n = board.copper_count();
    let mut spans: BTreeMap<Span, Vec<Hole>> = BTreeMap::new();
    for v in board.vias.values() {
        let span = Span {
            from: v.from,
            to: v.to,
            plated: true,
        };
        spans.entry(span).or_default().push(Hole {
            at: v.at,
            dia: v.drill,
            slot: None,
        });
    }
    for (plated, hole) in footprint_holes(board) {
        let span = Span {
            from: 0,
            to: n - 1,
            plated,
        };
        spans.entry(span).or_default().push(hole);
    }
    for hs in spans.values_mut() {
        hs.sort_by_key(|h| (h.dia, h.at));
    }
    spans
}

/// One row of the drill table: a tool of one span.
#[derive(Clone, Debug, PartialEq)]
pub struct DrillRow {
    pub span: Span,
    pub dia: Coord,
    /// Holes drilled with this tool, slots included.
    pub count: usize,
    pub slots: usize,
}

/// The drill table: one row per span and diameter, spans in file order, then by diameter.
pub fn drill_table(board: &Board) -> Vec<DrillRow> {
    let mut rows = Vec::new();
    for (span, hs) in holes(board) {
        let mut tools: Vec<Coord> = hs.iter().map(|h| h.dia).collect();
        tools.dedup();
        for dia in tools {
            let of: Vec<&Hole> = hs.iter().filter(|h| h.dia == dia).collect();
            rows.push(DrillRow {
                span,
                dia,
                count: of.len(),
                slots: of.iter().filter(|h| h.slot.is_some()).count(),
            });
        }
    }
    rows
}

/// Excellon drill files: one per plated span (`PTH`, then blind/buried spans), one for
/// non-plated holes and one per back-drill face and depth layer ([`backdrill_files`]), keyed by
/// file name.
pub fn excellon(board: &Board) -> BTreeMap<String, String> {
    let mut out = plated_and_npth(board);
    out.extend(backdrill_files(board));
    out
}

/// One Excellon file per back-drill face and must-not-cut layer: non-plated, controlled-depth
/// holes (`BackDrill`), each tool's depth from that face in a comment, as the fabrication
/// drawing's note states it.
pub fn backdrill_files(board: &Board) -> BTreeMap<String, String> {
    use crate::high_speed::stubs::{board_backdrills, cut_layers};
    let n = board.copper_count();
    let names = board.stackup.copper_names();
    let mut groups: BTreeMap<_, Vec<_>> = BTreeMap::new();
    for (at, b) in board_backdrills(board) {
        groups
            .entry((b.face, b.must_not_cut))
            .or_default()
            .push((at, b));
    }
    let mut out = BTreeMap::new();
    for ((face, keep), holes) in groups {
        let (a, z) = cut_layers(&holes[0].1, n);
        let mut tools: Vec<(Coord, Coord)> =
            holes.iter().map(|(_, b)| (b.diameter, b.depth)).collect();
        tools.sort();
        tools.dedup();
        let mut s = String::new();
        let _ = writeln!(
            s,
            "M48\n; DRILL file {{RunMat runmat-circuit {}}}",
            env!("CARGO_PKG_VERSION")
        );
        let _ = writeln!(s, "; FORMAT={{-:-/ absolute / metric / decimal}}");
        let _ = writeln!(s, "; #@! TF.FileFunction,NonPlated,{a},{z},Blind");
        let _ = writeln!(
            s,
            "; back-drill from the {} face; must not cut layer {} ({})",
            face.word(),
            keep + 1,
            names.get(keep as usize).cloned().unwrap_or_default()
        );
        let _ = writeln!(s, "FMAT,2\nMETRIC\nG90");
        for (i, (dia, depth)) in tools.iter().enumerate() {
            let _ = writeln!(s, "; #@! TA.AperFunction,NonPlated,Blind,BackDrill\n; T{} depth {:.3} mm from the {} face\nT{}C{:.3}", i + 1, to_mm(*depth), face.word(), i + 1, to_mm(*dia));
        }
        s.push_str("%\nG05\n");
        for (i, tool) in tools.iter().enumerate() {
            let _ = writeln!(s, "T{}", i + 1);
            for (at, _) in holes.iter().filter(|(_, b)| (b.diameter, b.depth) == *tool) {
                let _ = writeln!(s, "X{:.4}Y{:.4}", to_mm(at.x), to_mm(at.y));
            }
        }
        s.push_str("M30\n");
        out.insert(
            format!("{}-Backdrill-{}-{a}-{z}.drl", board.name, face.word()),
            s,
        );
    }
    out
}

fn plated_and_npth(board: &Board) -> BTreeMap<String, String> {
    let n = board.copper_count();
    let spans = holes(board);
    let mut out = BTreeMap::new();
    for (span, holes) in spans {
        let Span { from, to, plated } = span;
        let mut tools: Vec<Coord> = holes.iter().map(|h| h.dia).collect();
        tools.dedup();
        let kind = span.kind(n);
        let mut s = String::new();
        let _ = writeln!(
            s,
            "M48\n; DRILL file {{RunMat runmat-circuit {}}}",
            env!("CARGO_PKG_VERSION")
        );
        let _ = writeln!(s, "; FORMAT={{-:-/ absolute / metric / decimal}}");
        let _ = writeln!(
            s,
            "; #@! TF.FileFunction,{},{},{},{kind}",
            if plated { "Plated" } else { "NonPlated" },
            from + 1,
            to + 1
        );
        let _ = writeln!(s, "FMAT,2\nMETRIC\nG90");
        for (i, d) in tools.iter().enumerate() {
            let _ = writeln!(
                s,
                "; #@! TA.AperFunction,{},{}\nT{}C{:.3}",
                if plated { "Plated" } else { "NonPlated" },
                kind,
                i + 1,
                to_mm(*d)
            );
        }
        s.push_str("%\nG05\n");
        for (i, d) in tools.iter().enumerate() {
            let _ = writeln!(s, "T{}", i + 1);
            for h in holes.iter().filter(|h| h.dia == *d) {
                match h.slot {
                    Some((a, b)) => {
                        let _ = writeln!(
                            s,
                            "X{:.4}Y{:.4}G85X{:.4}Y{:.4}",
                            to_mm(a.x),
                            to_mm(a.y),
                            to_mm(b.x),
                            to_mm(b.y)
                        );
                    }
                    None => {
                        let _ = writeln!(s, "X{:.4}Y{:.4}", to_mm(h.at.x), to_mm(h.at.y));
                    }
                }
            }
        }
        s.push_str("M30\n");
        let name = if plated {
            format!("{}-{}-{}-{}.drl", board.name, kind, from + 1, to + 1)
        } else {
            format!("{}-NPTH.drl", board.name)
        };
        out.insert(name, s);
    }
    out
}
