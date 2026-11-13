//! The board as exact solids' input, in the 3D view's frame (mm, z = 0 at the board's bottom
//! face): the substrate's outline and drilled holes, each fitted component's body (its package's
//! body box, and the placement of its package's 3D model when it names one), the solder mask's
//! extent and the outer copper the 3D tab draws. Assemblies read [`board_solid`] for interference
//! and clearance; the board's STEP export (runmat-cad-wasm `exchange::board`) writes all of it.

use super::{body_extent, ring_mm, signed_area, thickness_mm, MASK_T, TOL};
use crate::geom::shape::Prim;
use crate::geom::to_mm;
use crate::item_ref::ItemRef;
use crate::layout::{Board, Footprint, Side, StackKind};

/// A component body of [`BoardSolid`]: its outline on the board and the heights it spans (mm).
#[derive(Clone, Debug, PartialEq)]
pub struct SolidBody {
    /// `component:<designator>`.
    pub reference: String,
    /// The reference designator (`U1`).
    pub designator: String,
    pub ring: Vec<(f64, f64)>,
    pub z0: f64,
    pub z1: f64,
    /// The body's colour in the 3D tab (linear RGBA).
    pub color: [f64; 4],
    /// Where the package's 3D model goes, when the package names one.
    pub model: Option<PlacedModel>,
}

/// A package's 3D model placed on the board: `p ↦ rotation · (scale ∘ p) + translation`, with the
/// model's points in millimetres. The model's own rotation (degrees about x, then y, then z), its
/// offset and scale come from the package (`model = { path, offset, rotation, scale }`); then the
/// footprint's rotation about z and, on the bottom side, a half turn about the board's y axis, so
/// the model hangs below the board as the footprint is mirrored. The model's z = 0 sits on the
/// solder mask, as a body box does.
#[derive(Clone, Debug, PartialEq)]
pub struct PlacedModel {
    /// As the package writes it (relative to the file that declares the package).
    pub path: String,
    pub rotation: [[f64; 3]; 3],
    pub translation: [f64; 3],
    pub scale: [f64; 3],
}

/// The board as an exact solid's input, in the 3D view's frame (mm, z = 0 at the bottom face):
/// the outline (outer ring, then its cut-outs) extruded to the board's thickness, less its
/// drilled holes, and each fitted component's body as a prism. Interference and clearance checks
/// in an assembly read the board as this solid, so a screw through a mounting hole clears it.
#[derive(Clone, Debug, PartialEq)]
pub struct BoardSolid {
    pub outline: Vec<Vec<(f64, f64)>>,
    /// Each footprint's drilled hole through the board, plated or not: a circle at its drill
    /// diameter, or a slot's stadium. Vias are left in (they are filled or tented, and nothing
    /// passes through them). A hole that breaks the board edge, meets a cut-out or overlaps
    /// another hole is left out, so every ring is a separate inner loop of the outline.
    pub holes: Vec<Vec<(f64, f64)>>,
    pub thickness: f64,
    pub bodies: Vec<SolidBody>,
}

/// The rings of `board`'s drilled footprint holes that lie wholly inside its outline, apart from
/// its cut-outs and from each other.
fn drilled_holes(board: &Board) -> Vec<Vec<(f64, f64)>> {
    let mut kept: Vec<Prim> = Vec::new();
    let mut rings = Vec::new();
    for (_, h) in crate::fab::drill::footprint_holes(board) {
        let r = h.dia / 2;
        let (prim, ring) = match h.slot {
            Some((a, b)) => (
                Prim::segment(a, b, h.dia),
                crate::geom::poly::capsule_ring(a, b, r, TOL),
            ),
            None => (
                Prim::circle(h.at, r),
                crate::geom::poly::circle_ring(h.at, r, TOL),
            ),
        };
        let inside = ring
            .iter()
            .all(|p| board.outline.contains(*p) == Some(true));
        if !inside || kept.iter().any(|k| k.gap(&prim) <= 0.0) {
            continue;
        }
        kept.push(prim);
        rings.push(ring_mm(&ring));
    }
    rings
}

type Matrix = [[f64; 3]; 3];

fn mul(a: &Matrix, b: &Matrix) -> Matrix {
    let mut out = [[0.0; 3]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    out
}

fn apply(m: &Matrix, v: [f64; 3]) -> [f64; 3] {
    [0, 1, 2].map(|i| m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2])
}

/// A right-handed rotation by `deg` about axis 0 (x), 1 (y) or 2 (z).
fn turn(axis: usize, deg: f64) -> Matrix {
    let (s, c) = deg.to_radians().sin_cos();
    let (i, j) = match axis {
        0 => (1, 2),
        1 => (2, 0),
        _ => (0, 1),
    };
    let mut m = [[0.0; 3]; 3];
    m[axis][axis] = 1.0;
    m[i][i] = c;
    m[j][j] = c;
    m[i][j] = -s;
    m[j][i] = s;
    m
}

/// Where `f`'s package model goes on a board `top` thick (mm), or `None` without one.
pub fn placed_model(f: &Footprint, top: f64) -> Option<PlacedModel> {
    let model = f.package.model.as_ref()?;
    let (rx, ry, rz) = model.rotation;
    let own = mul(&turn(2, rz), &mul(&turn(1, ry), &turn(0, rx)));
    let side = match f.side {
        Side::Top => turn(2, f.rotation),
        // A half turn about y mirrors x as the footprint is mirrored, and turns the model down.
        Side::Bottom => mul(&turn(2, f.rotation), &turn(1, 180.0)),
    };
    let at = (to_mm(f.at.x), to_mm(f.at.y));
    let base = match f.side {
        Side::Top => [at.0, at.1, top + MASK_T],
        Side::Bottom => [at.0, at.1, -MASK_T],
    };
    let (ox, oy, oz) = model.offset;
    let offset = apply(&side, [ox, oy, oz]);
    let (sx, sy, sz) = model.scale;
    Some(PlacedModel {
        path: model.path.clone(),
        rotation: mul(&side, &own),
        translation: [0, 1, 2].map(|k| base[k] + offset[k]),
        scale: [sx, sy, sz],
    })
}

pub fn board_solid(board: &Board) -> BoardSolid {
    let thickness = thickness_mm(board);
    let n = board.copper_count();
    let mut outline = vec![ring_mm(&board.outline.outer)];
    outline.extend(board.outline.holes.iter().map(|r| ring_mm(r)));
    let bodies = board
        .footprints
        .values()
        .filter(|f| f.fitted)
        .filter_map(|f| {
            let model = placed_model(f, thickness);
            let (ring, z0, z1) = body_extent(f, n, thickness)?;
            Some(SolidBody {
                reference: ItemRef::Component(f.reference.clone()).to_string(),
                designator: f.reference.clone(),
                ring,
                z0,
                z1,
                color: super::body_material(f).1,
                model,
            })
        })
        .collect();
    BoardSolid {
        outline,
        holes: drilled_holes(board),
        thickness,
        bodies,
    }
}

/// One outer copper layer as thin solids: each item (a track, a via's ring, a pad) is a prism of
/// its loops (the outer ring, then its holes), from `z0` to `z1` (mm).
#[derive(Clone, Debug, PartialEq)]
pub struct CopperSolid {
    /// The stack-up's name for the layer (`F.Cu`).
    pub layer: String,
    pub z0: f64,
    pub z1: f64,
    pub items: Vec<Vec<Vec<(f64, f64)>>>,
}

/// Outer copper solids and the solder-mask height interval on each board face (mm).
pub type BoardSurfaces = (Vec<CopperSolid>, [(f64, f64); 2]);

/// A primitive's outline as items: each outer ring with the holes that follow it.
fn items_of(prim: &Prim, hole: Option<Vec<(f64, f64)>>) -> Vec<Vec<Vec<(f64, f64)>>> {
    let mut items: Vec<Vec<Vec<(f64, f64)>>> = Vec::new();
    for ring in prim.outline(0, TOL) {
        let ring = ring_mm(&ring);
        if ring.len() < 3 {
            continue;
        }
        match items.last_mut() {
            Some(item) if signed_area(&ring) < 0.0 => item.push(ring),
            _ => items.push(vec![ring]),
        }
    }
    if let (Some(hole), [item]) = (hole, items.as_mut_slice()) {
        item.push(hole);
    }
    items
}

/// The outer copper the 3D tab draws (tracks, via rings and pads on the top and bottom layers),
/// standing on each face at the layer's thickness from the stack-up. Inner layers sit inside the
/// substrate and are left out, as the 3D tab leaves them out.
pub fn outer_copper(board: &Board) -> Vec<CopperSolid> {
    let n = board.copper_count();
    let top = thickness_mm(board);
    let names = board.stackup.copper_names();
    let layer = |i: crate::layer_set::CopperIndex| {
        let name = names
            .get(i as usize)
            .cloned()
            .unwrap_or_else(|| crate::layout::copper_name(i, n));
        let t = board
            .stackup
            .copper_index(i)
            .map(|k| &board.stackup.layers[k])
            .filter(|l| matches!(l.kind, StackKind::Copper { .. }))
            .map(|l| to_mm(l.thickness))
            .filter(|t| *t > 0.0)
            .unwrap_or(0.035);
        let (z0, z1) = if i == 0 { (top, top + t) } else { (-t, 0.0) };
        CopperSolid {
            layer: name,
            z0,
            z1,
            items: Vec::new(),
        }
    };
    let mut out = vec![layer(0)];
    if n > 1 {
        out.push(layer(n - 1));
    }
    let slot = |i: crate::layer_set::CopperIndex| {
        if i == 0 {
            Some(0)
        } else if i + 1 == n && n > 1 {
            Some(1)
        } else {
            None
        }
    };
    let circle = |c, d| ring_mm(&crate::geom::poly::circle_ring(c, d / 2, TOL));
    for t in board.tracks.values() {
        let Some(k) = slot(t.layer) else { continue };
        for prim in t.prims() {
            out[k].items.extend(items_of(&prim, None));
        }
    }
    for v in board.vias.values() {
        for k in v.layers().filter_map(slot) {
            let ring = Prim::circle(v.at, v.diameter / 2);
            out[k]
                .items
                .extend(items_of(&ring, Some(circle(v.at, v.drill))));
        }
    }
    for f in board.footprints.values() {
        for p in f.pads(n) {
            for &i in &p.layers {
                let Some(k) = slot(i) else { continue };
                let hole = p.drill.map(|d| circle(p.center, d));
                out[k].items.extend(items_of(&p.prim, hole));
            }
        }
    }
    out
}

/// The solder mask's extent on each face: the board outline with its cut-outs and drilled holes,
/// `MASK_T` thick on the top face and under the bottom face (mm). Its pad openings are left out.
pub fn mask_extent(board: &Board) -> [(f64, f64); 2] {
    let top = thickness_mm(board);
    [(top, top + MASK_T), (-MASK_T, 0.0)]
}

/// The 3D model files a circuit file's packages name, as written (relative to that file): a
/// layout's packages, and a `.circuit`'s footprints and parts. Paths through an environment
/// variable (`${…}/x.step`) name a model library outside the project and are left out. The CAD
/// host gathers these files for a board's STEP and its assemblies.
pub fn model_paths(path: &str, text: &str) -> Vec<String> {
    let lower = path.to_ascii_lowercase();
    let mut out: Vec<String> = Vec::new();
    if lower.ends_with(".circuit.layout") {
        let parsed = crate::syntax::board::parse_board(path, text);
        out.extend(
            parsed
                .packages
                .values()
                .filter_map(|p| p.model.as_ref().map(|m| m.path.clone())),
        );
    } else if lower.ends_with(".circuit") {
        if let Some(design) = crate::syntax::parse_circuit(path, text).design {
            out.extend(
                design
                    .footprints
                    .values()
                    .filter_map(|p| p.model.as_ref().map(|m| m.path.clone())),
            );
            out.extend(design.parts.values().filter_map(|p| p.body.clone()));
        }
    }
    out.retain(|p| !p.starts_with('$') && !p.is_empty());
    out.sort();
    out.dedup();
    out
}
