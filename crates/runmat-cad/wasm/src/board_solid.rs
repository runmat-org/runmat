//! A circuit board as an exact solid, so interference and clearance checks in an assembly
//! treat it as they treat a part: the outline (with its cut-outs and drilled holes) extruded to
//! the board's thickness, joined with each fitted component's body as a prism of its outline and height
//! (`runmat_circuit::board3d::board_solid`). Copper, mask and silkscreen are surface art and
//! add no volume. The solid is in the board's 3D frame, in metres, z = 0 at the bottom face.

use crate::kernel::DesignKernel;
use crate::runtime::Files;
use runmat_cad::assembly::source::Aabb;
use runmat_cad::ids::SketchEntityId;
use runmat_cad::model::datum::Frame;
use runmat_cad::regen::kernel::{
    BodyHandle, CanonicalProfile, Combine, KernelOp, ProfileSegment, SegmentKind, ToolOp, ToolShape,
};

/// A board's exact body, the kernel that made it, and its box (m).
pub(crate) struct BoardBody {
    pub kernel: Box<dyn DesignKernel>,
    pub body: BodyHandle,
    pub bbox: Aabb,
    /// Component bodies the kernel could not join, with its message.
    pub problems: Vec<String>,
}

/// A closed ring (mm) as one profile loop (m) of line segments.
fn ring_loop(ring: &[(f64, f64)], name: &str) -> Vec<ProfileSegment> {
    let id = SketchEntityId::new(name);
    let pts: Vec<[f64; 2]> = ring.iter().map(|&(x, y)| [x * 1e-3, y * 1e-3]).collect();
    (0..pts.len())
        .filter_map(|i| {
            let (a, b) = (pts[i], pts[(i + 1) % pts.len()]);
            (a != b).then(|| ProfileSegment {
                curve: id.clone(),
                kind: SegmentKind::Line,
                start: a,
                end: b,
            })
        })
        .collect()
}

/// A prism of `loops` from height `z0` to `z1` (m), combined with `operands` as `combine` says.
fn prism(
    kernel: &dyn DesignKernel,
    loops: Vec<Vec<ProfileSegment>>,
    z0: f64,
    z1: f64,
    combine: Combine,
    operands: &[&BodyHandle],
) -> Result<BodyHandle, String> {
    let frame = Frame {
        origin: [0.0, 0.0, z0],
        ..Frame::WORLD
    };
    let profile = CanonicalProfile {
        frame,
        loops,
        merge_collinear: false,
    };
    let op = KernelOp::Tool(ToolOp::new(
        ToolShape::prism(profile, 0.0, z1 - z0),
        combine,
    ));
    let out = kernel.execute(&op, operands).map_err(|e| e.to_string())?;
    out.bodies
        .into_iter()
        .next()
        .map(|b| b.handle)
        .ok_or_else(|| "the kernel made no body".into())
}

/// The exact body of the board that `file` (a `.circuit` or its `.circuit.layout`) lays out.
pub(crate) fn board_body(file: &str, files: &Files) -> Result<BoardBody, String> {
    let session = crate::board::compiled_board(file, files)?;
    let solid = session.board_solid().ok_or("no board")?;
    let kernel = crate::kernel::make();
    let outline: Vec<Vec<ProfileSegment>> = solid
        .outline
        .iter()
        .enumerate()
        .map(|(i, r)| ring_loop(r, &format!("outline{i}")))
        .collect();
    let mut loops = outline.clone();
    loops.extend(
        solid
            .holes
            .iter()
            .enumerate()
            .map(|(i, r)| ring_loop(r, &format!("hole{i}"))),
    );
    let thickness = solid.thickness * 1e-3;
    let mut problems = Vec::new();
    // Holes are inner loops of the one outline profile: a single extrusion, no boolean per hole.
    let mut body = match prism(kernel.as_ref(), loops, 0.0, thickness, Combine::New, &[]) {
        Ok(b) => b,
        Err(e) if !solid.holes.is_empty() => {
            problems.push(format!(
                "the board's {} drilled holes don't cut: {e}",
                solid.holes.len()
            ));
            prism(kernel.as_ref(), outline, 0.0, thickness, Combine::New, &[])
                .map_err(|e| format!("the board outline doesn't extrude: {e}"))?
        }
        Err(e) => return Err(format!("the board outline doesn't extrude: {e}")),
    };
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    let mut grow = |x: f64, y: f64, z: f64| {
        for (k, v) in [x, y, z].into_iter().enumerate() {
            min[k] = min[k].min(v);
            max[k] = max[k].max(v);
        }
    };
    for &(x, y) in &solid.outline[0] {
        grow(x * 1e-3, y * 1e-3, 0.0);
        grow(x * 1e-3, y * 1e-3, thickness);
    }
    for b in &solid.bodies {
        // From the board face it sits on, so the body joins the board as one solid.
        let (z0, z1) = if b.z0 >= solid.thickness {
            (thickness, b.z1 * 1e-3)
        } else {
            (b.z0 * 1e-3, 0.0)
        };
        match prism(
            kernel.as_ref(),
            vec![ring_loop(&b.ring, &b.reference)],
            z0,
            z1,
            Combine::Add,
            &[&body],
        ) {
            Ok(joined) => {
                body = joined;
                for &(x, y) in &b.ring {
                    grow(x * 1e-3, y * 1e-3, z0);
                    grow(x * 1e-3, y * 1e-3, z1);
                }
            }
            Err(e) => problems.push(format!("{}: {e}", b.reference)),
        }
    }
    Ok(BoardBody {
        kernel,
        body,
        bbox: Aabb { min, max },
        problems,
    })
}
