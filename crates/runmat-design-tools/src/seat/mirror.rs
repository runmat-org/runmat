//! `mirror`: seats a component as the mirror image of one already seated, across the target's
//! plane of symmetry (the plane through the middle of the target's box, square to the assembly's
//! Y axis unless `mirrorNormal` says otherwise), fastened to the target. A left leg's mirror is
//! the right leg, so the two sides of a landing gear are symmetric by construction.
//!
//! A placement can only turn and move a part, never reflect it, so the part must be symmetric
//! about its own mid-plane across its bore (an extruded profile is, about the middle of its
//! extrusion). Reflected about that plane and then across the target's, it is turned and moved
//! by an ordinary placement. The part's symmetry is checked on its tessellation.

use super::geometry::*;
use super::{Frame, Placed, PlacedSet, Seated};
use crate::ToolError;

/// A point of the part with no mirror within this distance (m) means it isn't symmetric.
const SYMMETRY_TOLERANCE: f64 = 2e-4;
/// The points checked.
const CHECKED: usize = 400;

/// The reflection of `p` across the plane through `at` square to unit `n`.
fn reflect(p: V3, at: V3, n: V3) -> V3 {
    sub(p, scale(n, 2.0 * dot(sub(p, at), n)))
}

/// The mirrored seating: the frames on the target and on the component, and its pose.
pub(super) struct Mirrored {
    pub(super) target_frame: Frame,
    pub(super) own_frame: Frame,
    pub(super) seated: Seated,
}

pub(super) fn mirror(set: &PlacedSet, moving: &Placed, target: &Placed, name: &str, of: &str, normal: Option<V3>, notes: &mut Vec<String>) -> Result<Mirrored, ToolError> {
    if of == name {
        return Err(ToolError::invalid(format!("component `{name}` can't mirror itself; name the component seated on the other side")));
    }
    let source = set.get(of, "mirror")?;
    let same = match (bounds(&source.display), bounds(&moving.display)) {
        (Some((a, b)), Some((c, d))) => norm(sub(a, c)) < 1e-6 && norm(sub(b, d)) < 1e-6,
        _ => false,
    };
    if !same {
        return Err(ToolError::invalid(format!("`{name}` isn't the same part as `{of}`; mirror seats a copy of `{of}`'s part as its mirror image")));
    }
    let n = unit(normal.unwrap_or([0.0, 1.0, 0.0])).ok_or_else(|| ToolError::invalid("mirrorNormal has no length"))?;
    let (tlo, thi) = bounds(&target.display).ok_or_else(|| ToolError::invalid("the target has no body to mirror across"))?;
    let plane = target.world.apply(scale(add(tlo, thi), 0.5));

    // The part's own mirror plane: square to its bore, through the middle of its extent there.
    let a = rounds(&source.display)
        .into_iter()
        .next()
        .map(|r| r.axis)
        .ok_or_else(|| ToolError::invalid(format!("`{of}` has no round face (a bore); mirror turns a part extruded along its bore, such as a leg, to its mirror image")))?;
    let pts = points(&source.display);
    let (lo, hi) = pts.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), p| (l.min(dot(*p, a)), h.max(dot(*p, a))));
    let own_mid = scale(a, (lo + hi) / 2.0);
    let step = pts.len().div_ceil(CHECKED).max(1);
    for p in pts.iter().step_by(step) {
        let q = reflect(*p, own_mid, a);
        let nearest = pts.iter().map(|r| norm(sub(*r, q))).fold(f64::INFINITY, f64::min);
        if nearest > SYMMETRY_TOLERANCE {
            return Err(ToolError::invalid(format!(
                "`{of}` isn't symmetric about its mid-plane across its bore (its point at [{}] mm has no mirror image), so no placement makes its mirror image; seat `{name}` with its own target instead",
                p.map(|v| format!("{:.2}", v * 1000.0)).join(", ")
            )));
        }
    }

    // p ↦ across the target's plane (source placement (own reflection p)): two reflections, so a
    // turn and a move.
    let place = |p: V3| reflect(source.world.apply(reflect(p, own_mid, a)), plane, n);
    let origin = place([0.0; 3]);
    let columns: Vec<V3> = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]].into_iter().map(|e| sub(place(e), origin)).collect();
    let rotation: M3 = [0, 1, 2].map(|i| [columns[0][i], columns[1][i], columns[2][i]]);
    let seated = Seated { rotation, translation: origin };

    // The fasten's frames: the part's box centre, and where the mirror image puts it on the target.
    let (blo, bhi) = bounds(&moving.display).unwrap_or(([0.0; 3], [0.0; 3]));
    let at = scale(add(blo, bhi), 0.5);
    let x = perpendicular(a);
    let inv = target.world.inverse();
    let target_frame = Frame { at: inv.apply(seated.apply(at)), z: inv.apply_vector(apply(&rotation, a)), x: inv.apply_vector(apply(&rotation, x)) };
    notes.push(format!(
        "the mirror image of `{of}` across the plane through the middle of the target's box at [{}] mm, square to [{}]",
        plane.map(|v| format!("{:.2}", v * 1000.0)).join(", "),
        n.map(|v| format!("{v:.3}")).join(", ")
    ));
    Ok(Mirrored { target_frame, own_frame: Frame { at, z: a, x }, seated })
}
