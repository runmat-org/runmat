//! Dragging a component through the mechanism's remaining degrees of freedom (`motion.drag`, and
//! the Free drag mode of Move component).
//!
//! The drag is the position problem of computational kinematics with a soft goal (Haug, 1989,
//! ch. 3): the mates stay hard constraints `Φ(q) = 0`, and the point of the component under the
//! pointer is pulled toward the pointer's target with a small weight (`runmat-constraint`'s
//! `drag`, through a free handle ball-jointed to that point). The solve keeps every mate and leaves
//! the point as near the target as they allow. A hinged lid dragged upward by its free edge swings
//! about its hinge; a slider follows the drag's component along its axis; a component with no
//! freedom left stays put. Every component the drag moves (the dragged one and those mated to it)
//! keeps its new pose as its placement, the start the mate solve
//! begins from next time.

use super::kinematics::{frame_failure, warm_placement, MotionContext};
use crate::assembly::model::{AssemblyDocument, Placement};
use crate::assembly::resolve::{resolve, DragTarget, NodeId, ResolveRequest, ResolvedAssembly};
use crate::regen::kernel::Transform;

/// A drag's outcome.
pub struct Dragged {
    /// The assembly as the drag leaves it.
    pub tree: ResolvedAssembly,
    /// The new placement of each top-level component the drag moved, by name.
    pub placements: Vec<(String, Placement)>,
}

/// The top-level node of occurrence `name`.
pub fn top_node(base: &ResolvedAssembly, name: &str) -> Option<NodeId> {
    base.node(0)
        .children
        .iter()
        .copied()
        .find(|&c| base.node(c).segment.name == name)
}

/// Drags top-level component `occurrence` of `doc` (the top assembly of `base`) by `delta`
/// (metres, top assembly frame): its point `grab` (its own frame, metres) is pulled to where it
/// is plus `delta`.
pub fn drag(
    doc: &AssemblyDocument,
    base: &ResolvedAssembly,
    occurrence: &str,
    grab: [f64; 3],
    delta: [f64; 3],
    ctx: &MotionContext<'_>,
) -> Result<Dragged, String> {
    let o = doc
        .occurrences
        .iter()
        .find(|o| o.name == occurrence)
        .ok_or_else(|| format!("the assembly has no component `{occurrence}`"))?;
    if o.fixed {
        return Err(format!("`{occurrence}` is fixed; float it to drag it"));
    }
    if o.suppressed {
        return Err(format!("`{occurrence}` is suppressed"));
    }
    let node = top_node(base, occurrence)
        .ok_or_else(|| format!("`{occurrence}` isn't placed in the assembly"))?;
    let from = base.node(node).world.apply(grab);
    let request = ResolveRequest {
        drag: Some(DragTarget {
            occurrence: occurrence.to_owned(),
            grab,
            to: std::array::from_fn(|i| from[i] + delta[i]),
        }),
        ..ctx.request.clone()
    };
    let tree = resolve(&base.top, &request, ctx.resolve);
    let same = tree.nodes.len() == base.nodes.len()
        && tree
            .nodes
            .iter()
            .zip(&base.nodes)
            .all(|(a, b)| a.uuid == b.uuid);
    if !same {
        return Err("the assembly's structure changed during the drag".into());
    }
    if let Some(why) = frame_failure(&tree, base) {
        return Err(format!("the drag can't be solved: {why}"));
    }
    let mut placements = Vec::new();
    for o in doc.occurrences.iter().filter(|o| !o.fixed && !o.suppressed) {
        if matches!(o.placement, Placement::Attach { .. }) {
            continue;
        }
        let Some(n) = top_node(&tree, &o.name) else {
            continue;
        };
        let local = &tree.node(n).local;
        if local.max_difference(&base.node(n).local) > 1e-9 {
            placements.push((o.name.clone(), warm_placement(&rounded(local))));
        }
    }
    Ok(Dragged { tree, placements })
}

/// A pose with its translation to 0.1 µm and its rotation to 9 places: a floating component's
/// placement is only where the mate solve starts, so the rounding moves nothing.
fn rounded(t: &Transform) -> Transform {
    let r = |v: f64, places: i32| {
        let f = 10f64.powi(places);
        let x = (v * f).round() / f;
        if x == 0.0 {
            0.0
        } else {
            x
        }
    };
    let mut out = *t;
    out.translation = t.translation.map(|v| r(v, 7));
    out.rotation = t.rotation.map(|row| row.map(|v| r(v, 9)));
    out
}
