//! Bodies that fall apart: a solid body whose faces form more than one connected shell that
//! isn't a cavity is several separate pieces under one name. The check names the body and,
//! for each piece, the features that made its faces.

use super::*;

/// One connected set of a body's faces: the faces, by index, and their bounding box.
struct Piece {
    faces: Vec<usize>,
    min: [f64; 3],
    max: [f64; 3],
}

/// The connected face sets of `topology`, joined through shared edges.
fn pieces(topology: &Topology) -> Vec<Piece> {
    let n = topology.faces.len();
    let mut parent: Vec<usize> = (0..n).collect();
    fn root(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    for edge in &topology.edges {
        let (a, b) = (edge.faces[0] as usize, edge.faces[1] as usize);
        if a >= n || b >= n {
            continue;
        }
        let (ra, rb) = (root(&mut parent, a), root(&mut parent, b));
        if ra != rb {
            parent[ra] = rb;
        }
    }
    let mut by_root: BTreeMap<usize, Piece> = BTreeMap::new();
    for face in 0..n {
        let r = root(&mut parent, face);
        let geom = &topology.faces[face].geom;
        let piece = by_root.entry(r).or_insert_with(|| Piece {
            faces: Vec::new(),
            min: [f64::INFINITY; 3],
            max: [f64::NEG_INFINITY; 3],
        });
        piece.faces.push(face);
        for k in 0..3 {
            piece.min[k] = piece.min[k].min(geom.bbox_min[k]);
            piece.max[k] = piece.max[k].max(geom.bbox_max[k]);
        }
    }
    by_root.into_values().collect()
}

/// A piece whose box lies inside another piece's box is the wall of a cavity in it, not a
/// separate piece.
fn inside(inner: &Piece, outer: &Piece) -> bool {
    let tol = 1e-9;
    (0..3).all(|k| inner.min[k] > outer.min[k] - tol && inner.max[k] < outer.max[k] + tol)
        && (0..3).any(|k| inner.min[k] > outer.min[k] + tol || inner.max[k] < outer.max[k] - tol)
}

/// The feature a face or body name comes from.
fn feature_of(name: &EntityName) -> Option<&FeatureId> {
    match name {
        EntityName::Generated(g) => Some(&g.feature),
        EntityName::Copy { pattern, .. } => Some(pattern),
        EntityName::CadFile { import, .. } => Some(import),
    }
}

/// The separate pieces of each solid body in `state`, as (the body's feature, one list of
/// feature names per piece). Bodies in one piece, and surface bodies, are left out.
fn split_bodies(state: &PartState) -> Vec<(FeatureId, Vec<Vec<String>>)> {
    let mut out = Vec::new();
    for body in &state.bodies {
        if body.topology.edges.iter().any(|e| e.laminar) {
            continue;
        }
        let all = pieces(&body.topology);
        if all.len() < 2 {
            continue;
        }
        let separate: Vec<&Piece> = all
            .iter()
            .filter(|p| !all.iter().any(|q| !std::ptr::eq(*p, q) && inside(p, q)))
            .collect();
        if separate.len() < 2 {
            continue;
        }
        let Some(owner) = body.names.body.iter().find_map(feature_of).cloned() else {
            continue;
        };
        let named: Vec<Vec<String>> = separate
            .iter()
            .map(|p| {
                let features: BTreeSet<String> = p
                    .faces
                    .iter()
                    .filter_map(|f| body.names.faces.get(*f))
                    .flat_map(|names| names.iter().filter_map(feature_of))
                    .map(|f| f.0.clone())
                    .collect();
                features.into_iter().collect()
            })
            .collect();
        out.push((owner, named));
    }
    out
}

/// A warning for each solid body that is several separate pieces, naming each piece by the
/// features that made it.
pub(super) fn report_split_bodies(state: &PartState, diags: &mut DiagnosticReport) {
    for (owner, named) in split_bodies(state) {
        let list: Vec<String> = named
            .iter()
            .enumerate()
            .map(|(i, features)| {
                let made: Vec<String> = features.iter().map(|f| format!("`{f}`")).collect();
                format!("piece {} from {}", i + 1, made.join(", "))
            })
            .collect();
        diags.push(
            DiagCode::BodyDisconnected,
            Severity::Warning,
            Site::feature(&owner, None),
            format!(
                "the body made by `{owner}` is {} separate pieces that don't touch: {}; overlap them or add material between them so the body is one solid, or make each its own body",
                named.len(),
                list.join("; ")
            ),
            Cause::Root,
            Vec::new(),
        );
    }
}

/// The features that made a body's faces, by name.
fn body_features(names: &BodyNames) -> Vec<String> {
    let features: BTreeSet<String> = names
        .faces
        .iter()
        .flat_map(|names| names.iter().filter_map(feature_of))
        .map(|f| f.0.clone())
        .collect();
    features.into_iter().collect()
}

/// A warning when feature `id` adds material that touches none of the bodies it joins: the
/// operation then returns more solids than it was given (`operands` in, `outputs` out), and
/// the part falls apart into pieces. Each piece is named by the features that made it.
pub(super) fn report_detached_add(
    id: &FeatureId,
    op: &KernelOp,
    operands: usize,
    outputs: &[BodyNames],
    diags: &mut DiagnosticReport,
) {
    let adds = match op {
        KernelOp::Tool(t) => t.combine == Combine::Add,
        KernelOp::Boolean {
            combine,
            keep_tools,
            ..
        } => *combine == Combine::Add && !keep_tools,
        _ => false,
    };
    if !adds || operands == 0 || outputs.len() <= operands {
        return;
    }
    let list: Vec<String> = outputs
        .iter()
        .enumerate()
        .map(|(i, names)| {
            let made: Vec<String> = body_features(names)
                .iter()
                .map(|f| format!("`{f}`"))
                .collect();
            format!("piece {} from {}", i + 1, made.join(", "))
        })
        .collect();
    diags.push(
        DiagCode::BodyDisconnected,
        Severity::Warning,
        Site::feature(id, None),
        format!(
            "`{id}` adds material that doesn't touch the body it joins, so the part is {} separate solids: {}; overlap the new material with the body, or make it its own body",
            outputs.len(),
            list.join("; ")
        ),
        Cause::Root,
        Vec::new(),
    );
}
