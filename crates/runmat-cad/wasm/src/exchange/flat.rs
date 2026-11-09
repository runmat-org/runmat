//! A part's flat pattern as cut: the flat pattern feature's layout (frame, bend lines and
//! the developed outline) with the outline taken from the flat body itself, seen from the
//! layout's +z side through the kernel's exact projection. Cuts, holes and notches that
//! other features made in the flats are on the flat body (see the plate flat pattern's
//! steps), so they reach drawings, DXF and DWG this way.
//!
//! Projected curves are polylines whose vertices lie on the model edges. A closed one whose
//! vertices all lie on one circle (within 1 nm) is written as that circle; the rest as their
//! straight segments.

use crate::runtime::RegeneratedPart;
use runmat_cad::ids::{FeatureId, SketchEntityId};
use runmat_cad::model::sheet_metal::FlatLayout;
use runmat_cad::regen::kernel::{
    Placed, ProfileSegment, ProjectionRequest, SegmentKind, Transform,
};
use runmat_cad::regen::records::FeatureRecord;

/// The layout of the part's first flat pattern, with its outline from the flat body when the
/// kernel projects exactly (else the layout as recorded); `None` without a flat pattern.
pub fn flat_layout(part: &RegeneratedPart) -> Option<(FeatureId, FlatLayout)> {
    let feature = part.document.features.iter().find(|f| {
        matches!(
            part.result.state.records.get(&f.id),
            Some(FeatureRecord::FlatPattern { .. })
        )
    })?;
    Some((feature.id.clone(), flat_layout_of(part, &feature.id)?))
}

/// The layout the flat pattern or unfold feature `feature` made, its outline from the body it
/// made as in [`flat_layout`]; `None` when the feature made no flat pattern.
pub fn flat_layout_of(part: &RegeneratedPart, feature: &FeatureId) -> Option<FlatLayout> {
    let Some(FeatureRecord::FlatPattern { layout, .. }) = part.result.state.records.get(feature)
    else {
        return None;
    };
    let mut layout = layout.clone();
    let body = part
        .result
        .state
        .bodies
        .iter()
        .find(|b| b.names.body.iter().any(|n| n.maker() == feature));
    if let Some(body) = body {
        if let Some(loops) = projected_loops(part, &body.handle, &layout) {
            layout.loops = loops;
        }
    }
    Some(layout)
}

fn projected_loops(
    part: &RegeneratedPart,
    body: &runmat_cad::regen::kernel::BodyHandle,
    layout: &FlatLayout,
) -> Option<Vec<Vec<ProfileSegment>>> {
    let f = layout.frame;
    // Seen from the side the layout's depth grows towards, so its +z faces the viewer.
    let (y, z) = if layout.depth >= 0.0 {
        (f.y, f.z)
    } else {
        (f.y.map(|v| -v), f.z.map(|v| -v))
    };
    let request = ProjectionRequest {
        frame: runmat_cad::model::datum::Frame {
            origin: f.origin,
            x: f.x,
            y,
            z,
        },
        hidden: false,
        smooth: false,
        tolerance: 1.0e-6,
    };
    let placed = [Placed {
        body,
        transform: Transform::translation([0.0; 3]),
    }];
    let curves = part.kernel.project(&placed, &request).ok()?;
    let flip = layout.depth < 0.0;
    let mut loops = Vec::new();
    for (k, c) in curves.iter().filter(|c| c.visible).enumerate() {
        let pts: Vec<[f64; 2]> = c
            .polyline
            .iter()
            .map(|p| if flip { [p[0], -p[1]] } else { *p })
            .collect();
        if pts.len() < 2 {
            continue;
        }
        let id = SketchEntityId::new(format!("cut{k}"));
        if let Some((center, radius)) = circle(&pts) {
            let at = [center[0] + radius, center[1]];
            loops.push(vec![ProfileSegment {
                curve: id,
                kind: SegmentKind::Circle { center, radius },
                start: at,
                end: at,
            }]);
            continue;
        }
        loops.push(
            pts.windows(2)
                .filter(|w| w[0] != w[1])
                .map(|w| ProfileSegment {
                    curve: id.clone(),
                    kind: SegmentKind::Line,
                    start: w[0],
                    end: w[1],
                })
                .collect(),
        );
    }
    (!loops.is_empty()).then_some(loops)
}

/// The circle through every vertex of a closed polyline, when there is one.
fn circle(pts: &[[f64; 2]]) -> Option<([f64; 2], f64)> {
    let closed = pts.len() >= 8 && {
        let (a, b) = (pts[0], pts[pts.len() - 1]);
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt() < 1e-9
    };
    if !closed {
        return None;
    }
    let ring = &pts[..pts.len() - 1];
    let n = ring.len() as f64;
    let c = [
        ring.iter().map(|p| p[0]).sum::<f64>() / n,
        ring.iter().map(|p| p[1]).sum::<f64>() / n,
    ];
    let d: Vec<f64> = ring
        .iter()
        .map(|p| ((p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2)).sqrt())
        .collect();
    let r = d.iter().sum::<f64>() / n;
    d.iter().all(|x| (x - r).abs() < 1e-9).then_some((c, r))
}
