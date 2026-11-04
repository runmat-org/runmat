//! Canonical tool sections shared by every kernel, so the labels a hole's faces get are defined
//! once: which segment of the section makes the wall, the counterbore, the countersink, a
//! modelled thread or the bottom, with ordinals from the definition.

use crate::history::{HistorySource, KernelTag};
use crate::names::Role;

fn gen(role: Role, tag: KernelTag, ordinal: Option<u32>) -> HistorySource {
    HistorySource::Generated {
        role,
        tag,
        ordinal,
        seed: None,
        instance: None,
    }
}

/// One straight segment of a section in (radius, z), with the label of the face it sweeps.
#[derive(Clone, Debug, PartialEq)]
pub struct SectionSegment {
    pub a: [f64; 2],
    pub b: [f64; 2],
    pub label: HistorySource,
}

/// A series stage of a hole: where the axis leaves its body (depth, positive down), the radius
/// through it, and whether it is threaded.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StageSpan {
    pub exit: f64,
    pub radius: f64,
    pub threaded: bool,
}

/// A hole's section about its axis in (radius, z), z up and the hole going down: the top (on the
/// start face), any counterbore or countersink, the wall spans (by series stage, with a modelled
/// thread's span its own face, tapering for a pipe tap), then the bottom or drill point. The
/// closing segment lies on the axis and makes no face. Segments run in order around the
/// section; `bottom` is the (negative) z of the full-diameter bottom.
#[allow(clippy::too_many_arguments)]
pub fn hole_section(
    key: &str,
    radius: f64,
    bottom: f64,
    counterbore: Option<(f64, f64)>,
    countersink: Option<(f64, f64)>,
    drill_point: Option<f64>,
    taper: Option<f64>,
    thread: Option<Option<f64>>,
    stages: &[StageSpan],
) -> Vec<SectionSegment> {
    let k = || KernelTag::Key { key: key.into() };
    let mut pts: Vec<([f64; 2], HistorySource)> = Vec::new();
    // Each point carries the label of the segment from it to the next point.
    pts.push(([0.0, 0.0], gen(Role::HoleBottom, k(), Some(1))));
    let z0 = match (counterbore, countersink) {
        (Some((d, depth)), _) => {
            pts.push(([d / 2.0, 0.0], gen(Role::HoleCounterboreWall, k(), None)));
            pts.push(([d / 2.0, -depth], gen(Role::HoleCounterbore, k(), None)));
            -depth
        }
        (None, Some((d, angle))) => {
            let rc = d / 2.0;
            let drop = (rc - radius).max(0.0) / (angle / 2.0).tan().max(1e-9);
            pts.push(([rc, 0.0], gen(Role::HoleCountersink, k(), None)));
            -drop
        }
        (None, None) => 0.0,
    };
    let tan = taper.map(f64::tan).unwrap_or(0.0);
    let r_at = |r0: f64, z: f64| (r0 - (z0 - z) * tan).max(radius * 0.05);
    // Wall spans: (bottom z, radius, label).
    let mut spans: Vec<(f64, f64, HistorySource)> = Vec::new();
    if stages.is_empty() {
        let threaded_to = thread.map(|l| l.map(|l| (z0 - l).max(bottom)).unwrap_or(bottom));
        match threaded_to {
            Some(t) if t > bottom + 1e-12 => {
                spans.push((t, radius, gen(Role::ThreadFace, k(), None)));
                spans.push((bottom, radius, gen(Role::HoleWall, k(), None)));
            }
            Some(_) => spans.push((bottom, radius, gen(Role::ThreadFace, k(), None))),
            None => spans.push((bottom, radius, gen(Role::HoleWall, k(), None))),
        }
    } else {
        let n = stages.len();
        for (i, st) in stages.iter().enumerate() {
            let zb = if i + 1 == n { bottom } else { -st.exit };
            let ord = (i > 0).then_some(i as u32);
            let role = if st.threaded {
                Role::ThreadFace
            } else {
                Role::HoleWall
            };
            spans.push((zb, st.radius, gen(role, k(), ord)));
        }
    }
    let mut z = z0;
    let mut last_r = spans.first().map(|s| s.1).unwrap_or(radius);
    for (i, (zb, r, label)) in spans.iter().enumerate() {
        if (r - last_r).abs() > 1e-12 && i > 0 {
            pts.push((
                [r_at(last_r, z), z],
                gen(Role::HoleBottom, k(), Some(10 + i as u32)),
            ));
        }
        pts.push(([r_at(*r, z), z], label.clone()));
        z = *zb;
        last_r = *r;
    }
    let rb = r_at(last_r, z);
    match drill_point {
        Some(a) if a > 0.0 && a < std::f64::consts::PI => {
            pts.push(([rb, z], gen(Role::HoleBottom, k(), None)));
            let tip = z - rb / (a / 2.0).tan();
            pts.push(([0.0, tip], gen(Role::HoleWall, k(), None)));
        }
        _ => {
            pts.push(([rb, z], gen(Role::HoleBottom, k(), None)));
            pts.push(([0.0, z], gen(Role::HoleWall, k(), None)));
        }
    }
    let n = pts.len();
    (0..n)
        .map(|i| SectionSegment {
            a: pts[i].0,
            b: pts[(i + 1) % n].0,
            label: pts[i].1.clone(),
        })
        .collect()
}
