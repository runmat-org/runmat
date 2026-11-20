use super::*;
use std::f64::consts::PI;

fn id() -> SketchEntityId {
    SketchEntityId::new("t1")
}

/// Signed area of a loop (counter-clockwise positive), from finely sampled segments.
fn area(l: &[ProfileSegment]) -> f64 {
    l.iter()
        .flat_map(|s| {
            let pts = s.polyline(400);
            pts.windows(2)
                .map(|w| (w[0][0] * w[1][1] - w[1][0] * w[0][1]) / 2.0)
                .collect::<Vec<_>>()
        })
        .sum()
}

/// Every loop closes: each segment starts exactly where the one before it ends.
fn assert_closed(loops: &[Vec<ProfileSegment>]) {
    for l in loops {
        for k in 0..l.len() {
            let (end, start) = (l[k].end, l[(k + 1) % l.len()].start);
            assert!(dist(end, start) < 1e-12, "{l:#?}");
        }
    }
}

#[test]
fn one_segment_is_a_capsule_of_two_lines_and_two_arcs() {
    let loops = stroke_outline(&[vec![[0.0, 0.0], [10.0, 0.0]]], 1.0, &id());
    assert_eq!(loops.len(), 1);
    assert_closed(&loops);
    let kinds: Vec<&str> = loops[0]
        .iter()
        .map(|s| match s.kind {
            SegmentKind::Line => "line",
            SegmentKind::Arc { .. } => "arc",
            _ => "other",
        })
        .collect();
    assert_eq!(
        kinds.iter().filter(|k| **k == "line").count(),
        2,
        "{kinds:?}"
    );
    assert_eq!(
        kinds.iter().filter(|k| **k == "arc").count(),
        2,
        "{kinds:?}"
    );
    let want = 2.0 * 10.0 + PI;
    assert!((area(&loops[0]) - want).abs() < 1e-3, "{}", area(&loops[0]));
}

#[test]
fn a_dot_is_one_circle() {
    let loops = stroke_outline(&[vec![[3.0, 4.0]]], 0.5, &id());
    assert_eq!(loops.len(), 1);
    assert!(matches!(loops[0][0].kind, SegmentKind::Circle { radius, .. } if radius == 0.5));
}

#[test]
fn a_bent_stroke_and_a_crossing_merge_into_one_outline() {
    // An L: two legs with a round outer corner.
    let l = stroke_outline(&[vec![[0.0, 10.0], [0.0, 0.0], [6.0, 0.0]]], 1.0, &id());
    assert_eq!(l.len(), 1);
    assert_closed(&l);
    // Two capsules less their overlap: the corner disk and the unit square inside the bend.
    let want = 2.0 * 10.0 + 2.0 * 6.0 + PI - 1.0 + PI / 4.0;
    assert!(
        (area(&l[0]) - want).abs() < 1e-3,
        "{} vs {want}",
        area(&l[0])
    );
    // A plus: two crossing capsules overlap in a 2 x 2 square.
    let plus = stroke_outline(
        &[vec![[-5.0, 0.0], [5.0, 0.0]], vec![[0.0, -5.0], [0.0, 5.0]]],
        1.0,
        &id(),
    );
    assert_eq!(plus.len(), 1);
    assert_closed(&plus);
    let want = 2.0 * (2.0 * 10.0 + PI) - 4.0;
    assert!(
        (area(&plus[0]) - want).abs() < 1e-3,
        "{} vs {want}",
        area(&plus[0])
    );
}

#[test]
fn a_closed_square_stroke_has_an_outer_loop_and_a_hole() {
    let square = vec![
        [0.0, 0.0],
        [10.0, 0.0],
        [10.0, 10.0],
        [0.0, 10.0],
        [0.0, 0.0],
    ];
    let loops = stroke_outline(&[square], 1.0, &id());
    assert_eq!(loops.len(), 2, "{loops:#?}");
    assert_closed(&loops);
    let mut areas: Vec<f64> = loops.iter().map(|l| area(l)).collect();
    areas.sort_by(f64::total_cmp);
    // The hole runs clockwise: an 8 x 8 square. The outer: 12 x 12 with rounded corners.
    assert!((areas[0] + 64.0).abs() < 1e-3, "{areas:?}");
    assert!((areas[1] - (144.0 - 4.0 + PI)).abs() < 1e-3, "{areas:?}");
}

#[test]
fn nothing_to_draw_or_no_pen_gives_no_loops() {
    assert!(stroke_outline(&[], 1.0, &id()).is_empty());
    assert!(stroke_outline(&[vec![[0.0, 0.0], [1.0, 0.0]]], 0.0, &id()).is_empty());
}
