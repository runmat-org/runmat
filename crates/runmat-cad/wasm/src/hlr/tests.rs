use super::*;

/// The public bracket fixture: an 80 x 50 x 10 plate, a 20 x 10 through-cut at
/// (20..40, 20..30) and a 4 mm through-hole at (60, 10).
pub(super) const BRACKET: &str = include_str!("../../../tests/fixtures/drawing/bracket.part");

fn lines(orientation: &str) -> ViewLines {
    let part = crate::runtime::compile_part("/p/bracket.part", BRACKET)
        .unwrap_or_else(|d| panic!("{d:?}"));
    let display = serde_json::json!({ "kind": "part", "data": part.display });
    view_lines(&display, &axes(orientation))
}

/// Segments of `lines` (flat u, v pairs) as (u0, v0, u1, v1).
fn segments(lines: &[Vec<f64>]) -> Vec<[f64; 4]> {
    lines
        .iter()
        .flat_map(|l| {
            l.chunks_exact(2)
                .collect::<Vec<_>>()
                .windows(2)
                .map(|w| [w[0][0], w[0][1], w[1][0], w[1][1]])
                .collect::<Vec<_>>()
        })
        .collect()
}

fn near(a: f64, b: f64) -> bool {
    (a - b).abs() < 0.05
}

/// Length of the segments at constant `u = at` (vertical) within `v` in [lo, hi].
fn vertical(segs: &[[f64; 4]], at: f64, lo: f64, hi: f64) -> f64 {
    segs.iter()
        .filter(|s| near(s[0], at) && near(s[2], at))
        .map(|s| (s[1].max(s[3]).min(hi) - s[1].min(s[3]).max(lo)).max(0.0))
        .sum()
}

#[test]
fn the_top_view_shows_the_cut_and_the_hole() {
    let top = lines("top");
    let visible = segments(&top.visible);
    // The cut's rectangle: four sides, 20 + 10 + 20 + 10 mm.
    let cut: f64 = visible
        .iter()
        .filter(|s| {
            (20.0 - 0.05..=40.05).contains(&s[0])
                && (20.0 - 0.05..=30.05).contains(&s[1])
                && (20.0 - 0.05..=40.05).contains(&s[2])
                && (20.0 - 0.05..=30.05).contains(&s[3])
        })
        .map(|s| ((s[2] - s[0]).powi(2) + (s[3] - s[1]).powi(2)).sqrt())
        .sum();
    assert!((cut - 60.0).abs() < 0.5, "cut outline {cut} mm of 60");
    // The hole: a closed outline of radius 2 about (60, 10), its circumference.
    let hole: Vec<&[f64; 4]> = visible
        .iter()
        .filter(|s| {
            let r = |u: f64, v: f64| ((u - 60.0).powi(2) + (v - 10.0).powi(2)).sqrt();
            (r(s[0], s[1]) - 2.0).abs() < 0.1 && (r(s[2], s[3]) - 2.0).abs() < 0.1
        })
        .collect();
    let around: f64 = hole
        .iter()
        .map(|s| ((s[2] - s[0]).powi(2) + (s[3] - s[1]).powi(2)).sqrt())
        .sum();
    assert!(hole.len() >= 8, "{} hole segments", hole.len());
    assert!(
        (around - 4.0 * std::f64::consts::PI).abs() < 1.0,
        "hole outline {around} mm"
    );
}

#[test]
fn holes_carry_center_marks_and_centerlines_and_fillets_none() {
    // Top: the 4 mm hole's axis points at the viewer, a mark at (60, 10).
    let top = lines("top");
    assert_eq!(top.centers.len(), 1, "{:?}", top.centers);
    let Center::Mark { at, radius } = top.centers[0] else {
        panic!("{:?}", top.centers);
    };
    assert!(near(at[0], 60.0) && near(at[1], 10.0) && near(radius, 2.0));
    // Front: the axis lies in the view, a centerline at x = 60 through the plate.
    let front = lines("front");
    assert_eq!(front.centers.len(), 1, "{:?}", front.centers);
    let Center::Line { from, to, radius } = front.centers[0] else {
        panic!("{:?}", front.centers);
    };
    assert!(near(from[0], 60.0) && near(to[0], 60.0) && near(radius, 2.0));
    assert!(near(from[1].min(to[1]), 0.0) && near(from[1].max(to[1]), 10.0));
}

#[test]
fn a_fillet_draws_its_end_arcs_so_every_outline_closes() {
    // Front (u = x, v = z): the 1.5 mm round on the top front edge ends in arcs at x = 0
    // and x = 80, from the tangent line at z = 8.5 to the top at z = 10.
    let visible = segments(&lines("front").visible);
    for x in [0.0, 80.0] {
        let end = vertical(&visible, x, 8.5, 10.0);
        assert!(
            end > 1.4,
            "front: the outline at x = {x} has {end} mm of 1.5 above the tangent line"
        );
    }
    // Top (u = x, v = y): the end of the round closes the outline from y = 0 to 1.5.
    let top = segments(&lines("top").visible);
    for x in [0.0, 80.0] {
        assert!(vertical(&top, x, 0.0, 1.5) > 1.4, "top: no end at x = {x}");
    }
}

#[test]
fn the_front_and_right_views_show_the_cut_and_the_hole_hidden() {
    let front = lines("front");
    let hidden = segments(&front.hidden);
    // The cut's walls at x = 20 and 40 and the hole's silhouette at x = 58 and 62, through
    // the 10 mm plate, dashed.
    for x in [20.0, 40.0, 58.0, 62.0] {
        let len = vertical(&hidden, x, 0.0, 10.0);
        assert!(len > 9.0, "front: hidden line at x = {x} is {len} mm");
    }
    assert!(
        vertical(&segments(&front.visible), 20.0, 0.0, 10.0) < 0.5,
        "the cut is behind the front face"
    );
    let right = lines("right");
    let hidden = segments(&right.hidden);
    // Seen from the right, u is y: the cut at y = 20 and 30, the hole at y = 8 and 12.
    for y in [20.0, 30.0, 8.0, 12.0] {
        let len = vertical(&hidden, y, 0.0, 10.0);
        assert!(len > 9.0, "right: hidden line at y = {y} is {len} mm");
    }
}

/// A hole tessellated with facets square to the view at both sides (an octagon turned 22.5°,
/// as OCCT may mesh it): seen from the front, both silhouettes draw, at x = ±r cos 22.5°.
#[test]
fn a_facet_edge_on_at_the_silhouette_still_draws_it() {
    let r = 2.0;
    let ring = |z: f64| -> Vec<P3> {
        (0..8)
            .map(|k| {
                let a = (22.5 + 45.0 * k as f64).to_radians();
                [r * a.cos(), r * a.sin(), z]
            })
            .collect()
    };
    let (lo, hi) = (ring(0.0), ring(10.0));
    let mut tris = Vec::new();
    for k in 0..8 {
        let j = (k + 1) % 8;
        tris.push([lo[k], lo[j], hi[j]]);
        tris.push([lo[k], hi[j], hi[k]]);
    }
    let model = Model {
        faces: vec![Face { tris, body: 0 }],
        edges: Vec::new(),
        axes: Vec::new(),
        bodies: 1,
    };
    let front = axes("front");
    let xs: Vec<f64> = silhouettes(&model, front.depth)
        .iter()
        .map(|e| dot(e.points[0], front.u))
        .collect();
    let x = r * 22.5_f64.to_radians().cos();
    assert!(
        xs.iter().any(|u| (u - x).abs() < 1e-9) && xs.iter().any(|u| (u + x).abs() < 1e-9),
        "silhouettes at {xs:?}"
    );
}

/// A closed loft's lateral surface split into two faces along its crest, with no edge drawn
/// between them (the camera drone's body): the front view's outline runs along the split.
#[test]
fn a_silhouette_along_a_smooth_split_between_two_faces_still_draws() {
    let r = 2.0;
    // 15-degree facets, as a fine tessellation has them; the split runs through the top and bottom.
    let ring = |x: f64| -> Vec<P3> {
        (0..24)
            .map(|k| {
                let a = (90.0 + 15.0 * k as f64).to_radians();
                [x, r * a.cos(), r * a.sin()]
            })
            .collect()
    };
    let (lo, hi) = (ring(0.0), ring(10.0));
    let mut halves = [Vec::new(), Vec::new()];
    for k in 0..24 {
        let j = (k + 1) % 24;
        halves[k / 12].push([lo[k], hi[j], lo[j]]);
        halves[k / 12].push([lo[k], hi[k], hi[j]]);
    }
    let [a, b] = halves;
    let front = axes("front");
    // One body: the outline at the top and the bottom, along the split.
    let model = Model {
        faces: vec![
            Face {
                tris: a.clone(),
                body: 0,
            },
            Face {
                tris: b.clone(),
                body: 0,
            },
        ],
        edges: Vec::new(),
        axes: Vec::new(),
        bodies: 1,
    };
    let mut vs: Vec<f64> = silhouettes(&model, front.depth)
        .iter()
        .map(|e| dot(e.points[0], front.v))
        .collect();
    vs.sort_by(f64::total_cmp);
    vs.dedup_by(|x, y| (*x - *y).abs() < 1e-9);
    assert_eq!(vs.len(), 2, "silhouettes at {vs:?}");
    assert!(
        (vs[0] + r).abs() < 1e-9 && (vs[1] - r).abs() < 1e-9,
        "silhouettes at {vs:?}"
    );
    let lines = lines_of(&model, &front, true);
    let top: f64 = segments(&lines.visible)
        .iter()
        .filter(|s| (s[1] - r).abs() < 1e-9 && (s[3] - r).abs() < 1e-9)
        .map(|s| (s[2] - s[0]).abs())
        .sum();
    assert!(
        (top - 10.0).abs() < 1e-6,
        "the top outline draws {top} of 10 mm"
    );
    // Two bodies that touch: no outline between them.
    let apart = Model {
        faces: vec![Face { tris: a, body: 0 }, Face { tris: b, body: 1 }],
        edges: Vec::new(),
        axes: Vec::new(),
        bodies: 2,
    };
    assert!(silhouettes(&apart, front.depth).is_empty());
}

/// Hidden runs under visible lines are dropped and the others kept, and a view of an assembly's
/// size (tens of thousands of runs each way) merges at once: each point tests only the visible
/// segments near it.
#[test]
fn hidden_runs_under_visible_lines_merge_by_neighbourhood() {
    let mut out = ViewLines::default();
    let n = 40_000;
    for i in 0..n {
        let (x, y) = ((i % 200) as f64 * 3.0, (i / 200) as f64 * 3.0);
        // A visible stroke, a hidden run along it (dropped) and one beside it (kept).
        out.visible.push(vec![x, y, x + 2.0, y]);
        out.hidden.push(vec![x + 0.5, y, x + 1.5, y]);
        out.hidden.push(vec![x + 0.5, y + 1.0, x + 1.5, y + 1.0]);
    }
    // A long diagonal visible line with a hidden run on it.
    out.visible.push(vec![0.0, 0.0, 600.0, 600.0]);
    out.hidden.push(vec![100.0, 100.0, 101.0, 101.0]);
    let started = std::time::Instant::now();
    merge::merge_overlaps(&mut out);
    assert_eq!(out.hidden.len(), n, "only the runs beside the strokes stay");
    assert!(out.hidden.iter().all(|l| l[1] % 3.0 == 1.0));
    assert!(
        started.elapsed().as_secs_f64() < 5.0,
        "{:?}",
        started.elapsed()
    );
}

/// In a model 40 m across, an edge 3 mm behind a skin is hidden: how far in front a face must
/// be to hide a line doesn't grow with the model (an airliner's frames under its 3 mm skin).
#[test]
fn a_thin_skin_hides_what_touches_it_in_a_large_model() {
    // The skin: the plane y = -3 mm over x in [0, 40 m], z in [-100, 100] mm.
    let skin = vec![
        [
            [0.0, -3.0, -100.0],
            [40_000.0, -3.0, -100.0],
            [40_000.0, -3.0, 100.0],
        ],
        [
            [0.0, -3.0, -100.0],
            [40_000.0, -3.0, 100.0],
            [0.0, -3.0, 100.0],
        ],
    ];
    // A frame's face touching the skin's inside, and its edge on it.
    let frame = vec![[
        [20_000.0, 0.0, -50.0],
        [20_040.0, 0.0, -50.0],
        [20_000.0, 0.0, 50.0],
    ]];
    let model = Model {
        faces: vec![
            Face {
                tris: skin,
                body: 0,
            },
            Face {
                tris: frame,
                body: 1,
            },
        ],
        edges: vec![Edge {
            points: vec![[20_000.0, 0.0, -50.0], [20_000.0, 0.0, 50.0]],
            faces: [1, 1],
        }],
        axes: Vec::new(),
        bodies: 2,
    };
    let lines = lines_of(&model, &axes("front"), true);
    assert!(lines.visible.is_empty(), "visible: {:?}", lines.visible);
    assert!(!lines.hidden.is_empty());
}

/// In a view of visible lines, a cylinder under the skin (a rivet's shank, a frame) draws no
/// center mark: only the cylinders that show carry one.
#[test]
fn a_hidden_cylinder_draws_no_center_mark() {
    let skin = vec![
        [
            [-100.0, -3.0, -100.0],
            [100.0, -3.0, -100.0],
            [100.0, -3.0, 100.0],
        ],
        [
            [-100.0, -3.0, -100.0],
            [100.0, -3.0, 100.0],
            [-100.0, -3.0, 100.0],
        ],
    ];
    let shank = Axis {
        point: [0.0, 0.0, 0.0],
        dir: [0.0, 1.0, 0.0],
        radius: 2.0,
        lo: 0.0,
        hi: 10.0,
        face: 1,
    };
    let model = |covered: bool| Model {
        faces: vec![
            Face {
                tris: if covered { skin.clone() } else { Vec::new() },
                body: 0,
            },
            Face {
                tris: Vec::new(),
                body: 1,
            },
        ],
        edges: Vec::new(),
        axes: vec![shank],
        bodies: 2,
    };
    let front = axes("front");
    assert!(lines_of(&model(true), &front, false).centers.is_empty());
    assert_eq!(lines_of(&model(false), &front, false).centers.len(), 1);
    // A view that draws hidden lines marks it too, as it draws the hidden cylinder.
    assert_eq!(lines_of(&model(true), &front, true).centers.len(), 1);
}

/// Two cylinders on one axis line join into one centerline only where they meet: a hole and
/// its counterbore do, two wheels' axles in line across an aircraft don't.
#[test]
fn centerlines_in_line_join_only_where_they_meet() {
    let axle = |x: f64, lo: f64, hi: f64| Axis {
        point: [x, 0.0, 0.0],
        dir: [1.0, 0.0, 0.0],
        radius: 5.0,
        lo,
        hi,
        face: 0,
    };
    let model = |axes: Vec<Axis>| Model {
        faces: vec![Face {
            tris: Vec::new(),
            body: 0,
        }],
        edges: Vec::new(),
        axes,
        bodies: 1,
    };
    let front = axes("front");
    let apart = centers(
        &model(vec![axle(0.0, 0.0, 10.0), axle(100.0, 0.0, 10.0)]),
        None,
        &front,
    );
    assert_eq!(apart.len(), 2, "{apart:?}");
    let meeting = centers(
        &model(vec![axle(0.0, 0.0, 10.0), axle(10.0, 0.0, 10.0)]),
        None,
        &front,
    );
    assert_eq!(meeting.len(), 1, "{meeting:?}");
}
