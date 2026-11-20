//! Sketch text: the text form round-trips, and every glyph of the single-line font becomes closed
//! outline loops whose area is the stroke's.

use runmat_cad::model::sketch::*;
use runmat_cad::model::FeatureKind;
use runmat_cad::sketch_solver::{ConstraintSketchSolver, SketchSolver};
use runmat_cad::syntax::{parse_part, print_part};
use runmat_cad::units::Units;
use runmat_cad::value::{Evaluator, NoDynamicValues, ValueEnv};
use runmat_geometry_kernel::op::ProfileSegment;
use runmat_types::font::{layout, Anchor};

fn part(body: &str) -> String {
    format!("runmat 0.7\nstd 1\n\npart p {{\n    sketch s1 on origin.xy {{\n{body}    }}\n}}\n")
}

fn solved(text: &str) -> (Sketch, SolvedSketch) {
    let parsed = parse_part("p.part", text);
    assert!(
        parsed.diagnostics.is_empty(),
        "{text}\n{:#?}",
        parsed.diagnostics
    );
    let FeatureKind::Sketch(s) = &parsed.document.unwrap().features[0].kind else {
        panic!()
    };
    let units = Units;
    let ev = Evaluator {
        units: &units,
        dynamic: &NoDynamicValues,
    };
    let solved = ConstraintSketchSolver
        .solve(s, &ev, &ValueEnv::default())
        .expect("solves");
    (s.clone(), solved)
}

fn signed_area(l: &[ProfileSegment]) -> f64 {
    l.iter()
        .flat_map(|s| {
            s.polyline(64)
                .windows(2)
                .map(|w| (w[0][0] * w[1][1] - w[1][0] * w[0][1]) / 2.0)
                .collect::<Vec<_>>()
        })
        .sum()
}

fn assert_closed(loops: &[Vec<ProfileSegment>], what: &str) {
    for l in loops {
        for k in 0..l.len() {
            let (end, start) = (l[k].end, l[(k + 1) % l.len()].start);
            let gap = (end[0] - start[0]).hypot(end[1] - start[1]);
            assert!(gap < 1e-12, "{what}: a loop is open by {gap}");
        }
    }
}

#[test]
fn text_prints_as_it_is_written_and_names_its_font() {
    let text = part(concat!(
        "        t1 = text(text = \"RUN \\\"MAT\\\"\", at = [5 mm, 5 mm], height = 8 mm, angle = 90 deg, font = single_line, stroke = 1.2 mm)\n",
        "        t2 = text(text = \"A\", at = t1.at, height = 3 mm, font = single_line)\n",
        "        constrain fix(t1.at)\n",
    ));
    let parsed = parse_part("p.part", &text);
    assert!(parsed.diagnostics.is_empty(), "{:#?}", parsed.diagnostics);
    let printed = print_part(parsed.document.as_ref().unwrap(), &parsed.uses).unwrap();
    assert_eq!(printed, text);
    // `font` defaults to the single-line font and prints by name.
    let bare = part("        t1 = text(text = \"A\", at = [0 mm, 0 mm], height = 3 mm)\n");
    let parsed = parse_part("p.part", &bare);
    let printed = print_part(parsed.document.as_ref().unwrap(), &parsed.uses).unwrap();
    assert!(
        printed.contains("height = 3 mm, font = single_line)"),
        "{printed}"
    );
    // Any other font says which one sketches use.
    let other =
        part("        t1 = text(text = \"A\", at = [0 mm, 0 mm], height = 3 mm, font = sans)\n");
    let message = format!("{:?}", parse_part("p.part", &other).diagnostics);
    assert!(message.contains("`single_line`"), "{message}");
}

#[test]
fn text_is_closed_outlines_at_its_anchor_height_and_angle() {
    let (_, s) = solved(&part(
        "        t1 = text(text = \"RUNMAT\", at = [10 mm, 20 mm], height = 6 mm, angle = 90 deg, stroke = 1 mm)\n",
    ));
    let loops = s.text_loops(&"t1".into(), "RUNMAT");
    assert_closed(&loops, "RUNMAT");
    // R, U, N, M, A, T are one outline each; R and A each have a counter (a hole).
    let holes = loops.iter().filter(|l| signed_area(l) < 0.0).count();
    assert_eq!(
        (loops.len() - holes, holes),
        (6, 2),
        "{} loops",
        loops.len()
    );
    // Turned a quarter turn about the anchor: the lettering runs up from (10, 20) mm, its
    // capitals reaching 6 mm (plus half the pen) to the left.
    let pts: Vec<[f64; 2]> = loops.iter().flatten().flat_map(|s| s.polyline(8)).collect();
    let (x0, x1) = pts
        .iter()
        .fold((f64::MAX, f64::MIN), |(a, b), p| (a.min(p[0]), b.max(p[0])));
    let (y0, _) = pts
        .iter()
        .fold((f64::MAX, f64::MIN), |(a, b), p| (a.min(p[1]), b.max(p[1])));
    assert!((x0 - (0.010 - 0.006 - 0.0005)).abs() < 1e-9, "{x0}");
    assert!((x1 - (0.010 + 0.0005)).abs() < 1e-9, "{x1}");
    assert!((y0 - (0.020 - 0.0005)).abs() < 1e-9, "{y0}");
}

/// Every printable character's outline closes, and its area matches the area within the pen's
/// reach of its strokes, counted on a fine grid.
#[test]
fn every_glyph_outlines_to_its_stroke_area() {
    let r = 0.45; // grid units: the default pen on a 6-unit capital
    for b in 0x21u8..0x7f {
        let c = (b as char).to_string();
        let strokes: Vec<Vec<[f64; 2]>> = layout(&c, Anchor::BaselineLeft)
            .into_iter()
            .map(|pl| pl.into_iter().map(|(x, y)| [x, y]).collect())
            .collect();
        let loops = runmat_geometry_kernel::stroke::stroke_outline(&strokes, r, &"t".into());
        assert_closed(&loops, &c);
        let area: f64 = loops.iter().map(|l| signed_area(l)).sum();
        let near = |p: [f64; 2]| {
            strokes.iter().any(|s| {
                s.iter().any(|q| (p[0] - q[0]).hypot(p[1] - q[1]) <= r)
                    || s.windows(2).any(|w| {
                        let (a, b) = (w[0], w[1]);
                        let d = [b[0] - a[0], b[1] - a[1]];
                        let l2 = d[0] * d[0] + d[1] * d[1];
                        let t = ((p[0] - a[0]) * d[0] + (p[1] - a[1]) * d[1]) / l2;
                        (0.0..=1.0).contains(&t)
                            && ((p[0] - a[0]) * d[1] - (p[1] - a[1]) * d[0]).abs() / l2.sqrt() <= r
                    })
            })
        };
        let step = 0.025;
        let mut count = 0usize;
        let (mut y, mut cells) = (-2.5, 0usize);
        while y < 7.5 {
            let mut x = -0.5;
            while x < 4.5 {
                cells += 1;
                if near([x + step / 2.0, y + step / 2.0]) {
                    count += 1;
                }
                x += step;
            }
            y += step;
        }
        let counted = count as f64 * step * step;
        assert!(
            cells > 0 && (area - counted).abs() <= 0.01 * counted,
            "{c}: outline {area}, grid {counted}"
        );
    }
}
