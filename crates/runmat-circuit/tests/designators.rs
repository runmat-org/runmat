//! Reference designators: one placement, shared by the board display and the KiCad export.
use runmat_circuit::designators::{designators, names_itself, to_local};
use runmat_circuit::fixture::{large, LargeParams};
use runmat_circuit::geom::to_mm;

fn board() -> runmat_circuit::layout::Board {
    let params = LargeParams {
        channels: 4,
        copper_layers: 2,
        route_every: 0,
        pours: false,
    };
    large(&params).3
}

#[test]
fn every_unnamed_part_gets_a_designator_clear_of_pads_and_of_the_others() {
    let board = board();
    let placed = designators(&board);
    for f in board.footprints.values() {
        assert_eq!(placed.contains_key(&f.id), !names_itself(f), "{}", f.reference);
    }
    assert!(!placed.is_empty());
    let pads: Vec<_> = board
        .footprints
        .values()
        .flat_map(|f| f.pads(board.copper_count()))
        .map(|p| p.prim.bbox())
        .collect();
    let boxes: Vec<(String, [f64; 4])> = placed
        .iter()
        .map(|(id, d)| {
            let f = &board.footprints[id];
            let size = to_mm(d.size);
            assert!((0.6..=1.0).contains(&size), "{} at {size} mm", f.reference);
            let (w, h) = (0.62 * size * f.reference.len() as f64, 0.78 * size);
            let (x, y) = (to_mm(d.at.x), to_mm(d.at.y));
            (f.reference.clone(), [x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0])
        })
        .collect();
    let overlaps = |a: &[f64; 4], b: &[f64; 4]| a[0] < b[2] && b[0] < a[2] && a[1] < b[3] && b[1] < a[3];
    let mut on_pads = 0;
    for (name, b) in &boxes {
        if pads.iter().any(|p| {
            overlaps(b, &[to_mm(p.min.x), to_mm(p.min.y), to_mm(p.max.x), to_mm(p.max.y)])
        }) {
            on_pads += 1;
            eprintln!("{name} touches a pad");
        }
    }
    assert_eq!(on_pads, 0, "designators on pads");
    for (i, (a, ba)) in boxes.iter().enumerate() {
        for (b, bb) in &boxes[i + 1..] {
            assert!(!overlaps(ba, bb), "{a} overlaps {b}");
        }
    }
}

#[test]
fn the_kicad_export_writes_the_displayed_designator_positions() {
    let board = board();
    let placed = designators(&board);
    let text = runmat_circuit::kicad::pcb::write_board(&board);
    let (id, d) = placed.iter().next().unwrap();
    let f = &board.footprints[id];
    let local = to_local(f, d.at);
    // The Reference property of that footprint carries the same point in the package's frame
    // (KiCad's y runs down).
    let at = format!(
        "(property \"Reference\" \"{}\" (at {} {})",
        f.reference,
        fmt(to_mm(local.x)),
        fmt(to_mm(-local.y))
    );
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    assert!(flat.contains(&at), "{at}");
}

fn fmt(v: f64) -> String {
    runmat_circuit::kicad::sexpr::fmt_num(v)
}
