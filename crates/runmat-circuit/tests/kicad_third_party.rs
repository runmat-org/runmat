//! Real KiCad 8 boards by other authors (`tests/kicad/third_party`, licenses in its README):
//! read, written back token for token, and exported from RunMat's own model with every pad
//! where it was, on its net.

use runmat_circuit::geom::Point;
use runmat_circuit::kicad::{pcb, sexpr};
use runmat_circuit::layout::Board;

const BOARDS: [(&str, &str); 2] = [
    (
        "atmega328p-datalogger-4l",
        include_str!("kicad/third_party/atmega328p-datalogger-4l.kicad_pcb"),
    ),
    (
        "unipa-bioamp-pro",
        include_str!("kicad/third_party/unipa-bioamp-pro.kicad_pcb"),
    ),
];

/// Every pad: footprint, pad name, centre and net.
fn pads(b: &Board) -> Vec<(String, String, Point, Option<String>)> {
    let mut v: Vec<_> = b
        .footprints
        .values()
        .flat_map(|f| {
            f.pads(b.copper_count()).into_iter().map(move |p| {
                (
                    f.reference.clone(),
                    p.name.clone(),
                    p.center,
                    p.net.map(|n| b.net_name(n).to_string()),
                )
            })
        })
        .collect();
    v.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
    v
}

#[test]
fn third_party_boards_round_trip_token_for_token() {
    for (name, text) in BOARDS {
        let doc = pcb::read(text).unwrap_or_else(|e| panic!("{name}: {e}"));
        let out = doc.write();
        assert!(
            sexpr::parse(&out).unwrap() == sexpr::parse(text).unwrap(),
            "{name}: the written file differs from the original"
        );
        assert_eq!(pcb::read(&out).unwrap().write(), out, "{name}: a second write differs");
    }
}

#[test]
fn third_party_boards_read_every_item() {
    for (name, text) in BOARDS {
        let tree = sexpr::parse(text).unwrap();
        let count = |head: &str| tree.children(head).count();
        let doc = pcb::read(text).unwrap();
        let b = &doc.board;
        assert_eq!(b.footprints.len(), count("footprint"), "{name}: footprints");
        assert_eq!(b.tracks.len(), count("segment") + count("arc"), "{name}: tracks");
        assert_eq!(b.vias.len(), count("via"), "{name}: vias");
        assert_eq!(b.zones.len() + b.keepouts.len(), count("zone"), "{name}: zones");
        assert!(b.copper_count() >= 2, "{name}: copper layers");
        assert!(b.outline.area() > 0.0, "{name}: an outline");
        assert!(!pads(b).is_empty(), "{name}: pads");
    }
}

#[test]
fn third_party_boards_export_from_the_runmat_model_with_every_pad_in_place() {
    for (name, text) in BOARDS {
        let board = pcb::read(text).unwrap().board;
        let written = pcb::write_board(&board);
        let back = pcb::read(&written).unwrap_or_else(|e| panic!("{name}: {e}"));
        let b2 = &back.board;
        assert_eq!(b2.copper_count(), board.copper_count(), "{name}");
        assert_eq!(b2.footprints.len(), board.footprints.len(), "{name}");
        assert_eq!(b2.tracks.len(), board.tracks.len(), "{name}");
        assert_eq!(b2.vias.len(), board.vias.len(), "{name}");
        assert_eq!(b2.outline, board.outline, "{name}: outline");
        assert_eq!(pads(b2), pads(&board), "{name}: pads");
    }
}
