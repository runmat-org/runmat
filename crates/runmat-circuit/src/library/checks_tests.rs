use super::*;
use crate::geom::poly::rect_ring;
use crate::geom::{mm, Rect};
use crate::library::catalog::CatalogPin;
use crate::library::footprint::from_string;
use crate::library::{PadDef, PadShape};

fn pad(name: &str, x: f64) -> PadDef {
    PadDef::smd(
        name,
        Point::mm(x, 0.0),
        PadShape::Rect {
            w: mm(1.0),
            h: mm(1.0),
        },
    )
}

fn footprint(pads: Vec<PadDef>) -> Package {
    let mut p = from_string("R0402").unwrap();
    p.pads = pads;
    p.courtyard = vec![rect_ring(Rect::new(
        Point::mm(-3.0, -1.0),
        Point::mm(3.0, 1.0),
    ))];
    p
}

#[test]
fn a_generated_footprint_passes() {
    for s in [
        "QFN24_4mm_P0.5_EP2.6",
        "SOIC8",
        "PinHeader2x05",
        "MountingHole_3.2mm_Pad6",
        "BGA256_16x16_P1",
        "SMDConn4_P1_MP1.2x1.8",
    ] {
        assert_eq!(
            footprint_findings(s, &from_string(s).unwrap()),
            Vec::<String>::new(),
            "{s}"
        );
    }
}

#[test]
fn each_footprint_finding_names_the_footprint_and_the_pad() {
    let f = footprint(vec![pad("1", -1.0), pad("1", 1.0), pad("3", 2.8)]);
    let found = footprint_findings("lcc", &f);
    assert!(
        found
            .iter()
            .any(|m| m == "footprint `lcc`: pad `1` is declared twice"),
        "{found:?}"
    );
    assert!(
        found
            .iter()
            .any(|m| m.starts_with("footprint `lcc`: pad `3` reaches past the courtyard")),
        "{found:?}"
    );
    let mut big = footprint(vec![PadDef::tht(
        "1",
        Point::ORIGIN,
        PadShape::Circle { d: mm(1.0) },
        mm(1.0),
    )]);
    assert!(
        footprint_findings("hole", &big)[0].contains("pad `1` has a 1 mm drill in a 1 x 1 mm pad")
    );
    big.pads[0].drill.as_mut().unwrap().diameter = mm(0.6);
    assert!(footprint_findings("hole", &big).is_empty());
    let mut bare = footprint(vec![pad("1", 0.0)]);
    bare.courtyard.clear();
    assert_eq!(
        footprint_findings("bare", &bare),
        ["footprint `bare` has no courtyard"]
    );
}

#[test]
fn every_pad_has_a_pin_or_is_listed_nc() {
    let pin = |name: &str, pad: &str| CatalogPin {
        name: name.into(),
        pads: vec![pad.into()],
        kind: "passive".into(),
    };
    let package = from_string("SOT23-5").unwrap();
    let mut part = CatalogPart {
        mpn: "X".into(),
        package: "SOT23-5".into(),
        pins: vec![pin("A", "1"), pin("B", "2"), pin("C", "3"), pin("D", "9")],
        ..Default::default()
    };
    let found = part_findings(&part, &package);
    assert!(
        found
            .iter()
            .any(|m| m == "part `X`: pin `D` lands on pad `9`, which its footprint has no pad of"),
        "{found:?}"
    );
    assert!(
        found
            .iter()
            .any(|m| m.contains("pads `4`, `5` have no pin")),
        "{found:?}"
    );
    part.pins.pop();
    part.nc = vec!["4".into(), "5".into()];
    assert_eq!(part_findings(&part, &package), Vec::<String>::new());
    part.nc.push("1".into());
    assert_eq!(
        part_findings(&part, &package),
        ["part `X`: pad `1` has a pin and is also listed in `nc`"]
    );
}
