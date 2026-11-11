use super::*;

/// Every family, as the docs' table writes it: the string reads, names its family, and its
/// canonical form makes the identical package.
#[test]
fn every_family_reads_and_its_canonical_string_makes_the_same_package() {
    for (s, family, pads) in [
        ("R0402", "chip", 2),
        ("LED0805", "chip", 2),
        ("C0603@A", "chip", 2),
        ("SOIC8", "soic", 8),
        ("SOIC16W", "soic", 16),
        ("TSSOP20", "tssop", 20),
        ("MSOP10", "msop", 10),
        ("QFP48_7mm_P0.5", "qfp", 48),
        ("QFN24_4mm_P0.5_EP2.6", "qfn", 25),
        ("QFN16_3mm_P0.5", "qfn", 16),
        ("DFN8_3x3mm_P0.65_EP1.6x2.4", "dfn", 9),
        ("BGA256_16x16_P1", "bga", 256),
        ("SOT23-6", "sot23", 6),
        ("SOT223", "sot223", 4),
        ("PinHeader1x04", "pin_header", 4),
        ("PinHeader2x05_P2", "pin_header", 10),
        ("SMDConn4_P1", "smd_connector", 4),
        ("SMDConn4_P1_MP1.2x1.8", "smd_connector", 6),
        ("MountingHole_3.2mm_Pad6", "mounting_hole", 1),
    ] {
        let d = describe(s).unwrap_or_else(|| panic!("{s} describes"));
        assert_eq!(d.family, family, "{s}");
        assert_eq!(d.canonical, s, "{s} is already canonical");
        let p = from_string(s).unwrap();
        assert_eq!(p.pads.len(), pads, "{s}");
        assert_eq!(from_string(&d.canonical), Some(p), "{s}");
    }
}

#[test]
fn other_spellings_print_as_the_canonical_string() {
    for (written, canon) in [
        ("qfn32_5mm_p0.5_ep3.3", "QFN32_5mm_P0.5_EP3.3"),
        ("QFN32_5_P0.50_EP3.30", "QFN32_5mm_P0.5_EP3.3"),
        ("soic16w", "SOIC16W"),
        ("C0402@B", "C0402"),
        ("c0402@c", "C0402@C"),
        ("sot23", "SOT23-3"),
        ("pinheader1x04_p2.54", "PinHeader1x04"),
        ("bga100_16x16_p1.0_ball0.5", "BGA256_16x16_P1"),
        ("PinHeader1x04@A", "PinHeader1x04"),
        ("MountingHole_3.2", "MountingHole_3.2mm_Pad6"),
        ("dfn8_3x3_p0.65_ep1.6", "DFN8_3x3mm_P0.65_EP1.6x1.6"),
        ("SMDConn4_Pad0.6x1.5_P1", "SMDConn4_P1"),
    ] {
        assert_eq!(canonical(written), canon, "{written}");
        assert_eq!(from_string(written), from_string(canon), "{written}");
    }
    // A footprint file, or text that names nothing, is left as written.
    assert_eq!(
        canonical("libs/M.pretty/X.kicad_mod"),
        "libs/M.pretty/X.kicad_mod"
    );
    assert_eq!(canonical("frob"), "frob");
}

#[test]
fn a_description_carries_the_parameters_the_wizard_edits() {
    let d = describe("QFN24_4mm_P0.5_EP2.6@C").unwrap();
    assert_eq!((d.label, d.density), ("QFN", Some('C')));
    let p: Vec<(&str, serde_json::Value)> = d
        .parameters
        .iter()
        .map(|p| (p.name, p.value.clone()))
        .collect();
    assert_eq!(
        p,
        [
            ("pins", 24.into()),
            ("body", 4.0.into()),
            ("pitch", 0.5.into()),
            ("exposed_pad", 2.6.into())
        ]
    );
    assert_eq!(describe("MountingHole_3.2mm_Pad6").unwrap().density, None);
    for bad in [
        "frob",
        "R9999",
        "SOT23-4",
        "QFN24_4mm",
        "MountingHole_3mm_Pad2",
        "C0402@Q",
    ] {
        assert!(describe(bad).is_none(), "{bad}");
        assert!(from_string(bad).is_none(), "{bad}");
    }
}
