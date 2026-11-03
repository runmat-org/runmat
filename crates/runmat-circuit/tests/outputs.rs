//! Fabrication outputs, design-review suites and the board → FEA contract.

use runmat_circuit::board_rules::{BoardGraph, BoardRules};
use runmat_circuit::elab::{elaborate, ElabOptions, Severity};
use runmat_circuit::fab;
use runmat_circuit::fea;
use runmat_circuit::fixture::{self, standard_pool, LargeParams};
use runmat_circuit::ir::*;
use runmat_circuit::review::{self, ReviewOptions};
use runmat_rules::{Dim, Quantity};
use std::collections::BTreeMap;

fn small() -> (
    runmat_circuit::library::Pool,
    Design,
    runmat_circuit::elab::Netlist,
    runmat_circuit::layout::Board,
) {
    let p = LargeParams {
        channels: 4,
        copper_layers: 4,
        route_every: 1,
        pours: true,
    };
    let (pool, mut d, _, _) = fixture::large(&p);
    // One part not fitted in the variant, and authored power on the FPGA.
    d.variants.insert(
        "lite".into(),
        VariantDef {
            name: "lite".into(),
            overrides: [("/ch[3]/CF".to_string(), Fitting::NotFitted)].into(),
            ..Default::default()
        },
    );
    d.sheets
        .get_mut("main")
        .unwrap()
        .components
        .iter_mut()
        .find(|c| c.name == "U9000")
        .unwrap()
        .power = Some(Quantity::new(2.5, Dim::POWER));
    let nl = elaborate(
        &d,
        &pool,
        &ElabOptions {
            variant: Some("lite".into()),
        },
        None,
    );
    let mut board = fixture::large_board(&pool, &nl, &p);
    let (mut rules, _) = BoardRules::new(&board.limits, &[]);
    rules.rebuild(&BoardGraph::new(&board, &[]));
    runmat_circuit::zone::fill_zones(&mut board, &rules, &[]);
    (pool, d, nl, board)
}

#[test]
fn gerber_drill_netlist_bom_pnp() {
    let (pool, _, nl, board) = small();
    let g = fab::gerber_set(&board);
    assert_eq!(g.len(), 4 + 6 + 1);
    let top = &g["large-F_Cu.gbr"];
    assert!(top.starts_with("%TF.GenerationSoftware,RunMat"));
    assert!(top.contains("%TF.FileFunction,Copper,L1,Top*%"));
    assert!(top.contains("%FSLAX46Y46*%") && top.contains("%MOMM*%"));
    assert!(top.contains("%TO.N,GND*%"));
    assert!(top.contains("%TO.P,"));
    assert!(top.contains("SMDPad,CuDef"));
    assert!(top.trim_end().ends_with("M02*"));
    let inner = &g["large-In1_Cu.gbr"];
    assert!(
        inner.contains("G36*") && inner.contains("%LPC*%"),
        "zone fill regions with holes"
    );
    assert!(g["large-F_Mask.gbr"].contains("%TF.FilePolarity,Negative*%"));
    assert!(g["large-Edge_Cuts.gbr"].contains("Profile,NP"));
    // Same input, same bytes.
    assert_eq!(fab::gerber_set(&board), g);

    let drl = fab::excellon(&board);
    let pth = drl
        .iter()
        .find(|(k, _)| k.contains("PTH-1-4"))
        .map(|(_, v)| v)
        .expect("through-hole drill file");
    assert!(pth.starts_with("M48") && pth.contains("METRIC") && pth.trim_end().ends_with("M30"));
    assert!(pth.contains("TF.FileFunction,Plated,1,4,PTH"));
    assert!(pth.lines().filter(|l| l.starts_with('X')).count() >= board.vias.len());

    let d356 = fab::ipc356(&board);
    assert!(d356.contains("P  UNITS CUST 1"));
    assert!(d356.lines().any(|l| l.starts_with("327")));
    assert!(d356.lines().any(|l| l.starts_with("317GND")));
    assert!(d356.trim_end().ends_with("999"));

    let bom = fab::bom(&nl, &pool);
    let total: usize = bom.iter().map(|l| l.quantity).sum();
    let fitted = nl.components.iter().filter(|c| c.fitted()).count();
    assert_eq!(total, fitted);
    assert_eq!(
        fitted,
        nl.components.len() - 1,
        "one part not fitted in the variant"
    );
    let r = bom.iter().find(|l| l.mpn == "R0402-10K").unwrap();
    assert_eq!(r.quantity, 12);
    assert!(r
        .designators
        .windows(2)
        .all(|w| fab::natural_key(&w[0]) < fab::natural_key(&w[1])));
    let csv = fab::bom_csv(&bom);
    assert!(csv.starts_with("Designators,Quantity"));
    let pnp = fab::pick_place(&board, false);
    assert!(pnp.lines().count() > 20);
    assert!(pnp.lines().skip(1).all(|l| l.ends_with(",top")));
}

#[test]
fn review_suites_find_and_fix() {
    let pool = standard_pool();
    let mut d = Design::new("review");
    {
        let mut b = DesignBuilder::new(&mut d, "main");
        b.global("GND", None);
        b.component("J1", "USB4105");
        b.component("U1", "MCU-QFN32");
        b.component("U2", "TMP117");
        b.component("U3", "TPS62130");
        b.component("L1", "L1210-4U7");
        b.component("C1", "C0805-10U");
        b.component("C2", "C0805-10U");
        b.component("R1", "R0402-10K");
        b.component("R2", "R0402-10K");
        b.component("Y1", "XTAL-3225");
        b.component("C3", "C0402-100N");
        b.component("C4", "C0402-100N");
        b.wire(
            "VBUS",
            &[("J1", "VBUS"), ("U3", "VIN"), ("U3", "EN"), ("C1", "1")],
        );
        b.wire("SW", &[("U3", "SW"), ("L1", "1")]);
        b.wire(
            "+3V3",
            &[
                ("L1", "2"),
                ("C2", "1"),
                ("R1", "1"),
                ("U3", "VOS"),
                ("U1", "VDD"),
                ("U2", "V+"),
            ],
        );
        b.wire("FB", &[("U3", "FB"), ("R1", "2"), ("R2", "1")]);
        b.wire(
            "GND",
            &[
                ("U3", "GND"),
                ("C1", "2"),
                ("C2", "2"),
                ("R2", "2"),
                ("J1", "GND"),
                ("U1", "VSS"),
                ("U2", "GND"),
                ("C3", "2"),
                ("C4", "2"),
            ],
        );
        b.wire("SCL", &[("U1", "SCL"), ("U2", "SCL")]);
        b.wire("SDA", &[("U1", "SDA"), ("U2", "SDA")]);
        b.wire("OSC_IN", &[("U1", "OSC_IN"), ("Y1", "1"), ("C3", "1")]);
        b.wire("OSC_OUT", &[("U1", "OSC_OUT"), ("Y1", "2"), ("C4", "1")]);
        b.connect(vec![
            Endpoint::pin("J1", "DP"),
            Endpoint::pin("U1", "USB_DP"),
            Endpoint::net("USB_P"),
        ]);
        b.connect(vec![
            Endpoint::pin("J1", "DN"),
            Endpoint::pin("U1", "USB_DN"),
            Endpoint::net("USB_N"),
        ]);
        b.diff_pair("USB", "USB_P", "USB_N", None);
    }
    let nl = elaborate(&d, &pool, &ElabOptions::default(), None);
    assert!(nl.diagnostics.is_empty(), "{:#?}", nl.diagnostics);
    let o = ReviewOptions::default();
    let f = review::run_all(&nl, &pool, &o);
    let rules: Vec<(&str, &str)> = f
        .iter()
        .map(|x| (x.suite.as_str(), x.rule.as_str()))
        .collect();
    for want in [
        ("decoupling", "cap-per-power-pin"),
        ("i2c", "pull-up"),
        ("crystal", "load-match"),
        ("straps", "reset-pull-up"),
        ("straps", "boot-pull-down"),
        ("usb-c", "cc-rd"),
        ("switcher", "output-voltage"),
    ] {
        assert!(rules.contains(&want), "missing {want:?} in {rules:?}");
    }
    // 10k/10k with 0.8 V reference sets 1.6 V on a 3.3 V rail.
    let sv = f.iter().find(|x| x.rule == "output-voltage").unwrap();
    assert!(sv.message.contains("1.600 V"), "{}", sv.message);
    assert!(f
        .iter()
        .filter(|x| x.rule == "cc-rd")
        .all(|x| x.severity == Severity::Error));
    // Apply every offered fix; the findings they address disappear.
    let fixes: Vec<_> = f.iter().filter_map(|x| x.fix.clone()).collect();
    assert!(fixes.len() >= 8, "{}", fixes.len());
    for e in &fixes {
        review::apply(&mut d, e).unwrap();
    }
    let nl2 = elaborate(&d, &pool, &ElabOptions::default(), None);
    assert!(nl2.diagnostics.is_empty(), "{:#?}", nl2.diagnostics);
    let f2 = review::run_all(&nl2, &pool, &o);
    let left: Vec<(&str, &str)> = f2
        .iter()
        .map(|x| (x.suite.as_str(), x.rule.as_str()))
        .collect();
    for gone in [
        ("i2c", "pull-up"),
        ("straps", "reset-pull-up"),
        ("straps", "boot-pull-down"),
        ("usb-c", "cc-rd"),
        ("switcher", "output-voltage"),
        ("decoupling", "cap-per-power-pin"),
        ("crystal", "load-match"),
    ] {
        assert!(!left.contains(&gone), "{gone:?} still reported: {f2:#?}");
    }
}

#[test]
fn fea_contract_regions_and_power() {
    let (pool, _, nl, board) = small();
    let mut op = BTreeMap::new();
    op.insert("U1".to_string(), 0.12);
    let src = fea::pcb_source(&board, Some(&nl), Some(&pool), &op);
    assert_eq!(src.region, "pcb/large/board");
    assert_eq!(
        src.layers
            .iter()
            .filter(|l| l.kind == fea::LayerKind::Copper)
            .count(),
        4
    );
    assert!(src
        .layers
        .windows(2)
        .all(|w| (w[0].z_bottom - w[1].z_top).abs() < 1e-12));
    assert!((src.layers.last().unwrap().z_bottom - src.thickness).abs() < 1e-9);
    let gnd = src
        .copper
        .iter()
        .find(|c| c.region == "pcb/large/copper/In1.Cu/net/GND")
        .expect("GND plane region");
    assert!(gnd.area.area() > 0.0);
    assert_eq!(gnd.id, fea::canonical_id("pcb/large/copper/In1.Cu/net/GND"));
    let inner = src.layers.iter().find(|l| l.name == "In1.Cu").unwrap();
    assert!(
        inner.copper_fraction.unwrap() > 0.5,
        "plane layer is mostly copper"
    );
    assert!(src
        .layers
        .iter()
        .any(|l| l.region.starts_with("pcb/large/dielectric/")));
    assert!(src.vias.iter().any(|v| v.region == "pcb/large/via/GND"));
    let fpga = src
        .components
        .iter()
        .find(|c| c.designator == "U9000")
        .unwrap();
    assert_eq!(fpga.power, Some(2.5));
    assert_eq!(fpga.power_source, fea::PowerSource::Authored);
    assert_eq!(fpga.theta_jb, Some(9.0));
    assert_eq!(src.power_map.get("pcb/large/component/U9000"), Some(&2.5));
    let u1 = src
        .components
        .iter()
        .find(|c| c.designator == "U1")
        .unwrap();
    assert_eq!(u1.power_source, fea::PowerSource::OperatingPoint);
    assert_eq!(fea::segment("/power/VOUT"), "%2Fpower%2FVOUT");
    // Deterministic.
    assert_eq!(fea::pcb_source(&board, Some(&nl), Some(&pool), &op), src);
}
