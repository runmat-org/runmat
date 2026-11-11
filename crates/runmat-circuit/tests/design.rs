//! Logical design: elaboration across hierarchy, channels, buses and harnesses; annotation;
//! variants; stable ids and ECO renames; ERC.

use runmat_circuit::elab::{elaborate, ElabOptions, FitState, Netlist, Severity};
use runmat_circuit::erc::{self, ErcOptions};
use runmat_circuit::fixture::{self, standard_pool, LargeParams};
use runmat_circuit::ids::{DerivedUuids, IdChange, LockSection, NamePath};
use runmat_circuit::ir::*;
use runmat_circuit::library::Pool;

/// A USB-C powered sensor node: USB-C in, buck to 3V3, MCU, I²C temperature sensor.
fn sensor_node() -> Design {
    let mut d = Design::new("sensor-node");
    d.net_classes.insert(
        "power".into(),
        NetClass {
            name: "power".into(),
            params: Default::default(),
            source: Default::default(),
        },
    );
    d.interfaces.insert(
        "usb2".into(),
        InterfaceDef {
            name: "usb2".into(),
            members: vec!["DP".into(), "DN".into()],
            diff_pairs: vec![("DP".into(), "DN".into())],
        },
    );
    {
        let mut b = DesignBuilder::new(&mut d, "power");
        b.port("VIN", PortDir::Power);
        b.port("VOUT", PortDir::Power);
        b.global("GND", Some("power"));
        b.component("U1", "TPS62130");
        b.component("L1", "L1210-4U7");
        b.component("C1", "C0805-10U");
        b.component("C2", "C0805-10U");
        b.component("R1", "R0402-10K");
        b.component("R2", "R0402-10K");
        b.wire("VIN", &[("U1", "VIN"), ("U1", "EN"), ("C1", "1")]);
        b.wire("SW", &[("U1", "SW"), ("L1", "1")]);
        b.wire(
            "VOUT",
            &[("L1", "2"), ("C2", "1"), ("R1", "1"), ("U1", "VOS")],
        );
        b.wire("FB", &[("U1", "FB"), ("R1", "2"), ("R2", "1")]);
        b.wire(
            "GND",
            &[("U1", "GND"), ("C1", "2"), ("C2", "2"), ("R2", "2")],
        );
        b.no_erc("U1", "PG");
        b.no_erc("U1", "SS");
        b.connect(vec![Endpoint::Port("VIN".into()), Endpoint::net("VIN")]);
        b.connect(vec![Endpoint::Port("VOUT".into()), Endpoint::net("VOUT")]);
    }
    let mut b = DesignBuilder::new(&mut d, "main");
    b.global("GND", Some("power"));
    b.net("VBUS").class = Some("power".into());
    b.net("+3V3").class = Some("power".into());
    b.design
        .sheets
        .get_mut("main")
        .unwrap()
        .harnesses
        .push(HarnessDecl {
            name: "USB".into(),
            interface: "usb2".into(),
            source: Default::default(),
        });
    b.component("J1", "USB4105");
    b.component("U2", "MCU-QFN32");
    b.component("U3", "TMP117");
    b.component("R?", "R0402-10K").name = "R?a".into();
    b.component("R?", "R0402-10K").name = "R?b".into();
    b.component("C?", "C0402-100N").name = "C?a".into();
    let inst = b.instance("pwr", "power", None);
    inst.bindings
        .push(("VIN".into(), Binding::Shared(Endpoint::net("VBUS"))));
    inst.bindings
        .push(("VOUT".into(), Binding::Shared(Endpoint::net("+3V3"))));
    b.wire("VBUS", &[("J1", "VBUS")]);
    b.wire(
        "GND",
        &[
            ("J1", "GND"),
            ("J1", "SHIELD"),
            ("U2", "VSS"),
            ("U3", "GND"),
            ("U3", "ADD0"),
            ("C?a", "2"),
        ],
    );
    b.wire(
        "+3V3",
        &[
            ("U2", "VDD"),
            ("U3", "V+"),
            ("R?a", "1"),
            ("R?b", "1"),
            ("C?a", "1"),
        ],
    );
    b.wire("SCL", &[("U2", "SCL"), ("U3", "SCL"), ("R?a", "2")]);
    b.wire("SDA", &[("U2", "SDA"), ("U3", "SDA"), ("R?b", "2")]);
    b.connect(vec![
        Endpoint::HarnessMember {
            harness: "USB".into(),
            member: "DP".into(),
        },
        Endpoint::pin("J1", "DP"),
        Endpoint::pin("U2", "USB_DP"),
    ]);
    b.connect(vec![
        Endpoint::HarnessMember {
            harness: "USB".into(),
            member: "DN".into(),
        },
        Endpoint::pin("J1", "DN"),
        Endpoint::pin("U2", "USB_DN"),
    ]);
    for p in ["CC1", "CC2", "SBU1", "SBU2"] {
        b.no_erc("J1", p);
    }
    b.no_erc("U3", "ALERT");
    for p in ["NRST", "BOOT0", "OSC_IN", "OSC_OUT"] {
        b.no_erc("U2", p);
    }
    for i in 0..20 {
        b.no_erc("U2", &format!("PA{i}"));
    }
    d
}

fn elab(d: &Design, pool: &Pool) -> Netlist {
    elaborate(d, pool, &ElabOptions::default(), None)
}

#[test]
fn sensor_node_elaborates() {
    let pool = standard_pool();
    assert!(pool.validate().is_empty(), "{:?}", pool.validate());
    let nl = elab(&sensor_node(), &pool);
    assert!(nl.diagnostics.is_empty(), "{:#?}", nl.diagnostics);
    // The child's VIN port joins the parent VBUS net; the root name wins, the port is an alias.
    let (_, vbus) = nl.net("VBUS").unwrap();
    assert!(
        vbus.aliases.iter().any(|a| a == "/pwr/VIN"),
        "{:?}",
        vbus.aliases
    );
    assert_eq!(nl.net_of("U1", "VIN").unwrap().name, "VBUS");
    assert_eq!(nl.net_of("U1", "SW").unwrap().name, "/pwr/SW");
    // Global GND joins across sheets.
    assert_eq!(nl.net_of("U1", "GND").unwrap().name, "GND");
    assert_eq!(nl.net_of("J1", "GND").unwrap().name, "GND");
    assert_eq!(nl.net("GND").unwrap().1.class.as_deref(), Some("power"));
    // Multi-pad pins: the buck's GND pin maps to three pads; the net has the pin once.
    let (_, u1) = nl.component("U1").unwrap();
    assert_eq!(u1.pin("GND").unwrap().pads, vec!["9", "10", "17"]);
    // Auto-annotation: R? → R3, R4 (R1, R2 are taken in the power sheet), C? → C3.
    let mut refs: Vec<&str> = nl
        .components
        .iter()
        .map(|c| c.designator.as_str())
        .collect();
    refs.sort();
    assert_eq!(
        refs,
        vec!["C1", "C2", "C3", "J1", "L1", "R1", "R2", "R3", "R4", "U1", "U2", "U3"]
    );
    // Harness interface gives the differential pair.
    assert_eq!(nl.diff_pairs.len(), 1);
    let dp = &nl.diff_pairs[0];
    assert_eq!(nl.nets[dp.p].name, "USB.DP");
    assert_eq!(nl.nets[dp.n].name, "USB.DN");
}

#[test]
fn erc_power_and_flags() {
    let pool = standard_pool();
    let nl = elab(&sensor_node(), &pool);
    let f = erc::run(&nl, &ErcOptions::default());
    let codes: Vec<(&str, &str)> = f
        .iter()
        .map(|d| (d.code.as_str(), d.item.as_str()))
        .collect();
    // VBUS comes from a passive connector pin: the buck's power input is not driven.
    assert!(codes.contains(&("power-not-driven", "VBUS")), "{codes:?}");
    // GND has only power inputs and passives too.
    assert!(codes.contains(&("power-not-driven", "GND")), "{codes:?}");
    // +3V3 is driven? No: the buck output is SW (power out) through an inductor. Flag it.
    let mut o = ErcOptions::default();
    o.power_flags
        .extend(["VBUS".to_string(), "GND".to_string(), "+3V3".to_string()]);
    let f = erc::run(&nl, &o);
    assert!(f.iter().all(|d| d.code != "power-not-driven"), "{f:#?}");
    assert!(f.iter().all(|d| d.severity != Severity::Error), "{f:#?}");
    // Every finding names an item.
    assert!(f.iter().all(|d| !d.item.is_empty()));
}

#[test]
fn erc_matrix_drivers_and_unconnected() {
    let mut pool = standard_pool();
    use runmat_circuit::library::{make, PinType};
    let drv = make::simple_entity(
        &mut pool,
        "driver",
        "U",
        &[
            ("O", PinType::Output),
            ("I", PinType::Input),
            ("NC", PinType::NotConnected),
            ("OC", PinType::OpenCollector),
        ],
    );
    let pkg = pool.package("SOT-23-6_IPCN").unwrap().uuid;
    make::part(
        &mut pool,
        "DRV",
        "Generic",
        drv,
        pkg,
        &[("O", &["1"]), ("I", &["2"]), ("NC", &["3"]), ("OC", &["4"])],
    );
    let mut d = Design::new("erc");
    let mut b = DesignBuilder::new(&mut d, "main");
    b.component("U1", "DRV");
    b.component("U2", "DRV");
    b.component("U3", "DRV");
    b.wire("BUS", &[("U1", "O"), ("U2", "O")]);
    b.wire("MIX", &[("U3", "O"), ("U3", "OC")]);
    b.wire("FLOAT", &[("U1", "I"), ("U2", "I")]);
    b.wire("BAD_NC", &[("U1", "NC"), ("U3", "I")]);
    let nl = elab(&d, &pool);
    let f = erc::run(&nl, &ErcOptions::default());
    let has = |code: &str, item: &str| f.iter().any(|d| d.code == code && d.item == item);
    assert!(has("multiple-drivers", "BUS"));
    assert!(has("pin-conflict", "MIX"), "{f:#?}");
    assert!(has("input-not-driven", "FLOAT"));
    assert!(has("no-connect-connected", "U1.NC (3)"), "{f:#?}");
    assert!(
        !has("pin-unconnected", "U2.NC (3)"),
        "NC pins may stay unconnected"
    );
    assert!(has("pin-unconnected", "U2.OC (4)"));
    let open = f
        .iter()
        .find(|d| d.code == "pin-unconnected" && d.item == "U2.OC (4)")
        .unwrap();
    assert_eq!(
        open.message,
        "open_collector pin U2.OC (4) is unconnected; mark it no-connect if intended"
    );
    let md = f.iter().find(|d| d.code == "multiple-drivers").unwrap();
    assert_eq!(md.related, vec!["U1.O (1)", "U2.O (1)"]);
}

#[test]
fn channels_buses_and_paths() {
    let pool = standard_pool();
    let p = LargeParams {
        channels: 12,
        ..Default::default()
    };
    let d = fixture::large_design(&p);
    let nl = elab(&d, &pool);
    assert!(nl.diagnostics.is_empty(), "{:#?}", nl.diagnostics);
    // 12 channels × 7 parts + 1 FPGA + 2 headers.
    assert_eq!(nl.components.len(), 12 * 7 + 1 + 2);
    assert_eq!(nl.sheets.iter().filter(|(p, _)| p.depth() == 1).count(), 12);
    // Bus-indexed binding: channel 5's input is AIN5, and its output feeds FPGA IO5.
    let rs = nl
        .components
        .iter()
        .find(|c| c.path == NamePath::parse("/ch[5]/RS"))
        .unwrap();
    let n = rs.pin("1").unwrap().net.unwrap();
    assert_eq!(nl.nets[n].name, "AIN5");
    assert!(nl.nets[n].aliases.iter().any(|a| a == "/ch[5]/IN"));
    let rf = nl
        .components
        .iter()
        .find(|c| c.path == NamePath::parse("/ch[5]/RF"))
        .unwrap();
    assert_eq!(nl.nets[rf.pin("2").unwrap().net.unwrap()].name, "ADC5");
    assert!(nl.net_of("U9000", "IO5").unwrap().name == "ADC5");
    // Local nets carry their sheet path.
    assert!(nl.net("/ch[5]/INF").is_some());
    // Designators unique, entity prefixes.
    let mut seen = std::collections::BTreeSet::new();
    for c in &nl.components {
        assert!(
            seen.insert(c.designator.clone()),
            "duplicate {}",
            c.designator
        );
    }
    assert!(rs.designator.starts_with('R'));
    // Gate-qualified pins and shared power pins.
    let u = nl
        .components
        .iter()
        .find(|c| c.path == NamePath::parse("/ch[0]/U"))
        .unwrap();
    assert_eq!(nl.nets[u.pin("A.OUT").unwrap().net.unwrap()].name, "ADC0");
    let erc = erc::run(
        &nl,
        &ErcOptions {
            power_flags: ["GND", "+5V", "-5V"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            ..Default::default()
        },
    );
    assert!(
        erc.iter().all(|d| d.severity != Severity::Error),
        "{erc:#?}"
    );
}

#[test]
fn variants_fit_and_alternate() {
    let pool = standard_pool();
    let p = LargeParams {
        channels: 3,
        ..Default::default()
    };
    let mut d = fixture::large_design(&p);
    let mut lite = VariantDef {
        name: "lite".into(),
        ..Default::default()
    };
    lite.overrides
        .insert("/ch[1]/CF".into(), Fitting::NotFitted);
    lite.overrides.insert(
        "/ch[2]/RF".into(),
        Fitting::Alternate {
            part: None,
            params: [("resistance".to_string(), Param::text("22k"))].into(),
        },
    );
    d.variants.insert("lite".into(), lite);
    d.variants.insert(
        "lite-plus".into(),
        VariantDef {
            name: "lite-plus".into(),
            base: Some("lite".into()),
            ..Default::default()
        },
    );
    let nl = elaborate(
        &d,
        &pool,
        &ElabOptions {
            variant: Some("lite-plus".into()),
        },
        None,
    );
    assert!(nl.diagnostics.is_empty(), "{:#?}", nl.diagnostics);
    let cf = nl
        .components
        .iter()
        .find(|c| c.path == NamePath::parse("/ch[1]/CF"))
        .unwrap();
    assert_eq!(cf.fitting, FitState::NotFitted);
    let rf = nl
        .components
        .iter()
        .find(|c| c.path == NamePath::parse("/ch[2]/RF"))
        .unwrap();
    assert_eq!(rf.fitting, FitState::Alternate);
    assert_eq!(rf.params["resistance"].as_text(), Some("22k"));
    let bad = elaborate(
        &d,
        &pool,
        &ElabOptions {
            variant: Some("nope".into()),
        },
        None,
    );
    assert!(bad.diagnostics.iter().any(|x| x.code == "unknown-variant"));
}

#[test]
fn lock_keeps_ids_and_designators_across_edits() {
    let pool = standard_pool();
    let d = sensor_node();
    let nl = elab(&d, &pool);
    let mut lock = LockSection::new();
    let mut src = DerivedUuids { namespace: 42 };
    let first = lock.reconcile(&nl.lock_items(), &mut src);
    assert!(first.iter().all(|c| matches!(c, IdChange::New { .. })));
    let u1 = lock.uuid_of(&NamePath::parse("/pwr/U1")).unwrap();
    let r3 = lock.entries[&NamePath::parse("/R?a")].designator.clone();
    assert_eq!(r3.as_deref(), Some("R3"));

    // Rename the power sheet instance pwr → buck (explicit), and add a resistor whose `R?`
    // would renumber the others without the lock.
    let mut d2 = d.clone();
    let main = d2.sheets.get_mut("main").unwrap();
    main.instances[0].name = "buck".into();
    main.instances[0].renamed_from = Some("pwr".into());
    main.components.insert(
        0,
        ComponentDecl::new("R?new", PartRef::Part("R0402-10K".into())),
    );
    let nl2 = elaborate(&d2, &pool, &ElabOptions::default(), Some(&lock));
    // The lock keeps R3/R4 on their paths; the new part takes the next free number.
    let ra = nl2
        .components
        .iter()
        .find(|c| c.path == NamePath::parse("/R?a"))
        .unwrap();
    assert_eq!(ra.designator, "R3");
    let rn = nl2
        .components
        .iter()
        .find(|c| c.path == NamePath::parse("/R?new"))
        .unwrap();
    assert_eq!(rn.designator, "R5");
    // Components inside the renamed sheet keep their UUIDs: an explicit ECO rename that
    // follows the sheet instance.
    let changes = lock.reconcile(&nl2.lock_items(), &mut src);
    let renamed_u1 = changes
        .iter()
        .find(|c| matches!(c, IdChange::Renamed { to, .. } if *to == NamePath::parse("/buck/U1")));
    match renamed_u1 {
        Some(IdChange::Renamed { uuid, inferred, .. }) => {
            assert_eq!(*uuid, u1);
            assert!(!inferred);
        }
        other => panic!("{other:?}\n{changes:#?}"),
    }
    assert!(changes
        .iter()
        .any(|c| matches!(c, IdChange::New { path, .. } if *path == NamePath::parse("/R?new"))));
    // The lock text is canonical.
    let t = lock.to_text();
    assert_eq!(LockSection::from_text(&t).unwrap().to_text(), t);
}

#[test]
fn diagnostics_for_bad_references() {
    let pool = standard_pool();
    let mut d = Design::new("bad");
    let mut b = DesignBuilder::new(&mut d, "main");
    b.component("U1", "OPA4171");
    b.component("X1", "NO-SUCH-PART");
    b.wire("N1", &[("U1", "IN+")]);
    b.wire("N2", &[("U1", "Q.OUT")]);
    b.wire("N3", &[("U9", "1")]);
    b.bus("D", &["D0", "D1", "D2"]);
    b.bus("E", &["E0", "E1"]);
    b.connect(vec![Endpoint::net("D"), Endpoint::net("E")]);
    let nl = elab(&d, &pool);
    let codes: Vec<&str> = nl.diagnostics.iter().map(|x| x.code.as_str()).collect();
    assert!(codes.contains(&"unknown-part"));
    assert!(codes.contains(&"unknown-component"));
    assert!(codes.contains(&"bus-width-mismatch"));
    let amb = nl
        .diagnostics
        .iter()
        .find(|x| x.code == "unknown-pin" && x.message.contains("ambiguous"))
        .unwrap();
    assert!(amb.message.contains("A, B, C, D"), "{}", amb.message);
}

#[test]
fn elaboration_is_deterministic() {
    let pool = standard_pool();
    let d = fixture::large_design(&LargeParams {
        channels: 40,
        ..Default::default()
    });
    let a = elab(&d, &pool);
    let b = elab(&d.clone(), &pool);
    assert_eq!(a, b);
    let names: Vec<&str> = a.nets.iter().map(|n| n.name.as_str()).collect();
    let mut sorted = names.clone();
    sorted.sort();
    assert_eq!(names, sorted);
}
