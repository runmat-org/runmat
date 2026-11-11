//! The `.circuit` front end: design → print → parse gives the same design, with a source key
//! on every declaration.

use runmat_circuit::fixture::{eco_edited_design, eco_synced_design, large_design, LargeParams};
use runmat_circuit::ir::*;
use runmat_circuit::syntax::{parse_circuit, print_circuit};
use runmat_rules::{BuiltinUnits, Quantity, SourceKey};

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

/// The design with every source key cleared (they say where text was, so they differ).
fn without_sources(mut design: Design) -> Design {
    let clear = |params: &mut std::collections::BTreeMap<String, Param>| {
        for param in params.values_mut() {
            param.source = SourceKey::default();
        }
    };
    clear(&mut design.params);
    for class in design.net_classes.values_mut() {
        class.source = SourceKey::default();
        clear(&mut class.params);
    }
    for variant in design.variants.values_mut() {
        for fitting in variant.overrides.values_mut() {
            if let Fitting::Alternate { params, .. } = fitting {
                clear(params);
            }
        }
    }
    for sheet in design.sheets.values_mut() {
        clear(&mut sheet.params);
        for port in &mut sheet.ports {
            port.source = SourceKey::default();
        }
        for component in &mut sheet.components {
            component.source = SourceKey::default();
            clear(&mut component.params);
            for fitting in component.variants.values_mut() {
                if let Fitting::Alternate { params, .. } = fitting {
                    clear(params);
                }
            }
        }
        for net in &mut sheet.nets {
            net.source = SourceKey::default();
            clear(&mut net.params);
        }
        for bus in &mut sheet.buses {
            bus.source = SourceKey::default();
        }
        for harness in &mut sheet.harnesses {
            harness.source = SourceKey::default();
        }
        for instance in &mut sheet.instances {
            instance.source = SourceKey::default();
        }
        for connection in &mut sheet.connections {
            connection.source = SourceKey::default();
        }
        for pair in &mut sheet.diff_pairs {
            pair.source = SourceKey::default();
        }
    }
    design
}

fn round_trip(design: &Design) -> String {
    let text = print_circuit(design).unwrap();
    let parsed = parse_circuit("board.circuit", &text);
    assert!(
        parsed.diagnostics.is_empty(),
        "{text}\n{:#?}",
        parsed.diagnostics
    );
    let back = parsed.design.unwrap();
    assert_eq!(
        without_sources(back.clone()),
        without_sources(design.clone()),
        "{text}"
    );
    assert_eq!(print_circuit(&back).unwrap(), text);
    text
}

#[test]
fn designs_round_trip_through_text() {
    let text = round_trip(&sensor_node());
    assert!(text.contains("circuit \"sensor-node\" {"), "{text}");
    assert!(
        text.contains("        component U1 = { part = TPS62130 }"),
        "{text}"
    );
    assert!(
        text.contains("        connect = [U1.VIN, net.VIN]") || text.contains("connect = ["),
        "{text}"
    );
    round_trip(&large_design(&LargeParams::default()));
    round_trip(&eco_synced_design());
    round_trip(&eco_edited_design());
}

#[test]
fn parameters_tolerances_and_variants_read_back_exactly() {
    let mut design = sensor_node();
    let units = BuiltinUnits;
    let q = |text: &str| runmat_rules::parse_quantity(text, &units).unwrap();
    let sheet = design.sheets.get_mut("main").unwrap();
    let r = sheet
        .components
        .iter_mut()
        .find(|c| c.name == "R?a")
        .unwrap();
    r.params.insert(
        "value".into(),
        Param::quantity(q("10 kohm"), Tolerance::pm_percent(1.0)),
    );
    r.params.insert(
        "tc".into(),
        Param::quantity(
            q("100 ppm"),
            Tolerance::Relative {
                minus: 0.07,
                plus: 0.03,
            },
        ),
    );
    r.params.insert(
        "v".into(),
        Param::quantity(
            q("3.3 V"),
            Tolerance::Absolute {
                minus: 0.1,
                plus: 0.2,
            },
        ),
    );
    r.params.insert(
        "t".into(),
        Param::quantity(
            q("25 degC"),
            Tolerance::Range {
                min: 270.0,
                max: 330.0,
            },
        ),
    );
    r.params
        .insert("note".into(), Param::text("do not fit for EMC"));
    r.params.insert(
        "dnp".into(),
        Param {
            value: ParamValue::Bool(true),
            source: SourceKey::default(),
            call: None,
            written: None,
        },
    );
    r.power = Some(Quantity::new(0.0625, runmat_rules::Dim::POWER));
    r.variants.insert("lite".into(), Fitting::NotFitted);
    design.variants.insert(
        "lite".into(),
        VariantDef {
            name: "lite".into(),
            base: Some("full".into()),
            overrides: [(
                "/power/U1".to_string(),
                Fitting::Alternate {
                    part: Some(PartRef::Entity {
                        entity: "buck".into(),
                        package: Some("QFN-16".into()),
                    }),
                    params: Default::default(),
                },
            )]
            .into(),
        },
    );
    let text = round_trip(&design);
    assert!(text.contains("value = 10000 ohm ± 1 %"), "{text}");
    assert!(text.contains("v = 3.3 V ± [0.2 V, 0.1 V]"), "{text}");
}

#[test]
fn hand_written_circuits_read_with_any_known_unit_and_keep_source_keys() {
    let text = "circuit c {\n    sheet main {\n        component R1 = { part = \"R0402\", value = 10 kohm ± 5 % }\n        net GND = { global = true }\n        connect = [\"R1.1\", net.GND]\n    }\n}\n";
    let parsed = parse_circuit("c.circuit", text);
    assert!(parsed.diagnostics.is_empty(), "{:#?}", parsed.diagnostics);
    let design = parsed.design.unwrap();
    let main = &design.sheets["main"];
    let component = &main.components[0];
    assert_eq!(
        component.params["value"].as_quantity().unwrap().value,
        10000.0
    );
    let key = &component.source;
    assert_eq!(
        &text[key.start as usize..key.end as usize],
        "component R1 = { part = \"R0402\", value = 10 kohm ± 5 % }"
    );
    assert_eq!(main.connections[0].ends[0], Endpoint::pin("R1", "1"));
    assert!(main.nets[0].global);
}

#[test]
fn misplaced_statements_are_diagnosed() {
    let parsed = parse_circuit("c.circuit", "circuit c {\n    sheet main {\n        widget w = 1\n        port p : sideways\n    }\n}\n");
    assert!(parsed.design.is_none());
    let messages: Vec<&str> = parsed
        .diagnostics
        .iter()
        .map(|d| d.message.as_str())
        .collect();
    assert!(
        messages
            .iter()
            .any(|m| m.contains("`widget` does not start")),
        "{messages:?}"
    );
    assert!(
        messages.iter().any(|m| m.contains("a port is")),
        "{messages:?}"
    );
}

#[test]
fn use_lines_read_print_and_resolve() {
    let power = "runmat 0.7\nstd 1\n\ncircuit power {\n    net_class rail = { trace_width = 0.5 mm }\n    interface usb2 = { members = [DP, DN] }\n    sheet buck {\n        port VIN : power\n        instance ldo = { sheet = ldo }\n    }\n    sheet ldo {\n        port VIN : power\n    }\n}\n";
    let main = "runmat 0.7\nstd 1\n\nuse sheet buck from \"./power\" as reg\nuse interface usb2 from \"./power\"\nuse library passives from \"lib/passives\" @ \"R3\"\nuse \"lib/connectors\" as conn\n\ncircuit ctl {\n    sheet main {\n        instance r = { sheet = reg }\n    }\n}\n";
    let parsed = parse_circuit("ctl.circuit", main);
    assert!(parsed.diagnostics.is_empty(), "{:#?}", parsed.diagnostics);
    let mut design = parsed.design.unwrap();
    assert_eq!(design.uses.len(), 4);
    assert_eq!(design.uses[0].local_name(), Some("reg"));
    assert_eq!(design.uses[2].revision.as_deref(), Some("R3"));
    // The use lines print back and read the same.
    let printed = print_circuit(&design).unwrap();
    assert!(printed.contains("use sheet buck from \"./power\" as reg\n"));
    assert_eq!(
        parse_circuit("ctl.circuit", &printed).design.unwrap(),
        design
    );
    // Resolving brings in the used sheet (and the sheet it instances) and the interface.
    let used = parse_circuit("power.circuit", power).design.unwrap();
    runmat_circuit::syntax::resolve_uses(&mut design, &|u: &runmat_circuit::ir::CircuitUse| {
        (u.from == "./power")
            .then(|| used.clone())
            .ok_or_else(|| "unread".to_string())
    })
    .unwrap();
    assert!(design.sheets.contains_key("reg") && design.sheets.contains_key("ldo"));
    assert!(design.interfaces.contains_key("usb2"));
    assert!(!design.net_classes.contains_key("rail"));
    // Errors: an unknown kind, a name used twice, a folder without a name, a missing file.
    for (line, message) in [
        ("use part R1 from \"./x\"", "not one of them"),
        (
            "use sheet a from \"./x\"\nuse sheet a from \"./y\"",
            "already used",
        ),
        ("use \"lib/x\"", "under a name"),
        (
            "use circuit p from \"./p.circuit\"",
            "name the design: write `use circuit p as …`",
        ),
    ] {
        let p = parse_circuit("ctl.circuit", &format!("{line}\ncircuit ctl {{\n}}\n"));
        assert!(
            p.diagnostics.iter().any(|d| d.message.contains(message)),
            "{line}: {:#?}",
            p.diagnostics
        );
    }
    let mut missing = parse_circuit("ctl.circuit", "use circuit p\ncircuit ctl {\n}\n")
        .design
        .unwrap();
    assert!(runmat_circuit::syntax::resolve_uses(
        &mut missing,
        &|_: &runmat_circuit::ir::CircuitUse| Err("unread".into())
    )
    .is_err());
}
