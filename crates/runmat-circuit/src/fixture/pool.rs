//! The standard fixture pool.

use crate::geom::{mm, Point};
use crate::ir::*;
use crate::library::footprint::{self, Density};
use crate::library::{make, Entity, Gate, PinType, Pool, Unit, UnitPin};
use runmat_rules::{Dim, Quantity};

fn unit(pool: &mut Pool, name: &str, pins: &[(&str, PinType)]) -> crate::ids::Uuid {
    pool.add_unit(Unit {
        uuid: make::uuid("unit", name),
        name: name.into(),
        pins: pins
            .iter()
            .map(|(n, k)| UnitPin {
                name: n.to_string(),
                kind: *k,
                swap_group: 0,
                alternates: vec![],
            })
            .collect(),
    })
}

/// The standard fixture pool. Part names (MPNs) used by the fixtures:
/// `R0402-10K`, `C0402-100N`, `C0805-10U`, `L1210-4U7`, `LED0603-G`, `OPA4171`, `TPS62130`,
/// `USB4105`, `TMP117`, `XTAL-3225`, `MCU-QFN32`, `FPGA-BGA256`, `HDR-2x05`.
pub fn standard_pool() -> Pool {
    use PinType::*;
    let mut p = Pool::new();
    let two = [("1", Passive), ("2", Passive)];
    let r = make::simple_entity(&mut p, "R", "R", &two);
    let c = make::simple_entity(&mut p, "C", "C", &two);
    let l = make::simple_entity(&mut p, "L", "L", &two);
    let led = make::simple_entity(&mut p, "LED", "D", &[("A", Passive), ("K", Passive)]);
    let pk = |p: &mut Pool, s: &str| p.add_package(footprint::from_string(s).unwrap());
    let r0402 = pk(&mut p, "R0402");
    let c0402 = pk(&mut p, "C0402");
    let c0805 = pk(&mut p, "C0805");
    let l1210 = pk(&mut p, "L1210");
    let led0603 = pk(&mut p, "LED0603");
    let set_q = |p: &mut Pool, part: crate::ids::Uuid, k: &str, v: f64, d: Dim, tol: f64| {
        p.parts.get_mut(&part).unwrap().params.insert(
            k.into(),
            Param::quantity(Quantity::new(v, d), Tolerance::pm_percent(tol)),
        );
    };
    let pr = make::part_by_name(&mut p, "R0402-10K", "Generic", r, r0402);
    set_q(&mut p, pr, "resistance", 10e3, Dim::RESISTANCE, 1.0);
    set_q(&mut p, pr, "power_rating", 0.0625, Dim::POWER, 0.0);
    let pc = make::part_by_name(&mut p, "C0402-100N", "Generic", c, c0402);
    set_q(&mut p, pc, "capacitance", 100e-9, Dim::CAPACITANCE, 10.0);
    set_q(&mut p, pc, "voltage_rating", 16.0, Dim::VOLTAGE, 0.0);
    let pc2 = make::part_by_name(&mut p, "C0805-10U", "Generic", c, c0805);
    set_q(&mut p, pc2, "capacitance", 10e-6, Dim::CAPACITANCE, 20.0);
    set_q(&mut p, pc2, "voltage_rating", 25.0, Dim::VOLTAGE, 0.0);
    let pl = make::part_by_name(&mut p, "L1210-4U7", "Generic", l, l1210);
    set_q(&mut p, pl, "inductance", 4.7e-6, Dim::INDUCTANCE, 20.0);
    make::part(
        &mut p,
        "LED0603-G",
        "Generic",
        led,
        led0603,
        &[("A", &["1"]), ("K", &["2"])],
    );

    // Quad op-amp: four gates plus a power gate.
    let ua = unit(
        &mut p,
        "opamp",
        &[("IN+", Input), ("IN-", Input), ("OUT", Output)],
    );
    let upw = unit(&mut p, "opamp-power", &[("V+", PowerIn), ("V-", PowerIn)]);
    let quad = p.add_entity(Entity {
        uuid: make::uuid("entity", "quad-opamp"),
        name: "quad-opamp".into(),
        prefix: "U".into(),
        gates: ["A", "B", "C", "D"]
            .iter()
            .map(|g| Gate {
                name: g.to_string(),
                unit: ua,
                swap_group: 1,
            })
            .chain(std::iter::once(Gate {
                name: "P".into(),
                unit: upw,
                swap_group: 0,
            }))
            .collect(),
        tags: vec!["opamp".into()],
    });
    let tssop14 = p.add_package(footprint::tssop(14, Density::B));
    let quad_part = make::uuid("part", "OPA4171");
    let pads = [
        ("A", "OUT", "1"),
        ("A", "IN-", "2"),
        ("A", "IN+", "3"),
        ("P", "V+", "4"),
        ("B", "IN+", "5"),
        ("B", "IN-", "6"),
        ("B", "OUT", "7"),
        ("C", "OUT", "8"),
        ("C", "IN-", "9"),
        ("C", "IN+", "10"),
        ("P", "V-", "11"),
        ("D", "IN+", "12"),
        ("D", "IN-", "13"),
        ("D", "OUT", "14"),
    ];
    p.add_part(crate::library::Part {
        uuid: quad_part,
        mpn: "OPA4171".into(),
        manufacturer: "Texas Instruments".into(),
        description: "Quad 36 V op-amp".into(),
        entity: quad,
        package: tssop14,
        pad_map: pads
            .iter()
            .map(|(g, n, pad)| crate::library::PadMapEntry {
                gate: g.to_string(),
                pin: n.to_string(),
                pads: vec![pad.to_string()],
            })
            .collect(),
        params: Default::default(),
        lifecycle: crate::library::Lifecycle::Active,
        datasheet: None,
        supply: vec![],
        thermal: crate::library::ThermalData {
            theta_jc: Some(32.0),
            theta_jb: Some(45.0),
            theta_ja: Some(97.0),
            p_max: Some(0.8),
            tj_max: Some(150.0),
        },
        spice: Some(crate::library::SpiceModel {
            model: "OPA4171".into(),
            library: Some("opa4171.lib".into()),
            pin_order: vec![
                ("A".into(), "IN+".into()),
                ("A".into(), "IN-".into()),
                ("P".into(), "V+".into()),
                ("P".into(), "V-".into()),
                ("A".into(), "OUT".into()),
            ],
            device: None,
        }),
        base: None,
    });

    // Buck regulator (QFN-like 3x3 with exposed pad as GND).
    let buck = make::simple_entity(
        &mut p,
        "buck",
        "U",
        &[
            ("VIN", PowerIn),
            ("EN", Input),
            ("SW", PowerOut),
            ("FB", Input),
            ("PG", OpenCollector),
            ("GND", PowerIn),
            ("VOS", Input),
            ("SS", Passive),
        ],
    );
    let qfn16 = p.add_package(footprint::qfn(16, 3.0, 0.5, Some(1.7), Density::B));
    let bp = make::part(
        &mut p,
        "TPS62130",
        "Texas Instruments",
        buck,
        qfn16,
        &[
            ("VIN", &["1", "2"]),
            ("EN", &["3"]),
            ("SW", &["4", "5", "6"]),
            ("FB", &["7"]),
            ("PG", &["8"]),
            ("GND", &["9", "10", "17"]),
            ("VOS", &["11"]),
            ("SS", &["12"]),
        ],
    );
    let bpp = p.parts.get_mut(&bp).unwrap();
    bpp.thermal.theta_jc = Some(28.0);
    bpp.thermal.theta_jb = Some(14.0);
    bpp.thermal.p_max = Some(1.5);

    // USB-C receptacle (USB 2.0 subset).
    let usbc = make::simple_entity(
        &mut p,
        "usb-c",
        "J",
        &[
            ("VBUS", Passive),
            ("CC1", Bidirectional),
            ("CC2", Bidirectional),
            ("DP", Bidirectional),
            ("DN", Bidirectional),
            ("GND", Passive),
            ("SHIELD", Passive),
            ("SBU1", Bidirectional),
            ("SBU2", Bidirectional),
        ],
    );
    let usb_pkg = p.add_package(footprint::smd_connector(
        "USB-C",
        12,
        0.5,
        (0.3, 1.0),
        &[
            (
                Point::mm(-4.32, 0.0),
                crate::library::PadShape::Oval {
                    w: mm(1.0),
                    h: mm(2.1),
                },
                Some(mm(0.6)),
            ),
            (
                Point::mm(4.32, 0.0),
                crate::library::PadShape::Oval {
                    w: mm(1.0),
                    h: mm(2.1),
                },
                Some(mm(0.6)),
            ),
        ],
        (8.94, 7.3),
    ));
    make::part(
        &mut p,
        "USB4105",
        "GCT",
        usbc,
        usb_pkg,
        &[
            ("GND", &["1", "12"]),
            ("VBUS", &["2", "11"]),
            ("CC1", &["3"]),
            ("DP", &["4", "6"]),
            ("DN", &["5", "7"]),
            ("SBU1", &["8"]),
            ("CC2", &["9"]),
            ("SBU2", &["10"]),
            ("SHIELD", &["MP1", "MP2"]),
        ],
    );
    // I²C sensor.
    let sensor = make::simple_entity(
        &mut p,
        "tmp117",
        "U",
        &[
            ("SCL", Input),
            ("SDA", Bidirectional),
            ("ALERT", OpenCollector),
            ("ADD0", Input),
            ("V+", PowerIn),
            ("GND", PowerIn),
        ],
    );
    let sot6 = p.add_package(footprint::sot23(6, Density::B).unwrap());
    make::part(
        &mut p,
        "TMP117",
        "Texas Instruments",
        sensor,
        sot6,
        &[
            ("SCL", &["1"]),
            ("GND", &["2"]),
            ("ALERT", &["3"]),
            ("ADD0", &["4"]),
            ("V+", &["5"]),
            ("SDA", &["6"]),
        ],
    );
    let xtal = make::simple_entity(&mut p, "crystal", "Y", &[("1", Passive), ("2", Passive)]);
    let x3225 = p.add_package(footprint::from_string("C1210").unwrap());
    let xp = make::part_by_name(&mut p, "XTAL-3225", "Generic", xtal, x3225);
    let xpp = p.parts.get_mut(&xp).unwrap();
    xpp.params.insert(
        "frequency".into(),
        Param::quantity(
            Quantity::new(16e6, Dim::FREQUENCY),
            Tolerance::Relative {
                minus: 20e-6,
                plus: 20e-6,
            },
        ),
    );
    xpp.params.insert(
        "load_capacitance".into(),
        Param::quantity(Quantity::new(10e-12, Dim::CAPACITANCE), Tolerance::Exact),
    );

    // MCU in QFN-32: 32 pins + exposed pad.
    let mut mcu_pins: Vec<(String, PinType)> = vec![
        ("VDD".into(), PowerIn),
        ("VSS".into(), PowerIn),
        ("NRST".into(), Input),
        ("BOOT0".into(), Input),
        ("OSC_IN".into(), Input),
        ("OSC_OUT".into(), Output),
        ("SCL".into(), Bidirectional),
        ("SDA".into(), Bidirectional),
        ("USB_DP".into(), Bidirectional),
        ("USB_DN".into(), Bidirectional),
    ];
    for i in 0..20 {
        mcu_pins.push((format!("PA{i}"), Bidirectional));
    }
    let refs: Vec<(&str, PinType)> = mcu_pins.iter().map(|(n, k)| (n.as_str(), *k)).collect();
    let mcu = make::simple_entity(&mut p, "mcu", "U", &refs);
    let qfn32 = p.add_package(footprint::qfn(32, 5.0, 0.5, Some(3.3), Density::B));
    let mut map: Vec<(String, Vec<String>)> = Vec::new();
    let mut free = (1..=32).filter(|k| ![1, 16, 17, 32].contains(k));
    for (n, _) in &mcu_pins {
        let pads = match n.as_str() {
            "VDD" => vec!["1".to_string(), "17".to_string()],
            "VSS" => vec!["16".to_string(), "32".to_string(), "33".to_string()],
            _ => vec![free.next().unwrap().to_string()],
        };
        map.push((n.clone(), pads));
    }
    let map_ref: Vec<(&str, Vec<&str>)> = map
        .iter()
        .map(|(n, v)| (n.as_str(), v.iter().map(|s| s.as_str()).collect()))
        .collect();
    let map_ref2: Vec<(&str, &[&str])> = map_ref.iter().map(|(n, v)| (*n, v.as_slice())).collect();
    make::part(&mut p, "MCU-QFN32", "Generic", mcu, qfn32, &map_ref2);

    // FPGA in BGA-256: banks of IO plus power balls.
    let mut fpins: Vec<(String, PinType)> = Vec::new();
    for i in 0..200 {
        fpins.push((format!("IO{i}"), Bidirectional));
    }
    fpins.push(("VCCINT".into(), PowerIn));
    fpins.push(("VCCIO".into(), PowerIn));
    fpins.push(("GND".into(), PowerIn));
    let fr: Vec<(&str, PinType)> = fpins.iter().map(|(n, k)| (n.as_str(), *k)).collect();
    let fpga = make::simple_entity(&mut p, "fpga", "U", &fr);
    let bga = p.add_package(footprint::bga(
        16,
        16,
        1.0,
        0.5,
        (17.0, 17.0),
        |_, _| true,
        Density::B,
    ));
    let ball_names: Vec<String> = p.packages[&bga]
        .pads
        .iter()
        .map(|x| x.name.clone())
        .collect();
    let mut fmap: Vec<(String, Vec<String>)> = Vec::new();
    for (i, ball) in ball_names.iter().take(200).enumerate() {
        fmap.push((format!("IO{i}"), vec![ball.clone()]));
    }
    fmap.push(("VCCINT".into(), ball_names[200..218].to_vec()));
    fmap.push(("VCCIO".into(), ball_names[218..236].to_vec()));
    fmap.push(("GND".into(), ball_names[236..256].to_vec()));
    let fm1: Vec<(&str, Vec<&str>)> = fmap
        .iter()
        .map(|(n, v)| (n.as_str(), v.iter().map(|s| s.as_str()).collect()))
        .collect();
    let fm2: Vec<(&str, &[&str])> = fm1.iter().map(|(n, v)| (*n, v.as_slice())).collect();
    let fp = make::part(&mut p, "FPGA-BGA256", "Generic", fpga, bga, &fm2);
    p.parts.get_mut(&fp).unwrap().thermal = crate::library::ThermalData {
        theta_jc: Some(6.5),
        theta_jb: Some(9.0),
        theta_ja: Some(24.0),
        p_max: Some(4.0),
        tj_max: Some(100.0),
    };
    let hdr_e = make::simple_entity(
        &mut p,
        "header-10",
        "J",
        &(1..=10)
            .map(|i| {
                (
                    ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"][i - 1],
                    Passive,
                )
            })
            .collect::<Vec<_>>(),
    );
    let hdr = p.add_package(footprint::pin_header(5, 2, 2.54));
    make::part_by_name(&mut p, "HDR-2x05", "Generic", hdr_e, hdr);
    super::pool_daq::add(&mut p);
    p
}
