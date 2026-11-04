//! Fixture generators for tests, benchmarks and examples.
//!
//! - [`standard_pool`]: a small but real pool: resistor, capacitor, inductor, LED, a
//!   quad op-amp with a power gate (multi-gate entity), a buck regulator, a USB-C receptacle,
//!   an I²C sensor, a crystal, an MCU in QFN-32, and a BGA FPGA.
//! - [`large`]: a realistic large design and board: `channels` repeated analog channels
//!   (hierarchical sheet, bus-indexed bindings, local and global nets), a BGA processor with a
//!   wide bus, 8 or more copper layers, thousands of nets, placed and partly routed, with
//!   ground and power pours. Routing is clean by construction: dog-bone fanouts (BGA balls to
//!   the diagonal), horizontal and vertical inner layers joined by buried vias, every object
//!   checked against clearances before it is placed; what does not fit stays an airwire.
//!   [`seed_violations`] then adds a known number of violations. Generation is
//!   deterministic for a given [`LargeParams`].

use crate::elab::{elaborate, ElabOptions, Netlist};
use crate::geom::poly::{rect_ring, Polygon};
use crate::geom::shape::Prim;
use crate::geom::{mm, Coord, Point, Rect};
use crate::ir::*;
use crate::layout::{Board, ItemId, NetId, PlacedPad, Side, Stackup, Zone};
use crate::library::footprint::{self, Density};
use crate::library::{make, Entity, Gate, PinType, Pool, Unit, UnitPin};
use crate::spatial::Grid;
use runmat_rules::{Dim, Quantity};
use std::collections::BTreeMap;

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
    p
}

#[derive(Clone, Debug)]
pub struct LargeParams {
    /// Repeated analog channels (each ≈ 5 local nets, 8 components).
    pub channels: u32,
    pub copper_layers: u8,
    /// Route a straight stub for every this-many nets (0 = no routing).
    pub route_every: u32,
    /// Add GND and power pours.
    pub pours: bool,
}

impl Default for LargeParams {
    fn default() -> Self {
        LargeParams {
            channels: 400,
            copper_layers: 8,
            route_every: 3,
            pours: true,
        }
    }
}

/// A channel: quad op-amp gate A as a non-inverting amplifier with an RC filter, a decoupling
/// cap per supply pin, and the unused gates tied off.
pub fn channel_sheet(d: &mut Design) {
    let mut b = DesignBuilder::new(d, "channel");
    b.port("IN", PortDir::In);
    b.port("OUT", PortDir::Out);
    b.component("U?", "OPA4171");
    b.component("R?", "R0402-10K").name = "RG".into();
    b.component("R?", "R0402-10K").name = "RF".into();
    b.component("R?", "R0402-10K").name = "RS".into();
    b.component("C?", "C0402-100N").name = "CF".into();
    b.component("C?", "C0402-100N").name = "CD1".into();
    b.component("C?", "C0402-100N").name = "CD2".into();
    // Rename the op-amp's local name.
    b.design.sheets.get_mut("channel").unwrap().components[0].name = "U".into();
    b.wire("IN", &[("RS", "1")]);
    b.wire("INF", &[("RS", "2"), ("CF", "1"), ("U", "A.IN+")]);
    b.wire("FBK", &[("U", "A.IN-"), ("RG", "1"), ("RF", "1")]);
    b.wire("OUT", &[("U", "A.OUT"), ("RF", "2")]);
    b.connect(vec![Endpoint::Port("IN".into()), Endpoint::net("IN")]);
    b.connect(vec![Endpoint::Port("OUT".into()), Endpoint::net("OUT")]);
    b.global("GND", Some("power"));
    b.global("+5V", Some("power"));
    b.global("-5V", Some("power"));
    b.wire(
        "GND",
        &[("RG", "2"), ("CF", "2"), ("CD1", "2"), ("CD2", "1")],
    );
    b.wire("+5V", &[("U", "V+"), ("CD1", "1")]);
    b.wire("-5V", &[("U", "V-"), ("CD2", "2")]);
    // Unused gates as grounded followers.
    for g in ["B", "C", "D"] {
        let n = format!("TIE_{g}");
        b.wire(
            &n,
            &[("U", &format!("{g}.OUT")), ("U", &format!("{g}.IN-"))],
        );
        b.wire("GND", &[("U", &format!("{g}.IN+"))]);
    }
}

/// The large design: `channels` channels into an FPGA, plus power entry.
pub fn large_design(params: &LargeParams) -> Design {
    let mut d = Design::new("large");
    d.net_classes.insert(
        "power".into(),
        NetClass {
            name: "power".into(),
            params: Default::default(),
            source: Default::default(),
        },
    );
    channel_sheet(&mut d);
    let mut b = DesignBuilder::new(&mut d, "main");
    let n = params.channels;
    let ins: Vec<String> = (0..n).map(|i| format!("AIN{i}")).collect();
    let outs: Vec<String> = (0..n).map(|i| format!("ADC{i}")).collect();
    let ins_ref: Vec<&str> = ins.iter().map(|s| s.as_str()).collect();
    let outs_ref: Vec<&str> = outs.iter().map(|s| s.as_str()).collect();
    b.bus("AIN", &ins_ref);
    b.bus("ADC", &outs_ref);
    b.global("GND", Some("power"));
    b.global("+5V", Some("power"));
    b.global("-5V", Some("power"));
    let inst = b.instance("ch", "channel", Some(n));
    inst.bindings.push((
        "IN".into(),
        Binding::BusIndexed {
            bus: "AIN".into(),
            offset: 0,
        },
    ));
    inst.bindings.push((
        "OUT".into(),
        Binding::BusIndexed {
            bus: "ADC".into(),
            offset: 0,
        },
    ));
    // FPGAs read the channel outputs: one FPGA per 200 channels.
    let fpgas = n.div_ceil(200);
    for f in 0..fpgas {
        let name = format!("U{}", 9000 + f);
        b.component(&name, "FPGA-BGA256");
        for k in 0..200u32 {
            let ch = f * 200 + k;
            if ch >= n {
                break;
            }
            b.wire(&format!("ADC{ch}"), &[(&name, &format!("IO{k}"))]);
        }
        b.wire("GND", &[(&name, "GND")]);
        b.wire("+5V", &[(&name, "VCCINT"), (&name, "VCCIO")]);
    }
    // Input connectors: one 10-pin header per 8 channels (pins 9, 10 ground).
    for h in 0..n.div_ceil(8) {
        let name = format!("J{}", 100 + h);
        b.component(&name, "HDR-2x05");
        for k in 0..8u32 {
            let ch = h * 8 + k;
            if ch < n {
                b.wire(&format!("AIN{ch}"), &[(&name, &(k + 1).to_string())]);
            } else {
                b.no_erc(&name, &(k + 1).to_string());
            }
        }
        b.wire("GND", &[(&name, "9"), (&name, "10")]);
    }
    d
}

/// Places, routes stubs and pours the large design on a board.
pub fn large_board(pool: &Pool, nl: &Netlist, params: &LargeParams) -> Board {
    large_board_report(pool, nl, params).0
}

/// What the large-board generator routed.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RouteReport {
    /// Signal nets chosen for routing (every `route_every`-th eligible net).
    pub attempted: u32,
    /// Chosen nets fully connected.
    pub completed: u32,
    /// Pad fanouts placed (signal and power).
    pub fanouts: u32,
    /// Fanouts the generator could not place without a violation (the pad stays unrouted).
    pub fanouts_skipped: u32,
}

/// [`large_board`] with its routing report. Routing is clean by construction: every object
/// is placed only where it keeps the clearances of the large test's rules (0.2 mm to power,
/// 0.15 mm otherwise), the hole-to-hole minimum and the edge clearance; connections the
/// generator cannot fit are left unrouted (airwires), as on a board in progress.
pub fn large_board_report(pool: &Pool, nl: &Netlist, params: &LargeParams) -> (Board, RouteReport) {
    let n = params.channels;
    let cols = ((n as f64).sqrt() * 1.5).ceil() as u32;
    let rows = n.div_ceil(cols);
    let pitch_x = mm(14.0);
    let pitch_y = mm(9.0);
    let h = rows as Coord * pitch_y + mm(40.0);
    // Headers in columns at the right, clear of the channel array.
    let headers = n.div_ceil(8) as Coord;
    let per_col = ((h - mm(20.0)) / mm(14.0)).max(1);
    let header_cols = (headers + per_col - 1) / per_col;
    let w = mm(40.0) + cols as Coord * pitch_x + mm(6.0) + header_cols * mm(9.0) + mm(10.0);
    let outline = Polygon::from_ring(rect_ring(Rect::new(Point::ORIGIN, Point::new(w, h))));
    let mut board = Board::new(
        "large",
        outline,
        Stackup::standard(params.copper_layers, mm(1.6), 1.0, 0.5),
    );
    let mut fpga_i = 0;
    let mut hdr_i: Coord = 0;
    for c in &nl.components {
        let Some(pkg) = c.package.and_then(|u| pool.packages.get(&u)) else {
            continue;
        };
        let at = if let Some(ch) = c.channel {
            let (cx, cy) = ((ch % cols) as Coord, (ch / cols) as Coord);
            let base = Point::new(mm(40.0) + cx * pitch_x, mm(20.0) + cy * pitch_y);
            let off = match c.path.leaf() {
                "U" => Point::mm(0.0, 0.0),
                "RG" => Point::mm(5.0, 2.0),
                "RF" => Point::mm(5.0, 0.0),
                "RS" => Point::mm(-5.0, 2.0),
                "CF" => Point::mm(-5.0, 0.0),
                "CD1" => Point::mm(5.0, -2.0),
                _ => Point::mm(-5.0, -2.0),
            };
            base + off
        } else if c.designator.starts_with("U9") {
            fpga_i += 1;
            Point::new(mm(15.0), mm(30.0) * fpga_i)
        } else {
            let (col, row) = (hdr_i / per_col, hdr_i % per_col);
            hdr_i += 1;
            Point::new(w - mm(10.0) - col * mm(9.0), mm(10.0) + row * mm(14.0))
        };
        let id = board.place(&c.designator, pkg.clone(), at, 0.0, Side::Top);
        board.footprints.get_mut(&id).unwrap().value = c.path.leaf().to_string();
    }
    board.assign_nets(nl);
    let report = route_large(&mut board, params);
    if params.pours {
        let gnd = board.net("GND");
        let p5 = board.net("+5V");
        let area = Polygon::from_ring(rect_ring(Rect::new(
            Point::mm(1.0, 1.0),
            Point::new(w - mm(1.0), h - mm(1.0)),
        )));
        let z1 = board.alloc();
        board.zones.insert(
            z1,
            Zone::new(
                z1,
                "GND",
                gnd,
                vec![1, board.copper_count() - 2],
                area.clone(),
            ),
        );
        let z2 = board.alloc();
        let mut pz = Zone::new(z2, "+5V", p5, vec![2], area);
        pz.priority = 1;
        board.zones.insert(z2, pz);
    }
    (board, report)
}

/// Clearances the generator keeps (a little above the large test's rules).
const GEN_CLEARANCE: Coord = 160_000;
const GEN_POWER_CLEARANCE: Coord = 210_000;
const GEN_HOLE_TO_HOLE: Coord = 260_000;
const GEN_EDGE: Coord = 350_000;

/// Copper placed so far, for the generator's legality checks.
struct Occupancy {
    grid: Grid<u32>,
    items: Vec<(Option<NetId>, bool, u64, Prim)>,
    holes: Grid<u32>,
    hole_list: Vec<(Point, Coord)>,
    area: Rect,
}

impl Occupancy {
    fn clearance(power_a: bool, power_b: bool) -> Coord {
        if power_a || power_b {
            GEN_POWER_CLEARANCE
        } else {
            GEN_CLEARANCE
        }
    }
    fn free(&self, net: Option<NetId>, power: bool, mask: u64, p: &Prim) -> bool {
        let bb = p.bbox();
        if bb.min.x < self.area.min.x
            || bb.min.y < self.area.min.y
            || bb.max.x > self.area.max.x
            || bb.max.y > self.area.max.y
        {
            return false;
        }
        let mut ok = true;
        self.grid.visit(&bb.inflate(GEN_POWER_CLEARANCE), |k, _| {
            if !ok {
                return;
            }
            let (on, opower, omask, op) = &self.items[k as usize];
            if omask & mask == 0 || (net.is_some() && *on == net) {
                return;
            }
            if p.gap(op) < Occupancy::clearance(power, *opower) as f64 {
                ok = false;
            }
        });
        ok
    }
    fn hole_free(&self, at: Point, r: Coord) -> bool {
        let mut ok = true;
        self.holes.visit(
            &Rect::new(at, at).inflate(r + GEN_HOLE_TO_HOLE + mm(1.0)),
            |k, _| {
                let (c, cr) = self.hole_list[k as usize];
                if at.dist(c) - ((r + cr) as f64) < GEN_HOLE_TO_HOLE as f64 {
                    ok = false;
                }
            },
        );
        ok
    }
    fn add(&mut self, net: Option<NetId>, power: bool, mask: u64, p: Prim) {
        let k = self.items.len() as u32;
        self.grid.insert(k, p.bbox());
        self.items.push((net, power, mask, p));
    }
    fn add_hole(&mut self, at: Point, r: Coord) {
        let k = self.hole_list.len() as u32;
        self.holes.insert(k, Rect::new(at, at).inflate(r));
        self.hole_list.push((at, r));
    }
}

struct Router<'b> {
    board: &'b mut Board,
    occ: Occupancy,
}

impl Router<'_> {
    fn power(&self, net: Option<NetId>) -> bool {
        net.is_some_and(|n| self.board.nets[n.0 as usize].class.as_deref() == Some("power"))
    }
    fn track_ok(&self, net: Option<NetId>, layer: u8, w: Coord, a: Point, b: Point) -> bool {
        self.occ
            .free(net, self.power(net), 1 << layer, &Prim::segment(a, b, w))
    }
    fn via_ok(
        &self,
        net: Option<NetId>,
        at: Point,
        d: Coord,
        drill: Coord,
        span: (u8, u8),
    ) -> bool {
        let mask = (span.0..=span.1).fold(0u64, |m, l| m | 1 << l);
        self.occ
            .free(net, self.power(net), mask, &Prim::circle(at, d / 2))
            && self.occ.hole_free(at, drill / 2)
    }
    fn track(&mut self, net: Option<NetId>, layer: u8, w: Coord, a: Point, b: Point) {
        let pw = self.power(net);
        self.occ.add(net, pw, 1 << layer, Prim::segment(a, b, w));
        self.board.add_track(net, layer, w, a, b);
    }
    fn via(&mut self, net: Option<NetId>, at: Point, d: Coord, drill: Coord, span: (u8, u8)) {
        let pw = self.power(net);
        let mask = (span.0..=span.1).fold(0u64, |m, l| m | 1 << l);
        self.occ.add(net, pw, mask, Prim::circle(at, d / 2));
        self.occ.add_hole(at, drill / 2);
        let id = self.board.add_via(net, at, d, drill);
        let v = self.board.vias.get_mut(&id).unwrap();
        v.from = span.0;
        v.to = span.1;
    }

    /// A dog-bone fanout from an SMD pad to a through via: candidate sites outward from the
    /// footprint along the pad's major axis (staggered for fine pitch), then sideways; BGA
    /// balls fan out to the diagonal between balls, away from the package centre.
    fn fanout(&mut self, pad: &PlacedPad, fp_at: Point, bga: bool, width: Coord) -> Option<Point> {
        let (d, drill) = if self.power(pad.net) {
            (mm(0.5), mm(0.25))
        } else {
            (mm(0.45), mm(0.2))
        };
        let rel = pad.center - fp_at;
        let mut sites: Vec<Point> = Vec::new();
        if bga {
            let sx = if rel.x >= 0 { 1 } else { -1 };
            let sy = if rel.y >= 0 { 1 } else { -1 };
            for (fx, fy) in [(sx, sy), (sx, -sy), (-sx, sy), (-sx, -sy)] {
                sites.push(pad.center + Point::new(fx * mm(0.5), fy * mm(0.5)));
            }
        } else {
            let along_x = rel.x.abs() >= rel.y.abs();
            let sign = |v: Coord| if v >= 0 { 1 } else { -1 };
            let stagger = if pad.index.is_multiple_of(2) {
                0
            } else {
                mm(0.7)
            };
            for dist in [
                mm(0.9) + stagger,
                mm(1.6) + stagger,
                mm(0.9),
                mm(1.25),
                mm(1.6),
                mm(2.3),
                mm(3.0),
            ] {
                let off = if along_x {
                    Point::new(sign(rel.x) * dist, 0)
                } else {
                    Point::new(0, sign(rel.y) * dist)
                };
                sites.push(pad.center + off);
            }
            for dist in [mm(0.7), mm(0.9), mm(1.2), mm(1.6), mm(2.0)] {
                let off = if along_x {
                    Point::new(0, dist)
                } else {
                    Point::new(dist, 0)
                };
                sites.push(pad.center + off);
                sites.push(pad.center - off);
            }
        }
        let last = self.board.copper_count() - 1;
        for v in sites {
            if self.track_ok(pad.net, 0, width, pad.center, v)
                && self.via_ok(pad.net, v, d, drill, (0, last))
            {
                self.track(pad.net, 0, width, pad.center, v);
                self.via(pad.net, v, d, drill, (0, last));
                return Some(v);
            }
        }
        None
    }

    /// Routes `a → b` on the layer pair `(hl, vl)`: horizontal runs on `hl`, vertical runs on
    /// `vl`, joined by buried vias spanning the pair. Tries a straight run, the two L shapes,
    /// then Z shapes over lanes around the midpoint.
    fn connect(&mut self, net: Option<NetId>, a: Point, b: Point, (hl, vl): (u8, u8)) -> bool {
        let w = mm(0.15);
        let (vd, vdr) = (mm(0.45), mm(0.2));
        let span = (hl.min(vl), hl.max(vl));
        let seg_layer = |p: Point, q: Point| {
            if p.y == q.y {
                Some(hl)
            } else if p.x == q.x {
                Some(vl)
            } else {
                None
            }
        };
        let mut candidates: Vec<Vec<Point>> = Vec::new();
        if a.x == b.x || a.y == b.y {
            candidates.push(vec![a, b]);
        }
        candidates.push(vec![a, Point::new(b.x, a.y), b]);
        candidates.push(vec![a, Point::new(a.x, b.y), b]);
        let (mx, my) = ((a.x + b.x) / 2, (a.y + b.y) / 2);
        for k in 1..=12 {
            for s in [1, -1] {
                let o = s * k * mm(0.4);
                let yl = my + o;
                candidates.push(vec![a, Point::new(a.x, yl), Point::new(b.x, yl), b]);
                let xl = mx + o;
                candidates.push(vec![a, Point::new(xl, a.y), Point::new(xl, b.y), b]);
            }
        }
        'cand: for path in candidates {
            let mut path = path;
            path.dedup();
            if path.len() < 2 {
                return true;
            }
            for s in path.windows(2) {
                let Some(l) = seg_layer(s[0], s[1]) else {
                    continue 'cand;
                };
                if !self.track_ok(net, l, w, s[0], s[1]) {
                    continue 'cand;
                }
            }
            // Bend vias where the layer changes.
            let mut bends = Vec::new();
            for k in 1..path.len() - 1 {
                if seg_layer(path[k - 1], path[k]) != seg_layer(path[k], path[k + 1]) {
                    if !self.via_ok(net, path[k], vd, vdr, span) {
                        continue 'cand;
                    }
                    bends.push(path[k]);
                }
            }
            // Bend vias must also clear each other.
            for (i, p) in bends.iter().enumerate() {
                for q in &bends[i + 1..] {
                    if p.dist(*q) < (vd + GEN_HOLE_TO_HOLE) as f64 {
                        continue 'cand;
                    }
                }
            }
            for s in path.windows(2) {
                let l = seg_layer(s[0], s[1]).unwrap();
                self.track(net, l, w, s[0], s[1]);
            }
            for p in bends {
                self.via(net, p, vd, vdr, span);
            }
            return true;
        }
        false
    }
}

/// Fans out and routes the large board (see [`large_board_report`]).
fn route_large(board: &mut Board, params: &LargeParams) -> RouteReport {
    let layers = board.copper_count();
    let n = layers;
    let mut occ = Occupancy {
        grid: Grid::new(mm(1.0)),
        items: Vec::new(),
        holes: Grid::new(mm(2.0)),
        hole_list: Vec::new(),
        area: board.outline.bbox().inflate(-GEN_EDGE),
    };
    let pads = board.all_pads();
    for p in &pads {
        let power = p
            .net
            .is_some_and(|nn| board.nets[nn.0 as usize].class.as_deref() == Some("power"));
        let mask = p.layers.iter().fold(0u64, |m, l| m | 1 << l);
        if mask != 0 {
            occ.items.push((p.net, power, mask, p.prim.clone()));
            let k = occ.items.len() as u32 - 1;
            occ.grid.insert(k, p.prim.bbox());
        }
        if let Some(d) = p.drill {
            occ.add_hole(p.center, d / 2);
        }
    }
    let mut r = Router { board, occ };
    let mut rep = RouteReport::default();
    let centres: BTreeMap<(ItemId, usize), (Point, bool)> = pads
        .iter()
        .map(|p| {
            let fp = &r.board.footprints[&p.footprint];
            (
                (p.footprint, p.index),
                (fp.at, fp.package.name.starts_with("BGA")),
            )
        })
        .collect();
    let mut by_net: BTreeMap<u32, Vec<&PlacedPad>> = BTreeMap::new();
    for p in &pads {
        if let Some(nn) = p.net {
            by_net.entry(nn.0).or_default().push(p);
        }
    }
    // Power first: each pad gets a fanout via to the planes.
    for (net, ps) in &by_net {
        if r.board.nets[*net as usize].class.as_deref() != Some("power") {
            continue;
        }
        for p in ps {
            if p.layers.len() > 1 {
                continue; // through-hole: already on every plane
            }
            let (at, bga) = centres[&(p.footprint, p.index)];
            match r.fanout(p, at, bga, mm(0.3)) {
                Some(_) => rep.fanouts += 1,
                None => rep.fanouts_skipped += 1,
            }
        }
    }
    // Signal layer pairs (horizontal, vertical): adjacent inner layers between the planes.
    let signal: Vec<u8> = (3..n.saturating_sub(2)).collect();
    let pairs: Vec<(u8, u8)> = match signal.len() {
        0 => vec![(0, n - 1)],
        1 => vec![(signal[0], signal[0])],
        _ => signal
            .chunks(2)
            .filter(|c| c.len() == 2)
            .map(|c| (c[0], c[1]))
            .collect(),
    };
    if params.route_every > 0 && !pairs.is_empty() {
        let mut k = 0u32;
        for (net, ps) in &by_net {
            if r.board.nets[*net as usize].class.as_deref() == Some("power") {
                continue;
            }
            if ps.len() < 2 || ps.len() > 6 {
                continue;
            }
            k += 1;
            if !k.is_multiple_of(params.route_every) {
                continue;
            }
            rep.attempted += 1;
            let nid = Some(NetId(*net));
            let pair = pairs[((k / params.route_every) as usize) % pairs.len()];
            let mut anchors: Vec<Point> = Vec::new();
            let mut ok = true;
            for p in ps {
                if p.layers.len() > 1 {
                    anchors.push(p.center);
                    continue;
                }
                let (at, bga) = centres[&(p.footprint, p.index)];
                match r.fanout(p, at, bga, mm(0.15)) {
                    Some(v) => {
                        rep.fanouts += 1;
                        anchors.push(v);
                    }
                    None => {
                        rep.fanouts_skipped += 1;
                        ok = false;
                    }
                }
            }
            // Chain anchors by Prim's MST on Manhattan distance.
            if anchors.len() >= 2 {
                let mut inside = vec![false; anchors.len()];
                inside[0] = true;
                for _ in 1..anchors.len() {
                    let mut best: Option<(Coord, usize, usize)> = None;
                    for (i, a) in anchors.iter().enumerate() {
                        if !inside[i] {
                            continue;
                        }
                        for (j, b) in anchors.iter().enumerate() {
                            if inside[j] {
                                continue;
                            }
                            let d = (a.x - b.x).abs() + (a.y - b.y).abs();
                            if best.is_none_or(|x| d < x.0) {
                                best = Some((d, i, j));
                            }
                        }
                    }
                    let (_, i, j) = best.unwrap();
                    inside[j] = true;
                    if !r.connect(nid, anchors[i], anchors[j], pair) {
                        ok = false;
                    }
                }
            }
            if ok {
                rep.completed += 1;
            }
        }
    }
    rep
}

/// Violations [`seed_violations`] added, by DRC code.
pub type Seeded = BTreeMap<&'static str, u32>;

/// Adds `count` known violations of each kind to a generated board, each producing exactly one
/// finding: a no-net track too close to one other track (`clearance`), a no-net via whose hole
/// is too close to another via's (`hole-to-hole` and `clearance`), and a no-net track below
/// the minimum width (`track-width`). Seeds are placed only where they conflict with exactly
/// the intended object.
pub fn seed_violations(board: &mut Board, count: u32) -> Seeded {
    let mut out = Seeded::new();
    let n = board.copper_count();
    let inner: Vec<ItemId> = board
        .tracks
        .values()
        .filter(|t| t.layer > 0 && t.layer + 1 < n && t.length() > mm(3.0) as f64)
        .map(|t| t.id)
        .collect();
    // Pads do not move while seeding.
    let pads = board.all_pads();
    // Objects within `radius` of `prim` on `mask`, with their gap and net.
    let probe = |board: &Board, prim: &Prim, mask: u64, radius: Coord| {
        let mut hits: Vec<(BItemLite, f64, Option<NetId>)> = Vec::new();
        for t in board.tracks.values() {
            if mask & (1 << t.layer) != 0 {
                for p in t.prims() {
                    let g = p.gap(prim);
                    if g < radius as f64 {
                        hits.push((BItemLite::Track(t.id), g, t.net));
                    }
                }
            }
        }
        for v in board.vias.values() {
            let vm = v.layers().fold(0u64, |m, l| m | 1 << l);
            let g = Prim::circle(v.at, v.diameter / 2).gap(prim);
            if vm & mask != 0 && g < radius as f64 {
                hits.push((BItemLite::Via(v.id), g, v.net));
            }
        }
        for p in &pads {
            let pm = p.layers.iter().fold(0u64, |m, l| m | 1 << l);
            let g = p.prim.gap(prim);
            if pm & mask != 0 && g < radius as f64 {
                hits.push((BItemLite::Pad, g, p.net));
            }
        }
        hits
    };
    let holes_clear = |board: &Board, c: Point, r: Coord, except: Option<ItemId>| {
        board
            .vias
            .values()
            .filter(|o| Some(o.id) != except)
            .map(|o| (o.at, o.drill / 2))
            .chain(
                pads.iter()
                    .filter_map(|p| p.drill.map(|d| (p.center, d / 2))),
            )
            .all(|(at, hr)| c.dist(at) - ((hr + r) as f64) > mm(0.27) as f64)
    };
    let mut placed = [0u32; 3];
    let stride = (inner.len() / (3 * count.max(1) as usize + 1)).max(1);
    // Spread seeds over the board first, then take any track that fits.
    let order: Vec<usize> = (0..inner.len())
        .step_by(stride)
        .chain(0..inner.len())
        .collect();
    for kind in 0..3usize {
        for &i in &order {
            if placed[kind] >= count {
                break;
            }
            let Some(t) = board.tracks.get(&inner[i]).cloned() else {
                continue;
            };
            let (a, b) = t.endpoints();
            let horizontal = a.y == b.y;
            let mid = Point::new((a.x + b.x) / 2, (a.y + b.y) / 2);
            let off = |d: Coord| {
                if horizontal {
                    Point::new(0, d)
                } else {
                    Point::new(d, 0)
                }
            };
            let along = |d: Coord| {
                if horizontal {
                    Point::new(d, 0)
                } else {
                    Point::new(0, d)
                }
            };
            match kind {
                0 => {
                    // Parallel stub 0.1 mm from the track edge (rules ≥ 0.12 mm), nothing else
                    // within 0.6 mm.
                    let c = mid + off(t.width / 2 + mm(0.1) + mm(0.075));
                    let (p, q) = (c - along(mm(0.5)), c + along(mm(0.5)));
                    let hits = probe(board, &Prim::segment(p, q, mm(0.15)), 1 << t.layer, mm(0.6));
                    if hits.len() == 1 && hits[0].0 == BItemLite::Track(t.id) {
                        board.add_track(None, t.layer, mm(0.15), p, q);
                        placed[0] += 1;
                    }
                }
                1 => {
                    // A via 0.49 mm from this track's end via, with a 0.5 mm land and 0.3 mm
                    // drill (ring 0.1 mm): holes 0.24 mm apart (minimum 0.25), lands
                    // overlapping, and the end via's own tracks at least 0.165 mm away.
                    let Some(v) = board
                        .vias
                        .values()
                        .find(|v| v.at == a && v.net == t.net)
                        .cloned()
                    else {
                        continue;
                    };
                    for dir in [off(mm(0.49)), off(-mm(0.49)), -along(mm(0.49))] {
                        let c = v.at + dir;
                        let hits =
                            probe(board, &Prim::circle(c, mm(0.25)), (1u64 << n) - 1, mm(0.22));
                        let ok = hits.iter().all(|(h, g, net)| {
                            *h == BItemLite::Via(v.id) || (*net == t.net && *g >= mm(0.16) as f64)
                        }) && hits.iter().filter(|h| h.0 == BItemLite::Via(v.id)).count()
                            == 1;
                        if ok && holes_clear(board, c, mm(0.15), Some(v.id)) {
                            board.add_via(None, c, mm(0.5), mm(0.3));
                            placed[1] += 1;
                            break;
                        }
                    }
                }
                _ => {
                    // A narrow isolated stub well clear of everything.
                    let c = mid + off(mm(1.5));
                    let (p, q) = (c - along(mm(0.4)), c + along(mm(0.4)));
                    if probe(board, &Prim::segment(p, q, mm(0.1)), 1 << t.layer, mm(0.8)).is_empty()
                    {
                        board.add_track(None, t.layer, mm(0.1), p, q);
                        placed[2] += 1;
                    }
                }
            }
        }
    }
    let placed = (placed[0], placed[1], placed[2]);
    out.insert("clearance", placed.0 + placed.1);
    out.insert("hole-to-hole", placed.1);
    out.insert("track-width", placed.2);
    out
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum BItemLite {
    Track(ItemId),
    Via(ItemId),
    Pad,
}

/// Everything for the large fixture.
pub fn large(params: &LargeParams) -> (Pool, Design, Netlist, Board) {
    let pool = standard_pool();
    let d = large_design(params);
    let nl = elaborate(&d, &pool, &ElabOptions::default(), None);
    let board = large_board(&pool, &nl, params);
    (pool, d, nl, board)
}

/* ------------------------------------------------------------------------------------------------
 * The ECO scenario: a synced board and a schematic edited every way an ECO handles
 * --------------------------------------------------------------------------------------------- */

/// A board synced with [`eco_synced_design`], edited on the board side, and the schematic
/// [`eco_edited_design`] with every kind of change, elaborated with the lock. Tests and the
/// Desktop's ECO fixture use it.
pub struct EcoScenario {
    pub pool: Pool,
    pub synced: Design,
    pub synced_netlist: Netlist,
    pub design: Design,
    pub netlist: Netlist,
    /// Reconciled with `netlist` (aliases carry the renames).
    pub lock: crate::ids::LockSection,
    pub board: Board,
}

pub const ECO_PARAMS: LargeParams = LargeParams {
    channels: 2,
    copper_layers: 4,
    route_every: 2,
    pours: true,
};

/// Gives every declaration a source span in `io.circuit`, in sheet order.
fn stamp_sources(d: &mut Design) {
    let mut at = 40u32;
    let mut next = |len: u32| {
        let k = runmat_rules::SourceKey::new("io.circuit", at, at + len);
        at += len + 1;
        k
    };
    for nc in d.net_classes.values_mut() {
        nc.source = next(48);
    }
    for s in d.sheets.values_mut() {
        for c in s.components.iter_mut() {
            c.source = next(36);
        }
        for n in s.nets.iter_mut() {
            n.source = next(24);
        }
        for c in s.connections.iter_mut() {
            c.source = next(30);
        }
    }
}

/// Two channels into the FPGA, a bulk capacitor `C900` and a bleed resistor `R900` on the rails.
pub fn eco_synced_design() -> Design {
    let mut d = large_design(&ECO_PARAMS);
    let mut b = DesignBuilder::new(&mut d, "main");
    b.component("C900", "C0402-100N");
    b.component("R900", "R0402-10K");
    b.wire("+5V", &[("C900", "1"), ("R900", "1")]);
    b.wire("GND", &[("C900", "2"), ("R900", "2")]);
    stamp_sources(&mut d);
    d
}

/// The synced design edited every way: a third channel; `C900` on a 0805 10 µF part
/// (footprint, value, parameters); `R900` removed; `J100` renamed `J1` (an identity-keeping
/// rename); `-5V` renamed `VNEG`; `J1.9` moved from ground to a new `SHIELD` net; a new
/// `analog` net class and a rule on `power`; a differential pair; and the `lite` variant, which
/// leaves out channel 0's negative-rail decoupler.
pub fn eco_edited_design() -> Design {
    let mut d = large_design(&LargeParams {
        channels: 3,
        ..ECO_PARAMS
    });
    let mut b = DesignBuilder::new(&mut d, "main");
    b.component("C900", "C0805-10U");
    b.wire("+5V", &[("C900", "1")]);
    b.wire("GND", &[("C900", "2")]);
    b.net("SHIELD");
    b.diff_pair("AIN01", "AIN0", "AIN1", None);
    let main = d.sheets.get_mut("main").unwrap();
    // J100 → J1, keeping its identity; pin 9 leaves ground for the shield.
    for c in main.components.iter_mut().filter(|c| c.name == "J100") {
        c.name = "J1".into();
        c.renamed_from = Some("J100".into());
    }
    for conn in main.connections.iter_mut() {
        let gnd = conn.ends.first() == Some(&Endpoint::net("GND"));
        for e in conn.ends.iter_mut() {
            if let Endpoint::Pin { component, .. } = e {
                if component == "J100" {
                    *component = "J1".into();
                }
            }
        }
        if gnd {
            conn.ends.retain(|e| *e != Endpoint::pin("J1", "9"));
        }
    }
    main.connections.push(Connection {
        ends: vec![Endpoint::net("SHIELD"), Endpoint::pin("J1", "9")],
        source: Default::default(),
    });
    // -5V → VNEG everywhere.
    for s in d.sheets.values_mut() {
        for n in s.nets.iter_mut().filter(|n| n.name == "-5V") {
            n.name = "VNEG".into();
            n.renamed_from = Some("-5V".into());
        }
        for conn in s.connections.iter_mut() {
            for e in conn.ends.iter_mut() {
                if *e == Endpoint::net("-5V") {
                    *e = Endpoint::net("VNEG");
                }
            }
        }
    }
    d.net_classes.get_mut("power").unwrap().params.insert("track_width".into(), Param::text("0.40 mm"));
    d.net_classes.insert(
        "analog".into(),
        NetClass {
            name: "analog".into(),
            params: [("clearance".to_owned(), Param::text("0.20 mm"))].into_iter().collect(),
            source: Default::default(),
        },
    );
    d.variants.insert(
        "lite".into(),
        VariantDef {
            name: "lite".into(),
            base: None,
            overrides: [("/ch[0]/CD2".to_owned(), Fitting::NotFitted)].into_iter().collect(),
        },
    );
    stamp_sources(&mut d);
    d
}

/// The scenario: the synced board (placed, routed, poured, then synced by a full ECO), the
/// board-side edits (gates B and C of channel 1's op-amp swapped, channel 1's `RS` renumbered
/// `R40`, channel 0's `RG` renumbered `R50` while the schematic renumbered it `R60`), and the
/// edited schematic in the `lite` variant.
pub fn eco_scenario() -> EcoScenario {
    use crate::eco::{self, ApplyOptions, ComparatorOptions, Schematic};
    use crate::ids::{DerivedUuids, LockSection};
    let pool = standard_pool();
    let synced = eco_synced_design();
    let synced_netlist = elaborate(&synced, &pool, &ElabOptions::default(), None);
    let mut lock = LockSection::default();
    lock.reconcile(&synced_netlist.lock_items(), &mut DerivedUuids { namespace: 7 });
    let (mut board, _) = large_board_report(&pool, &synced_netlist, &ECO_PARAMS);
    // The first sync is an ECO too: values, parameters, classes and rooms come over.
    {
        let s = Schematic {
            design: &synced,
            netlist: &synced_netlist,
            pool: &pool,
            lock: &lock,
        };
        let plan = eco::compare(&s, &board, &ComparatorOptions::default());
        eco::apply(&mut board, &plan, &plan.ids(), &pool, &ApplyOptions::default()).expect("the first sync validates");
        eco::record_sync(&mut board, &s);
    }
    // Board-side edits.
    let by_path = |nl: &Netlist, p: &str| nl.components.iter().find(|c| c.path.to_string() == p).cloned().unwrap();
    let u = by_path(&synced_netlist, "/ch[1]/U");
    let fid = board.footprints.values().find(|f| f.reference == u.designator).unwrap().id;
    {
        let f = board.footprints.get_mut(&fid).unwrap();
        for pin in u.pins.iter().filter(|p| p.gate == "B") {
            let other = u.pins.iter().find(|q| q.gate == "C" && q.pin == pin.pin).unwrap();
            let (a, b) = (&pin.pads[0], &other.pads[0]);
            let (na, nb) = (f.pad_nets.get(a).copied(), f.pad_nets.get(b).copied());
            if let (Some(na), Some(nb)) = (na, nb) {
                f.pad_nets.insert(a.clone(), nb);
                f.pad_nets.insert(b.clone(), na);
            }
        }
    }
    for (path, to) in [("/ch[1]/RS", "R40"), ("/ch[0]/RG", "R50")] {
        let d = by_path(&synced_netlist, path).designator;
        board.footprints.values_mut().find(|f| f.reference == d).unwrap().reference = to.into();
    }
    // The schematic renumbered channel 0's RG too.
    lock.entries.get_mut(&crate::ids::NamePath::parse("/ch[0]/RG")).unwrap().designator = Some("R60".into());
    let design = eco_edited_design();
    let netlist = elaborate(
        &design,
        &pool,
        &ElabOptions {
            variant: Some("lite".into()),
        },
        Some(&lock),
    );
    lock.reconcile(&netlist.lock_items(), &mut DerivedUuids { namespace: 7 });
    EcoScenario {
        pool,
        synced,
        synced_netlist,
        design,
        netlist,
        lock,
        board,
    }
}
