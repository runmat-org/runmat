//! Fixture generators for tests, benchmarks and examples.
//!
//! - [`standard_pool`]: a small but real pool: resistor, capacitor, inductor, LED, a
//!   quad op-amp with a power gate (multi-gate entity), a buck regulator, a USB-C receptacle,
//!   an I²C sensor, a crystal, an MCU in QFN-32, and a BGA FPGA.
//! - [`large`]: a realistic large design and board: `channels` repeated analog channels
//!   (hierarchical sheet, bus-indexed bindings, local and global nets), a BGA processor with a
//!   wide bus, 8 or more copper layers, thousands of nets, placed and partly routed, with
//!   ground and power pours. Generation is deterministic for a given [`LargeParams`].

use crate::elab::{elaborate, ElabOptions, Netlist};
use crate::geom::poly::{rect_ring, Polygon};
use crate::geom::{mm, Coord, Point, Rect};
use crate::ir::*;
use crate::layout::{Board, Side, Stackup, Zone};
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
    let n = params.channels;
    let cols = ((n as f64).sqrt() * 1.5).ceil() as u32;
    let rows = n.div_ceil(cols);
    let pitch_x = mm(14.0);
    let pitch_y = mm(9.0);
    let w = cols as Coord * pitch_x + mm(60.0);
    let h = rows as Coord * pitch_y + mm(40.0);
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
            let per_col = ((h - mm(20.0)) / mm(14.0)).max(1);
            let (col, row) = (hdr_i / per_col, hdr_i % per_col);
            hdr_i += 1;
            Point::new(w - mm(10.0) - col * mm(9.0), mm(10.0) + row * mm(14.0))
        };
        let id = board.place(&c.designator, pkg.clone(), at, 0.0, Side::Top);
        board.footprints.get_mut(&id).unwrap().value = c.path.leaf().to_string();
    }
    board.assign_nets(nl);
    // Signal routing: each routed net fans out from every pad with a short top stub to a
    // via, and runs between vias on an inner signal layer (alternating by net). Power pads
    // fan out to the planes with a via each.
    let layers = board.copper_count();
    let signal_layers: Vec<u8> = (3..layers.saturating_sub(3).max(4)).collect();
    if params.route_every > 0 {
        let pads = board.all_pads();
        let mut by_net: std::collections::BTreeMap<u32, Vec<Point>> = Default::default();
        for p in &pads {
            if let Some(nn) = p.net {
                by_net.entry(nn.0).or_default().push(p.center);
            }
        }
        // Pad centre → its footprint's centre, for outward dog-bone fanout.
        let centres: std::collections::BTreeMap<Point, Point> = pads
            .iter()
            .map(|p| (p.center, board.footprints[&p.footprint].at))
            .collect();
        let mut k = 0u32;
        for (net, pts) in by_net {
            let nid = Some(crate::layout::NetId(net));
            let class = board.nets[net as usize].class.clone();
            if class.as_deref() == Some("power") {
                for p in &pts {
                    let v = fanout(&centres, *p);
                    board.add_track(nid, 0, mm(0.3), *p, v);
                    board.add_via(nid, v, mm(0.5), mm(0.25));
                }
                continue;
            }
            if pts.len() < 2 || pts.len() > 6 {
                continue;
            }
            k += 1;
            if !k.is_multiple_of(params.route_every) {
                continue;
            }
            let layer = signal_layers[((k / params.route_every) as usize) % signal_layers.len()];
            let vias: Vec<Point> = pts.iter().map(|p| fanout(&centres, *p)).collect();
            for (p, v) in pts.iter().zip(&vias) {
                board.add_track(nid, 0, mm(0.15), *p, *v);
                board.add_via(nid, *v, mm(0.45), mm(0.2));
            }
            for w in vias.windows(2) {
                board.add_track(nid, layer, mm(0.15), w[0], w[1]);
            }
        }
    }
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
    board
}

/// A via site 1 mm outward from the footprint centre through the pad.
fn fanout(centres: &std::collections::BTreeMap<Point, Point>, p: Point) -> Point {
    let c = centres.get(&p).copied().unwrap_or(p);
    let d = p - c;
    let l = d.len();
    if l < 1.0 {
        return p + Point::mm(0.0, 1.0);
    }
    let s = 1.0e6 / l;
    p + Point::from_f(d.x as f64 * s, d.y as f64 * s)
}

/// Everything for the large fixture.
pub fn large(params: &LargeParams) -> (Pool, Design, Netlist, Board) {
    let pool = standard_pool();
    let d = large_design(params);
    let nl = elaborate(&d, &pool, &ElabOptions::default(), None);
    let board = large_board(&pool, &nl, params);
    (pool, d, nl, board)
}
