//! Parts of the quadcopter showcase's two boards (`runmat-cad/wasm/tests/drone`): the flight
//! controller (a Cortex-M4-class MCU in LQFP-48, a six-axis IMU, a barometer, a 36 V buck to
//! 5 V and a 3.3 V LDO, USB-C and JST-SH connectors) and the 4-in-1 ESC (per channel a small
//! MCU, a three-phase gate driver, six 40 V N-channel MOSFETs and a current shunt; wire pads for
//! the battery and the motor phases). Every part is a generic type with its own footprint; no
//! part names a vendor.

use crate::ir::*;
use crate::library::footprint::{self, Density};
use crate::library::{make, PadShape, PinType, Pool, ThermalData};
use crate::geom::{mm, Point};

use crate::ids::Uuid;
use runmat_rules::{Dim, Quantity};

fn entity(p: &Pool, name: &str) -> Uuid {
    p.entities
        .values()
        .find(|e| e.name == name)
        .map(|e| e.uuid)
        .unwrap_or_else(|| panic!("the standard pool has entity {name}"))
}

fn quantity(p: &mut Pool, part: Uuid, key: &str, v: f64, d: Dim, tol: f64) {
    p.parts.get_mut(&part).unwrap().params.insert(
        key.into(),
        Param::quantity(Quantity::new(v, d), Tolerance::pm_percent(tol)),
    );
}

fn describe(p: &mut Pool, part: Uuid, text: &str) {
    p.parts.get_mut(&part).unwrap().description = text.into();
}

/// A JST-SH-style 1.0 mm side-entry connector with two mechanical pads.
fn jst_sh(pins: u32) -> crate::library::Package {
    let half = (pins as f64 - 1.0) / 2.0 + 1.35;
    let mp = PadShape::RoundRect { w: mm(1.2), h: mm(1.8), ratio: 0.25 };
    footprint::smd_connector(
        "JST-SH",
        pins,
        1.0,
        (0.6, 1.55),
        &[(Point::mm(-half, 1.1), mp.clone(), None), (Point::mm(half, 1.1), mp, None)],
        (pins as f64 + 2.0, 4.25),
    )
}

/// Adds the quadcopter parts to the standard pool.
pub(super) fn add(p: &mut Pool) {
    use PinType::*;
    let r = entity(p, "R");
    let c = entity(p, "C");
    let l = entity(p, "L");
    let r0402 = p.add_package(footprint::from_string("R0402").unwrap());
    let c0402 = p.add_package(footprint::from_string("C0402").unwrap());
    let c0603 = p.add_package(footprint::from_string("C0603").unwrap());
    let c0805 = p.add_package(footprint::from_string("C0805").unwrap());
    let c1206 = p.add_package(footprint::from_string("C1206").unwrap());
    let c2512 = p.add_package(footprint::from_string("C2512").unwrap());
    let r2512 = p.add_package(footprint::from_string("R2512").unwrap());
    let l1210 = p.add_package(footprint::from_string("L1210").unwrap());

    for (mpn, ohms, tol, watts) in [
        ("R0402-10R", 10.0, 1.0, 0.0625),
        ("R0402-4K7", 4.7e3, 1.0, 0.0625),
        ("R0402-10K", 10e3, 1.0, 0.0625),
        ("R0402-22K", 22e3, 1.0, 0.0625),
        ("R0402-56K", 56e3, 1.0, 0.0625),
        ("R0402-52K3", 52.3e3, 1.0, 0.0625),
        ("R0402-200K", 200e3, 1.0, 0.0625),
    ] {
        if p.parts.values().any(|x| x.mpn == mpn) {
            continue;
        }
        let part = make::part_by_name(p, mpn, "Generic", r, r0402);
        quantity(p, part, "resistance", ohms, Dim::RESISTANCE, tol);
        quantity(p, part, "power_rating", watts, Dim::POWER, 0.0);
    }
    let shunt = make::part_by_name(p, "R2512-0M5", "Generic", r, r2512);
    quantity(p, shunt, "resistance", 0.5e-3, Dim::RESISTANCE, 1.0);
    quantity(p, shunt, "power_rating", 3.0, Dim::POWER, 0.0);
    describe(p, shunt, "Current shunt, 0.5 mOhm 1 %, 3 W, 2512");
    for (mpn, pkg, farads, tol, volts) in [
        ("C0402-12P", c0402, 12e-12, 5.0, 50.0),
        ("C0402-220P", c0402, 220e-12, 5.0, 50.0),
        ("C0402-10N", c0402, 10e-9, 10.0, 50.0),
        ("C0402-1U", c0402, 1e-6, 10.0, 16.0),
        ("C0603-1U-50V", c0603, 1e-6, 10.0, 50.0),
        ("C0603-4U7", c0603, 4.7e-6, 10.0, 25.0),
        ("C0805-22U", c0805, 22e-6, 20.0, 10.0),
        ("C1206-10U-50V", c1206, 10e-6, 10.0, 50.0),
        ("C2512-220U-35V", c2512, 220e-6, 20.0, 35.0),
    ] {
        let part = make::part_by_name(p, mpn, "Generic", c, pkg);
        quantity(p, part, "capacitance", farads, Dim::CAPACITANCE, tol);
        quantity(p, part, "voltage_rating", volts, Dim::VOLTAGE, 0.0);
    }
    let diode = make::simple_entity(p, "schottky", "D", &[("A", Passive), ("K", Passive)]);
    let sod = p.add_package(footprint::from_string("D1206").unwrap());
    let sk = make::part(p, "D-SCHOTTKY-1A", "Generic", diode, sod, &[("A", &["2"]), ("K", &["1"])]);
    describe(p, sk, "40 V 1 A Schottky rectifier");
    let l10 = make::part_by_name(p, "L1210-10U", "Generic", l, l1210);
    quantity(p, l10, "inductance", 10e-6, Dim::INDUCTANCE, 20.0);

    // Cortex-M4-class MCU, LQFP-48 7 x 7 mm, 0.5 mm pitch.
    let mut pins: Vec<(String, PinType, Vec<&str>)> = vec![
        ("VDD".into(), PowerIn, vec!["1", "24", "36", "48"]),
        ("VSS".into(), PowerIn, vec!["23", "35", "47"]),
        ("VDDA".into(), PowerIn, vec!["9"]),
        ("VSSA".into(), PowerIn, vec!["8"]),
        ("VCAP".into(), Passive, vec!["22"]),
        ("NRST".into(), Input, vec!["7"]),
        ("BOOT0".into(), Input, vec!["44"]),
        ("OSC_IN".into(), Input, vec!["5"]),
        ("OSC_OUT".into(), Output, vec!["6"]),
    ];
    let io: &[(&str, &str)] = &[
        ("PC13", "2"), ("PC14", "3"), ("PC15", "4"), ("PA0", "10"), ("PA1", "11"), ("PA2", "12"),
        ("PA3", "13"), ("PA4", "14"), ("PA5", "15"), ("PA6", "16"), ("PA7", "17"), ("PB0", "18"),
        ("PB1", "19"), ("PB2", "20"), ("PB10", "21"), ("PB12", "25"), ("PB13", "26"), ("PB14", "27"),
        ("PB15", "28"), ("PA8", "29"), ("PA9", "30"), ("PA10", "31"), ("PA11", "32"), ("PA12", "33"),
        ("PA13", "34"), ("PA14", "37"), ("PA15", "38"), ("PB3", "39"), ("PB4", "40"), ("PB5", "41"),
        ("PB6", "42"), ("PB7", "43"), ("PB8", "45"), ("PB9", "46"),
    ];
    for (n, pad) in io {
        pins.push((n.to_string(), Bidirectional, vec![pad]));
    }
    let defs: Vec<(&str, PinType)> = pins.iter().map(|(n, k, _)| (n.as_str(), *k)).collect();
    let mcu = make::simple_entity(p, "mcu-m4", "U", &defs);
    let lqfp48 = p.add_package(footprint::qfp(48, 7.0, 0.5, Density::B));
    let map: Vec<(&str, &[&str])> = pins.iter().map(|(n, _, v)| (n.as_str(), v.as_slice())).collect();
    let m4 = make::part(p, "MCU-M4-LQFP48", "Generic", mcu, lqfp48, &map);
    describe(p, m4, "32-bit Cortex-M4-class MCU, 168 MHz, FPU, LQFP-48");

    // Six-axis IMU (gyro + accelerometer), SPI, 14-pin 3 x 2.5 mm land grid.
    let imu = make::simple_entity(
        p,
        "imu-6axis",
        "U",
        &[("SDO", TriState), ("SDI", Input), ("INT1", TriState), ("VDDIO", PowerIn), ("GND", PowerIn), ("VDD", PowerIn),
          ("INT2", TriState), ("NC", NotConnected), ("SCLK", Input), ("CS", Input)],
    );
    let lga14 = p.add_package(footprint::dfn(14, (3.0, 2.5), 0.5, None, Density::B));
    let ip = make::part(
        p,
        "IMU-6AX-LGA14",
        "Generic",
        imu,
        lga14,
        &[("SDO", &["1"]), ("SDI", &["2"]), ("NC", &["3", "10", "11", "12"]), ("INT1", &["4"]), ("VDDIO", &["5"]),
          ("GND", &["6", "7"]), ("VDD", &["8"]), ("INT2", &["9"]), ("SCLK", &["13"]), ("CS", &["14"])],
    );
    describe(p, ip, "Six-axis MEMS IMU, +/-2000 dps gyro, +/-16 g accelerometer, SPI");

    // Barometer, I2C, 8-pin 2 x 2.5 mm land grid.
    let baro = make::simple_entity(
        p,
        "barometer",
        "U",
        &[("GND", PowerIn), ("CSB", Input), ("SDA", Bidirectional), ("SCL", Input), ("SDO", Input), ("VDDIO", PowerIn), ("VDD", PowerIn)],
    );
    let lga8 = p.add_package(footprint::dfn(8, (2.0, 2.5), 0.65, None, Density::B));
    let bp = make::part(
        p,
        "BARO-LGA8",
        "Generic",
        baro,
        lga8,
        &[("GND", &["1", "7"]), ("CSB", &["2"]), ("SDA", &["3"]), ("SCL", &["4"]), ("SDO", &["5"]), ("VDDIO", &["6"]), ("VDD", &["8"])],
    );
    describe(p, bp, "Barometric pressure sensor, 300-1100 hPa, I2C");

    // 36 V, 2 A synchronous buck in SOT-23-6 (BST, GND, FB, EN, VIN, SW).
    let buck = make::simple_entity(
        p,
        "buck-hv",
        "U",
        &[("BST", Passive), ("GND", PowerIn), ("FB", Input), ("EN", Input), ("VIN", PowerIn), ("SW", PowerOut)],
    );
    let sot6 = p.add_package(footprint::sot23(6, Density::B).unwrap());
    let bk = make::part(
        p,
        "BUCK-36V-2A",
        "Generic",
        buck,
        sot6,
        &[("BST", &["1"]), ("GND", &["2"]), ("FB", &["3"]), ("EN", &["4"]), ("VIN", &["5"]), ("SW", &["6"])],
    );
    describe(p, bk, "36 V 2 A synchronous buck, 500 kHz, 0.8 V reference, SOT-23-6");
    p.parts.get_mut(&bk).unwrap().thermal = ThermalData { theta_jc: Some(40.0), theta_jb: Some(30.0), theta_ja: Some(110.0), p_max: Some(1.0), tj_max: Some(150.0) };

    // 3.3 V 300 mA LDO in SOT-23-5 (VIN, GND, EN, NC, VOUT).
    let ldo = make::simple_entity(
        p,
        "ldo",
        "U",
        &[("VIN", PowerIn), ("GND", PowerIn), ("EN", Input), ("NC", NotConnected), ("VOUT", PowerOut)],
    );
    let sot5 = p.add_package(footprint::sot23(5, Density::B).unwrap());
    let lp = make::part(
        p,
        "LDO-3V3-SOT23",
        "Generic",
        ldo,
        sot5,
        &[("VIN", &["1"]), ("GND", &["2"]), ("EN", &["3"]), ("NC", &["4"]), ("VOUT", &["5"])],
    );
    describe(p, lp, "3.3 V 300 mA low-dropout regulator, 40 V input, SOT-23-5");

    // USB-C receptacle, the standard pool's package under a generic part number.
    let usbc = entity(p, "usb-c");
    let usb_pkg = p.parts.values().find(|x| x.mpn == "USB4105").map(|x| x.package).expect("USB-C package");
    let up = make::part(
        p,
        "USBC-RCPT-16P",
        "Generic",
        usbc,
        usb_pkg,
        &[("GND", &["1", "12"]), ("VBUS", &["2", "11"]), ("CC1", &["3"]), ("DP", &["4", "6"]), ("DN", &["5", "7"]),
          ("SBU1", &["8"]), ("CC2", &["9"]), ("SBU2", &["10"]), ("SHIELD", &["MP1", "MP2"])],
    );
    describe(p, up, "USB Type-C receptacle, USB 2.0, mid-mount");

    // JST-SH 1.0 mm connectors: four pins (receiver, telemetry) and eight (the ESC harness).
    for n in [4u32, 8] {
        let names: Vec<String> = (1..=n).map(|k| k.to_string()).collect();
        let defs: Vec<(&str, PinType)> = names.iter().map(|s| (s.as_str(), Passive)).collect();
        let e = make::simple_entity(p, &format!("conn-{n}"), "J", &defs);
        let pkg = p.add_package(jst_sh(n));
        let part = make::part_by_name(p, &format!("CONN-SH-{n}P"), "Generic", e, pkg);
        describe(p, part, &format!("{n}-pin 1.0 mm wire-to-board connector, side entry"));
    }

    // Wire pads: plated holes the battery leads (12 AWG) and motor leads (20 AWG) solder into.
    let pad_e = make::simple_entity(p, "wire-pad", "P", &[("1", Passive)]);
    let big = p.add_package(footprint::mounting_hole(2.2, 4.2));
    let bat = make::part_by_name(p, "PAD-WIRE-12AWG", "Generic", pad_e, big);
    describe(p, bat, "Solder pad for a 12 AWG battery lead");
    let small = p.add_package(footprint::mounting_hole(1.2, 2.6));
    let mot = make::part_by_name(p, "PAD-WIRE-20AWG", "Generic", pad_e, small);
    describe(p, mot, "Solder pad for a 20 AWG motor lead");
    let m3 = p.add_package(footprint::mounting_hole(3.6, 4.6));
    let hole = entity(p, "mounting-hole");
    let gh = make::part_by_name(p, "MH-M3-GROMMET", "Generic", hole, m3);
    describe(p, gh, "Stack mounting hole for an M3 grommet (3.6 mm)");

    // ESC channel MCU: the standard pool's QFN-32 MCU type.
    // Three-phase gate driver with bootstrap high sides, QFN-24 4 x 4 mm with exposed pad.
    let drv = make::simple_entity(
        p,
        "gate-driver-3ph",
        "U",
        &[("INHA", Input), ("INLA", Input), ("INHB", Input), ("INLB", Input), ("INHC", Input), ("INLC", Input),
          ("EN", Input), ("GND", PowerIn), ("VCC", PowerIn), ("VDRV", PowerOut), ("BSTA", Passive), ("GHA", Output),
          ("SHA", Passive), ("GLA", Output), ("BSTB", Passive), ("GHB", Output), ("SHB", Passive), ("GLB", Output),
          ("BSTC", Passive), ("GHC", Output), ("SHC", Passive), ("GLC", Output), ("FAULT", OpenCollector)],
    );
    let qfn24 = p.add_package(footprint::qfn(24, 4.0, 0.5, Some(2.6), Density::B));
    let dp = make::part(
        p,
        "GATEDRV-3PH-QFN24",
        "Generic",
        drv,
        qfn24,
        &[("INHA", &["1"]), ("INLA", &["2"]), ("INHB", &["3"]), ("INLB", &["4"]), ("INHC", &["5"]), ("INLC", &["6"]),
          ("EN", &["7"]), ("GND", &["8", "23", "25"]), ("VCC", &["9"]), ("VDRV", &["10"]), ("BSTC", &["11"]), ("GHC", &["12"]),
          ("SHC", &["13"]), ("GLC", &["14"]), ("BSTB", &["15"]), ("GHB", &["16"]), ("SHB", &["17"]), ("GLB", &["18"]),
          ("BSTA", &["19"]), ("GHA", &["20"]), ("SHA", &["21"]), ("GLA", &["22"]), ("FAULT", &["24"])],
    );
    describe(p, dp, "Three-phase half-bridge gate driver, 40 V, 1 A/1.5 A, bootstrap, QFN-24");

    // 40 V N-channel power MOSFET, 3.3 x 3.3 mm 8-pin power package (S 1-3, G 4, D 5-8 and tab).
    let fet = make::simple_entity(p, "nmos", "Q", &[("D", Passive), ("G", Input), ("S", Passive)]);
    let dfn8 = p.add_package(footprint::dfn(8, (3.3, 3.3), 0.65, Some((2.4, 1.8)), Density::B));
    let q = make::part(p, "NMOS-40V-DFN8", "Generic", fet, dfn8, &[("S", &["1", "2", "3"]), ("G", &["4"]), ("D", &["5", "6", "7", "8", "9"])]);
    describe(p, q, "40 V N-channel MOSFET, 1.6 mOhm at 10 V, 3.3 x 3.3 mm power package");
    quantity(p, q, "rds_on", 1.6e-3, Dim::RESISTANCE, 15.0);
    p.parts.get_mut(&q).unwrap().thermal = ThermalData { theta_jc: Some(1.8), theta_jb: Some(4.0), theta_ja: Some(55.0), p_max: Some(3.0), tj_max: Some(150.0) };
}
