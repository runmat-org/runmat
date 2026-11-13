//! Parts of the showcase data-acquisition board (`tests/projects/daq`): a single op-amp, a
//! 16-bit SAR ADC, a voltage reference, a mounting hole and the passive values the board's
//! channels, regulators and USB-C sink use.

use crate::ids::Uuid;
use crate::ir::*;
use crate::library::footprint::{self, Density};
use crate::library::{make, PinType, Pool};
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

/// Adds the showcase parts to the standard pool.
pub(super) fn add(p: &mut Pool) {
    use PinType::*;
    let r = entity(p, "R");
    let c = entity(p, "C");
    let r0402 = p.add_package(footprint::from_string("R0402").unwrap());
    let c0402 = p.add_package(footprint::from_string("C0402").unwrap());
    let c0805 = p.add_package(footprint::from_string("C0805").unwrap());
    for (mpn, ohms, tol) in [
        ("R0402-49R9", 49.9, 1.0),
        ("R0402-1K", 1e3, 1.0),
        ("R0402-5K1", 5.1e3, 1.0),
        ("R0402-32K4", 32.4e3, 1.0),
        ("R0402-49K9", 49.9e3, 1.0),
        ("R0402-100K", 100e3, 1.0),
    ] {
        let part = make::part_by_name(p, mpn, "Generic", r, r0402);
        quantity(p, part, "resistance", ohms, Dim::RESISTANCE, tol);
        quantity(p, part, "power_rating", 0.0625, Dim::POWER, 0.0);
    }
    for (mpn, farads, tol, volts) in [
        ("C0402-1N", 1e-9, 5.0, 50.0),
        ("C0402-2N2", 2.2e-9, 5.0, 50.0),
        ("C0402-3N3", 3.3e-9, 10.0, 50.0),
    ] {
        let part = make::part_by_name(p, mpn, "Generic", c, c0402);
        quantity(p, part, "capacitance", farads, Dim::CAPACITANCE, tol);
        quantity(p, part, "voltage_rating", volts, Dim::VOLTAGE, 0.0);
    }
    let c1u = make::part_by_name(p, "C0805-1U", "Generic", c, c0805);
    quantity(p, c1u, "capacitance", 1e-6, Dim::CAPACITANCE, 10.0);
    quantity(p, c1u, "voltage_rating", 25.0, Dim::VOLTAGE, 0.0);

    // Single rail-to-rail op-amp, SOT-23-5 (OUT, V-, IN+, IN-, V+).
    let opamp = make::simple_entity(
        p,
        "opamp-single",
        "U",
        &[
            ("IN+", Input),
            ("IN-", Input),
            ("OUT", Output),
            ("V+", PowerIn),
            ("V-", PowerIn),
        ],
    );
    let sot5 = p.add_package(footprint::sot23(5, Density::B).unwrap());
    make::part(
        p,
        "OPA171",
        "Texas Instruments",
        opamp,
        sot5,
        &[
            ("OUT", &["1"]),
            ("V-", &["2"]),
            ("IN+", &["3"]),
            ("IN-", &["4"]),
            ("V+", &["5"]),
        ],
    );

    // 16-bit SAR ADC, VSSOP-10, SPI.
    let adc = make::simple_entity(
        p,
        "sar-adc",
        "U",
        &[
            ("REF", PowerIn),
            ("AINP", Input),
            ("AINN", Input),
            ("GND", PowerIn),
            ("CONVST", Input),
            ("SCLK", Input),
            ("DOUT", TriState),
            ("DIN", Input),
            ("DVDD", PowerIn),
            ("AVDD", PowerIn),
        ],
    );
    let msop10 = p.add_package(footprint::msop(10, Density::B));
    make::part(
        p,
        "ADS8860",
        "Texas Instruments",
        adc,
        msop10,
        &[
            ("REF", &["1"]),
            ("AINP", &["2"]),
            ("AINN", &["3"]),
            ("GND", &["4"]),
            ("CONVST", &["5"]),
            ("SCLK", &["6"]),
            ("DOUT", &["7"]),
            ("DIN", &["8"]),
            ("DVDD", &["9"]),
            ("AVDD", &["10"]),
        ],
    );

    // 4.096 V reference, SOIC-8 (pins 1, 7 and 8 not connected inside).
    let reference = make::simple_entity(
        p,
        "voltage-reference",
        "U",
        &[
            ("VIN", PowerIn),
            ("TEMP", Output),
            ("GND", PowerIn),
            ("NR", Passive),
            ("VOUT", PowerOut),
            ("DNC", NotConnected),
        ],
    );
    let soic8 = p.add_package(footprint::soic(8, true, Density::B));
    make::part(
        p,
        "REF5040",
        "Texas Instruments",
        reference,
        soic8,
        &[
            ("DNC", &["1", "7", "8"]),
            ("VIN", &["2"]),
            ("TEMP", &["3"]),
            ("GND", &["4"]),
            ("NR", &["5"]),
            ("VOUT", &["6"]),
        ],
    );

    // The FPGA again with its power balls as FPGAs place them: core supply and ground in a
    // checkerboard in the centre (rows F to L, columns 6 to 11), the I/O supply in rows D and
    // N, and the I/O around them, IO0 at A1.
    let fpga = entity(p, "fpga");
    let bga = p
        .packages
        .values()
        .find(|k| k.name.starts_with("BGA-256"))
        .map(|k| k.uuid)
        .expect("the standard pool has the BGA-256 package");
    let balls: Vec<String> = p.packages[&bga]
        .pads
        .iter()
        .map(|x| x.name.clone())
        .collect();
    let (mut core, mut io_supply, mut ground, mut io) = (vec![], vec![], vec![], vec![]);
    for (k, ball) in balls.iter().enumerate() {
        let (row, col) = (k / 16, k % 16 + 1);
        let centre = (5..=10).contains(&row) && (6..=11).contains(&col);
        if centre {
            if (row + col) % 2 == 0 {
                core.push(ball.as_str());
            } else {
                ground.push(ball.as_str());
            }
        } else if (row == 3 || row == 12) && (4..=12).contains(&col) {
            io_supply.push(ball.as_str());
        } else if (row == 2 || row == 13) && col == 8 {
            ground.push(ball.as_str());
        } else {
            io.push(ball.as_str());
        }
    }
    let names: Vec<String> = (0..200).map(|i| format!("IO{i}")).collect();
    let mut map: Vec<(&str, Vec<&str>)> = names
        .iter()
        .zip(io.iter())
        .map(|(n, b)| (n.as_str(), vec![*b]))
        .collect();
    map.push(("VCCINT", core));
    map.push(("VCCIO", io_supply));
    map.push(("GND", ground));
    let map: Vec<(&str, &[&str])> = map.iter().map(|(n, v)| (*n, v.as_slice())).collect();
    let part = make::part(p, "FPGA-BGA256-C", "Generic", fpga, bga, &map);
    p.parts.get_mut(&part).unwrap().thermal = crate::library::ThermalData {
        theta_jc: Some(6.5),
        theta_jb: Some(9.0),
        theta_ja: Some(24.0),
        p_max: Some(4.0),
        tj_max: Some(100.0),
    };

    // M3 mounting hole with a plated pad.
    let hole = make::simple_entity(p, "mounting-hole", "H", &[("1", Passive)]);
    let mh = p.add_package(footprint::mounting_hole(3.2, 6.0));
    make::part_by_name(p, "MH-M3", "Generic", hole, mh);
}
