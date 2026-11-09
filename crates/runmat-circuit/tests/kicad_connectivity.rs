//! KiCad schematic connectivity, checked against KiCad itself.
//!
//! `tests/kicad/connectivity.kicad_sch` holds one small case per rule in `kicad::sch_connect`,
//! drawn in the form KiCad's editor saves. `tests/kicad/connectivity.net` is KiCad's own
//! netlist of that file (`kicad-cli sch export netlist`, KiCad 10.0.7; only the source path is
//! shortened). The import must put every pin on the net KiCad puts it on. The one case where
//! KiCad's netlist and its manual disagree (pins on a bus) is left out of the comparison and
//! checked against the manual on its own.

use runmat_circuit::elab::{elaborate, ElabOptions, Netlist};
use runmat_circuit::kicad::sch::read_hierarchy;
use runmat_circuit::kicad::sexpr;
use std::collections::{BTreeMap, BTreeSet};

fn netlist() -> Netlist {
    let mut files = BTreeMap::new();
    files.insert(
        "connectivity.kicad_sch".to_string(),
        include_str!("kicad/connectivity.kicad_sch").to_string(),
    );
    let imp = read_hierarchy(&files, "connectivity.kicad_sch").unwrap();
    elaborate(
        &imp.design,
        &imp.pool,
        &ElabOptions::default(),
        Some(&imp.lock),
    )
}

fn pin_name(nl: &Netlist, (c, p): (usize, usize)) -> String {
    let comp = &nl.components[c];
    format!("{}.{}", comp.designator, comp.pins[p].pin)
}

/// The pins sharing a net with `pin` (`R1.2`), itself included.
fn net_of(nl: &Netlist, pin: &str) -> BTreeSet<String> {
    nl.nets
        .iter()
        .map(|n| {
            n.pins
                .iter()
                .map(|&cp| pin_name(nl, cp))
                .collect::<BTreeSet<_>>()
        })
        .find(|s| s.contains(pin))
        .unwrap_or_else(|| [pin.to_string()].into())
}

fn net_name(nl: &Netlist, pin: &str) -> String {
    nl.nets
        .iter()
        .find(|n| n.pins.iter().any(|&cp| pin_name(nl, cp) == pin))
        .map(|n| n.name.clone())
        .unwrap_or_default()
}

fn set(pins: &[&str]) -> BTreeSet<String> {
    pins.iter().map(|s| s.to_string()).collect()
}

/// KiCad's nets: name and pins (`R1.2`) of each.
fn kicad_nets() -> Vec<(String, BTreeSet<String>)> {
    let export = sexpr::parse(include_str!("kicad/connectivity.net")).unwrap();
    assert_eq!(export.head(), Some("export"));
    let nets = export.child("nets").expect("nets");
    nets.children("net")
        .map(|n| {
            let pins = n
                .children("node")
                .map(|d| {
                    format!(
                        "{}.{}",
                        d.child_arg("ref").unwrap(),
                        d.child_arg("pin").unwrap()
                    )
                })
                .collect();
            (n.child_arg("name").unwrap().to_string(), pins)
        })
        .collect()
}

/// The pins on a bus: KiCad's netlist joins them, the manual says the connection is ignored.
const BUS_CASE: [&str; 2] = ["R61.2", "R62.1"];

#[test]
fn every_pin_is_on_the_net_kicad_puts_it_on() {
    let nl = netlist();
    let nets = kicad_nets();
    assert!(nets.len() > 50, "KiCad's netlist has {} nets", nets.len());
    let mut wrong = Vec::new();
    for (name, pins) in &nets {
        if pins.iter().any(|p| BUS_CASE.contains(&p.as_str())) {
            continue;
        }
        let first = pins.iter().next().unwrap();
        let ours = net_of(&nl, first);
        if &ours != pins {
            wrong.push(format!("KiCad {name}: {pins:?}; ours: {ours:?}"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_pin_or_wire_end_on_another_wires_middle_stays_apart_without_a_junction() {
    let nl = netlist();
    // sigB jogs to R3.2, which lies on sigA's wire: two nets, as KiCad keeps the flight
    // controller's usb+ and usb- apart. R5's wire ends on sigA's middle and joins nothing.
    assert_eq!(net_of(&nl, "R1.2"), set(&["R1.2", "R4.1"]));
    assert_eq!(net_of(&nl, "R2.2"), set(&["R2.2", "R3.2"]));
    assert_eq!(net_of(&nl, "R5.2"), set(&["R5.2"]));
    // With a junction at that pin, everything there joins.
    assert_eq!(
        net_of(&nl, "R11.2"),
        set(&["R11.2", "R12.2", "R13.2", "R14.1"])
    );
}

#[test]
fn collinear_overlapping_wires_connect_only_at_their_own_ends() {
    let nl = netlist();
    assert_eq!(net_of(&nl, "R31.2"), set(&["R31.2", "R32.1"]));
    assert_eq!(net_of(&nl, "R33.1"), set(&["R33.1"]));
}

#[test]
fn labels_name_wires_and_join_the_wires_under_their_anchor() {
    let nl = netlist();
    assert_eq!(net_of(&nl, "R51.2"), set(&["R51.2", "R53.2"]));
    assert_eq!(net_name(&nl, "R51.2"), "LBL");
    assert_eq!(
        net_of(&nl, "R55.2"),
        set(&["R55.2", "R56.1", "R57.2", "R58.1"])
    );
}

#[test]
fn power_symbols_join_by_value_and_power_flags_join_nothing() {
    let nl = netlist();
    assert_eq!(net_of(&nl, "R71.2"), set(&["R71.2", "R72.2", "R75.2"]));
    assert_eq!(net_of(&nl, "R73.1"), set(&["R73.1", "R74.1"]));
    assert_eq!(net_of(&nl, "R76.2"), set(&["R76.2"]));
}

#[test]
fn a_pin_on_a_bus_does_not_connect_to_it() {
    // The manual: "It is not possible to connect a pin directly to a bus; this type of
    // connection will be ignored by KiCad."
    let nl = netlist();
    assert_eq!(net_of(&nl, "R61.2"), set(&["R61.2"]));
    assert_eq!(net_of(&nl, "R62.1"), set(&["R62.1"]));
}
