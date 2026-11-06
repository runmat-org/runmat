//! KiCad schematic connectivity (`tests/kicad/connectivity.kicad_sch`): wires join only where
//! their ends meet each other, a pin or a sheet pin; a junction joins the wires through it;
//! labels attach anywhere along a wire; crossing wires stay apart; collinear overlapping wires
//! become one wire with only its outer ends; collinear wires touching at a pin stay joined.

use runmat_circuit::elab::{elaborate, ElabOptions, Netlist};
use runmat_circuit::kicad::sch::read_hierarchy;
use std::collections::{BTreeMap, BTreeSet};

fn netlist() -> Netlist {
    let mut files = BTreeMap::new();
    files.insert(
        "connectivity.kicad_sch".to_string(),
        include_str!("kicad/connectivity.kicad_sch").to_string(),
    );
    let imp = read_hierarchy(&files, "connectivity.kicad_sch").unwrap();
    elaborate(&imp.design, &imp.pool, &ElabOptions::default(), Some(&imp.lock))
}

/// The pins sharing a net with `pin` (`R1.2`), itself included.
fn net_of(nl: &Netlist, pin: &str) -> BTreeSet<String> {
    let name = |&(c, p): &(usize, usize)| {
        let comp = &nl.components[c];
        format!("{}.{}", comp.designator, comp.pins[p].pin)
    };
    nl.nets
        .iter()
        .map(|n| n.pins.iter().map(name).collect::<BTreeSet<_>>())
        .find(|s| s.contains(pin))
        .unwrap_or_else(|| [pin.to_string()].into())
}

fn set(pins: &[&str]) -> BTreeSet<String> {
    pins.iter().map(|s| s.to_string()).collect()
}

fn net_name(nl: &Netlist, pin: &str) -> String {
    nl.nets
        .iter()
        .find(|n| {
            n.pins.iter().any(|&(c, p)| {
                let comp = &nl.components[c];
                format!("{}.{}", comp.designator, comp.pins[p].pin) == pin
            })
        })
        .map(|n| n.name.clone())
        .unwrap_or_default()
}

#[test]
fn a_wire_ending_on_another_wires_middle_stays_apart_without_a_junction() {
    let nl = netlist();
    // sig0- jogs to R3.1, which lies on sig0+'s wire: two nets, as KiCad keeps usb+ and usb-.
    assert_eq!(net_of(&nl, "R1.2"), set(&["R1.2", "R4.1"]));
    assert_eq!(net_of(&nl, "R2.2"), set(&["R2.2", "R3.1"]));
    assert_eq!(net_name(&nl, "R1.2"), "sig0+");
    assert_eq!(net_name(&nl, "R2.2"), "sig0-");
}

#[test]
fn a_junction_joins_the_wires_through_it() {
    let nl = netlist();
    assert_eq!(
        net_of(&nl, "R11.2"),
        set(&["R11.2", "R12.2", "R13.1", "R14.1"])
    );
}

#[test]
fn crossing_wires_stay_apart() {
    let nl = netlist();
    assert_eq!(net_of(&nl, "R21.2"), set(&["R21.2", "R22.1"]));
    assert_eq!(net_of(&nl, "R23.2"), set(&["R23.2", "R24.1"]));
}

#[test]
fn collinear_overlapping_wires_become_one_wire_with_only_its_outer_ends() {
    let nl = netlist();
    assert_eq!(net_of(&nl, "R31.2"), set(&["R31.2", "R33.1"]));
    assert_eq!(net_of(&nl, "R32.1"), set(&["R32.1"]));
}

#[test]
fn collinear_wires_touching_at_a_pin_stay_joined_to_it() {
    let nl = netlist();
    assert_eq!(
        net_of(&nl, "R41.2"),
        set(&["R41.2", "R42.1", "R43.1"])
    );
}
