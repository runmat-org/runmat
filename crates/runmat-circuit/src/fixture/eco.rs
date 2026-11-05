//! The ECO scenario and the large board's projection.

use super::*;
use crate::elab::{elaborate, ElabOptions, Netlist};
use crate::ir::*;
use crate::layout::Board;
use crate::library::Pool;

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
    d.net_classes
        .get_mut("power")
        .unwrap()
        .params
        .insert("track_width".into(), Param::text("0.40 mm"));
    d.net_classes.insert(
        "analog".into(),
        NetClass {
            name: "analog".into(),
            params: [("clearance".to_owned(), Param::text("0.20 mm"))]
                .into_iter()
                .collect(),
            source: Default::default(),
        },
    );
    d.variants.insert(
        "lite".into(),
        VariantDef {
            name: "lite".into(),
            base: None,
            overrides: [("/ch[0]/CD2".to_owned(), Fitting::NotFitted)]
                .into_iter()
                .collect(),
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
    lock.reconcile(
        &synced_netlist.lock_items(),
        &mut DerivedUuids { namespace: 7 },
    );
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
        eco::apply(
            &mut board,
            &plan,
            &plan.ids(),
            &pool,
            &ApplyOptions::default(),
        )
        .expect("the first sync validates");
        eco::record_sync(&mut board, &s);
    }
    // Board-side edits.
    let by_path = |nl: &Netlist, p: &str| {
        nl.components
            .iter()
            .find(|c| c.path.to_string() == p)
            .cloned()
            .unwrap()
    };
    let u = by_path(&synced_netlist, "/ch[1]/U");
    let fid = board
        .footprints
        .values()
        .find(|f| f.reference == u.designator)
        .unwrap()
        .id;
    {
        let f = board.footprints.get_mut(&fid).unwrap();
        for pin in u.pins.iter().filter(|p| p.gate == "B") {
            let other = u
                .pins
                .iter()
                .find(|q| q.gate == "C" && q.pin == pin.pin)
                .unwrap();
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
        board
            .footprints
            .values_mut()
            .find(|f| f.reference == d)
            .unwrap()
            .reference = to.into();
    }
    // The schematic renumbered channel 0's RG too.
    lock.entries
        .get_mut(&crate::ids::NamePath::parse("/ch[0]/RG"))
        .unwrap()
        .designator = Some("R60".into());
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

/// The large design's projection as the Desktop reads it (netlist, schematic and board with its
/// DRC; no ERC or review findings): the board scale scene's input, `channels` channels strong.
pub fn large_index(params: &LargeParams) -> crate::index::CircuitIndex {
    use crate::board_rules::{BoardGraph, BoardRules};
    let (pool, design, nl, mut board) = large(params);
    let mut index = crate::index::build_index(&design, &nl, &pool, &nl.diagnostics, &[]);
    let (mut rules, _) = BoardRules::new(&board.limits, &board.rules);
    rules.rebuild(&BoardGraph::new(&board, &[]));
    let fills = crate::zone::fill_zones(&mut board, &rules, &[]);
    let drc = crate::drc::Drc::run(&board, &rules, &[], &fills);
    crate::index::add_board(&mut index, &board, &rules, &drc);
    index
}
