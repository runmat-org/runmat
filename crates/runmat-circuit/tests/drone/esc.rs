//! The 4-in-1 ESC: 52 x 52 mm, four layers (signal and phase copper, ground, battery, signal and
//! ground). Channel 0 is placed and routed in its room, the lower-left quadrant, around its
//! stack hole; Copy Room Formats turns that placement and routing about the board's center into
//! the other three quadrants. The battery pads, bulk capacitors, LDO and harness run up the
//! board's center column.

use super::build::Build;
use serde_json::json;
use std::path::Path;

/// Channel 0's parts, room-local millimetres from the room's lower-left corner (the board's
/// corner): the three half bridges (low side below its high side) along the bottom edge either
/// side of the stack hole, the motor pads below the hole, the shunt up the outer edge, the gate
/// driver and its gate resistors above the B and C bridges, and the MCU above the hole.
const CHANNEL: &[(&str, f64, f64, f64)] = &[
    ("QLA", 3.0, 3.0, 0.0),
    ("QHA", 3.0, 7.4, 0.0),
    ("QLB", 15.4, 3.0, 0.0),
    ("QHB", 15.4, 7.4, 0.0),
    ("QLC", 19.6, 3.0, 0.0),
    ("QHC", 19.6, 7.4, 0.0),
    ("PA", 7.6, 2.4, 0.0),
    ("PB", 10.8, 2.4, 0.0),
    ("PC", 10.8, 5.6, 0.0),
    ("RSH", 2.6, 14.4, 90.0),
    ("CB1", 17.5, 11.6, 0.0),
    ("DRV", 16.8, 16.4, 0.0),
    ("CVCC", 20.6, 14.6, 90.0),
    ("CDRV", 20.6, 17.8, 90.0),
    ("CBA", 14.6, 20.6, 90.0),
    ("CBB", 15.8, 20.6, 90.0),
    ("CBC", 17.0, 20.6, 90.0),
    ("RGHA", 12.6, 13.4, 0.0),
    ("RGLA", 12.6, 14.6, 0.0),
    ("RGHB", 12.6, 15.8, 0.0),
    ("RGLB", 12.6, 17.0, 0.0),
    ("RGHC", 12.6, 18.2, 0.0),
    ("RGLC", 12.6, 19.4, 0.0),
    ("MCU", 8.0, 18.6, 0.0),
    ("CMCU", 5.0, 14.0, 90.0),
    ("RBOOT", 4.2, 20.8, 90.0),
    ("RIS", 6.6, 14.0, 90.0),
    ("CIS", 7.8, 14.0, 90.0),
];

const ROOM: f64 = 22.0;
/// Room corners: lower-left quadrant first, then counter-clockwise (each a quarter turn of the one before).
const ROOMS: &[(&str, f64, f64)] = &[("ch[0]", -25.5, -25.5), ("ch[1]", 3.5, -25.5), ("ch[2]", 3.5, 3.5), ("ch[3]", -25.5, 3.5)];

/// The center column, absolute millimetres.
const MAIN: &[(&str, f64, f64, f64)] = &[
    ("P1", 0.0, -22.6, 0.0),
    ("P2", 0.0, -17.6, 0.0),
    ("C1", 0.0, -10.8, 90.0),
    ("C2", 0.0, -3.2, 90.0),
    ("C3", 0.0, 1.8, 0.0),
    ("U1", 0.0, 4.2, 0.0),
    ("C4", 0.0, 6.6, 0.0),
    ("J1", 0.0, 18.0, 90.0),
    ("H1", -15.25, -15.25, 0.0),
    ("H2", 15.25, -15.25, 0.0),
    ("H3", 15.25, 15.25, 0.0),
    ("H4", -15.25, 15.25, 0.0),
];

pub fn build(dir: &Path) -> Build {
    let start = &std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/drone/esc.start.circuit.layout")).unwrap();
    let mut b = Build::new(dir, "esc", start);
    b.start(4);
    for (name, x0, y0) in ROOMS {
        let (x1, y1) = (x0 + ROOM, y0 + ROOM);
        b.run("board.room", json!({ "name": name, "points": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]] }), &[]);
    }
    let (x0, y0) = (ROOMS[0].1, ROOMS[0].2);
    for (path, x, y, r) in CHANNEL {
        b.place(&format!("/ch[0]/{path}"), x0 + x, y0 + y, *r);
    }
    for (path, x, y, r) in MAIN {
        b.place(path, *x, *y, *r);
    }
    let all = [[-25.6, -23.6], [-23.6, -25.6], [23.6, -25.6], [25.6, -23.6], [25.6, 23.6], [23.6, 25.6], [-23.6, 25.6], [-25.6, 23.6]];
    b.pour("GND", "In1.Cu", &all, "relief");
    b.pour("VBAT", "In2.Cu", &all, "relief");
    b.pour("GND", "B.Cu", &all, "relief-tht");
    let local: Vec<String> = ["PHA", "PHB", "PHC", "SRC", "GHA", "GLA", "GHB", "GLB", "GHC", "GLC", "GHA_D", "GLA_D", "GHB_D", "GLB_D", "GHC_D", "GLC_D", "BSTA", "BSTB", "BSTC", "VDRV", "INHA", "INLA", "INHB", "INLB", "INHC", "INLC", "DRV_EN", "FAULT", "ISENSE", "BOOT0"]
        .iter()
        .map(|n| format!("net:/ch[0]/{n}"))
        .collect();
    b.run("board.autoroute", json!({ "nets": local }), &[]);
    b.run("board.copy-room-formats", json!({ "source": ["room:ch[0]"], "rotate": true }), &[]);
    b.finish();
    b
}
