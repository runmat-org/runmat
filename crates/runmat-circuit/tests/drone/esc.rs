//! The 4-in-1 ESC: 52 x 52 mm, six layers on 1.6 mm with 2 oz outer copper (the start file's
//! stack-up): top for the bridges, phase and shunt copper; In1 a ground plane 0.11 mm under the
//! top; In2 and In3 signal; In4 the battery plane; bottom the gate drivers, signal and ground.
//! Channel 0 is placed and routed in its room, the lower-left quadrant, around its stack hole;
//! Copy Room Formats turns that placement and routing about the board's center into the other
//! three quadrants. The battery pads, bulk capacitors, LDO and harness run up the
//! board's center column.

use super::build::Build;
use serde_json::json;
use std::path::Path;

/// Channel 0's parts, room-local millimetres from the channel's origin (the board's corner), and
/// whether each sits on the bottom side. Each half bridge stands as a stack: the low side's drain
/// faces the high side's sources, so the phase node is the short copper between them, the high
/// side's drain faces the VBAT vias and the low side's sources face the shunt's pad 1, which
/// sits over the source pins. Bridges A and B sit on the bottom edge either side of the stack
/// hole with their motor pads beside the phase nodes; bridge C stands in the inner corner with
/// its pad beside it. Each gate leaves its FET at the end of the pin row away from the source and
/// phase copper: A's gates go west to vias and the gate resistors under the bridge, B's go west
/// to resistors beside the stack hole and in the bottom corridor, C's go east to resistors on
/// the channel's inner edge. The bottom carries the gate driver under bridge B with phases A
/// and B's bootstrap capacitors west of it and phase C's beside its east side, outside the
/// heatsink pedestal (|x|, |y| > 19.5 mm on the board); its supply and drive capacitors sit on
/// the top over its corner. The MCU with its filter and boot parts sits above the shunts. VBAT and GND reach the
/// inner planes through vias.
const CHANNEL: &[(&str, f64, f64, f64, bool)] = &[
    ("QHA", 2.8, 2.6, 270.0, false),
    ("QLA", 2.8, 7.4, 270.0, false),
    ("QHB", 15.78, 2.6, 270.0, false),
    ("QLB", 15.78, 7.4, 270.0, false),
    ("QLC", 20.0, 13.0, 90.0, false),
    ("QHC", 20.0, 17.8, 90.0, false),
    ("PA", 6.4, 5.0, 0.0, false),
    ("PB", 10.25, 5.0, 0.0, false),
    ("PC", 16.4, 15.9, 0.0, false),
    ("RSHA", 4.55, 11.0, 0.0, false),
    ("RSHB", 15.15, 11.0, 180.0, false),
    ("RSHC", 20.0, 9.4, 0.0, false),
    ("CB1", 13.6, 15.2, 90.0, false),
    ("MCU", 8.0, 18.6, 0.0, false),
    ("CMCU", 5.2, 14.0, 90.0, false),
    ("RBOOT", 4.2, 20.8, 90.0, false),
    ("RRST", 4.2, 17.9, 90.0, false),
    ("CRST", 4.2, 16.3, 270.0, false),
    ("RISA", 6.6, 14.0, 90.0, false),
    ("CISA", 7.8, 14.0, 90.0, false),
    ("RISB", 9.0, 14.0, 90.0, false),
    ("CISB", 10.2, 14.0, 90.0, false),
    ("RISC", 13.0, 19.5, 90.0, false),
    ("CISC", 14.2, 19.5, 90.0, false),
    ("RGHA", 0.75, 3.05, 270.0, true),
    ("RGLA", 0.75, 7.85, 270.0, true),
    ("RGHB", 12.65, 3.05, 0.0, false),
    ("RGLB", 13.35, 8.6, 90.0, false),
    ("RGHC", 22.5, 17.1, 270.0, false),
    ("RGLC", 22.5, 14.3, 270.0, false),
    ("DRV", 17.6, 3.0, 0.0, true),
    ("CVCC", 20.1, 0.6, 0.0, false),
    ("CDRV", 19.22, 2.9, 0.0, false),
    ("CBA", 8.33, 4.8, 90.0, true),
    ("CBB", 8.33, 3.0, 90.0, true),
    ("CBC", 21.8, 4.5, 0.0, true),
];

/// Channel 0's origin: CHANNEL's room-local coordinates start here.
const ORIGIN: (f64, f64) = (-25.5, -25.5);
/// Channel 0's room, board millimetres: its quadrant out to the board edge and in to 1.9 mm
/// from the center lines, so the gate tracks on its inner edge copy with it. The other rooms are
/// its quarter turns about the board's center, counter-clockwise.
const ROOM_SPAN: (f64, f64) = (-25.9, -1.9);
const ROOMS: [&str; 4] = ["ch[0]", "ch[1]", "ch[2]", "ch[3]"];

/// `p` turned `k` quarter turns counter-clockwise about the board's center.
fn turn([x, y]: [f64; 2], k: usize) -> [f64; 2] {
    (0..k).fold([x, y], |[x, y], _| [-y, x])
}

/// The center column, absolute millimetres.
const MAIN: &[(&str, f64, f64, f64)] = &[
    ("P1", 0.0, -22.6, 0.0),
    ("P2", 0.0, -17.6, 0.0),
    ("C1", 0.0, -11.0, 90.0),
    ("C2", 0.0, -3.0, 90.0),
    ("C3", 0.0, 1.7, 0.0),
    ("U1", 0.0, 4.3, 0.0),
    ("C4", 0.0, 6.7, 0.0),
    ("J1", -0.4, 18.0, 90.0),
    ("H1", -15.25, -15.25, 0.0),
    ("H2", 15.25, -15.25, 0.0),
    ("H3", 15.25, 15.25, 0.0),
    ("H4", -15.25, 15.25, 0.0),
];

/// How far a rebuild goes: placement only (rooms copied), channel 0 routed, or the whole board.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Place,
    Channel,
    Board,
}

pub fn build(dir: &Path, stage: Stage) -> Build {
    let start = &std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/drone/esc.start.circuit.layout")).unwrap();
    let mut b = Build::new(dir, "esc", start);
    b.start(None);
    let (a, z) = ROOM_SPAN;
    for (k, name) in ROOMS.iter().enumerate() {
        let points: Vec<[f64; 2]> = [[a, a], [z, a], [z, z], [a, z]].into_iter().map(|p| turn(p, k)).collect();
        b.run("board.room", json!({ "name": name, "points": points }), &[]);
    }
    let (x0, y0) = ORIGIN;
    for (path, x, y, r, bottom) in CHANNEL {
        let path = format!("/ch[0]/{path}");
        b.place(&path, x0 + x, y0 + y, *r);
        if *bottom {
            b.flip(&path);
        }
    }
    for (path, x, y, r) in MAIN {
        b.place(path, *x, *y, *r);
    }
    let all = [[-25.6, -23.6], [-23.6, -25.6], [23.6, -25.6], [25.6, -23.6], [25.6, 23.6], [23.6, 25.6], [-23.6, 25.6], [-25.6, 23.6]];
    b.pour("GND", "In1.Cu", &all, "relief");
    b.pour("VBAT", "In4.Cu", &all, "relief");
    b.pour("GND", "B.Cu", &all, "relief-tht");
    // The bottom sits on the heatsink: no tracks or vias on B.Cu under each aluminium
    // standoff's top (6.8 mm across, 0.3 mm clear; the mounting hole's ground ring bears on it),
    // and no bottom-side parts over the 38 mm pedestal (0.5 mm under the board on its gap pad),
    // 0.5 mm clear.
    for (_, x, y, _) in MAIN.iter().filter(|m| m.0.starts_with('H')) {
        let r = 3.7;
        let ring: Vec<[f64; 2]> = (0..12).map(|k| {
            let a = std::f64::consts::PI * k as f64 / 6.0;
            [x + r * a.cos() / (std::f64::consts::PI / 12.0).cos(), y + r * a.sin() / (std::f64::consts::PI / 12.0).cos()]
        }).collect();
        for applies in ["tracks", "vias"] {
            b.run("board.keepout", json!({ "points": ring, "applies": applies, "layer": "B.Cu" }), &[]);
        }
    }
    b.run("board.keepout", json!({ "points": [[-19.5, -19.5], [19.5, -19.5], [19.5, 19.5], [-19.5, 19.5]], "applies": "components", "layer": "B.Cu" }), &[]);
    // Each phase node is a solid top-copper pour joining its high side's sources, its low side's
    // drain tab and its motor pad (room-local millimetres).
    let phases: [(&str, [[f64; 2]; 4]); 3] = [
        ("PHA", [[1.2, 3.7], [7.9, 3.7], [7.9, 6.6], [1.2, 6.6]]),
        ("PHB", [[8.7, 3.95], [17.4, 3.95], [17.4, 6.6], [8.7, 6.6]]),
        ("PHC", [[14.8, 14.0], [21.6, 14.0], [21.6, 16.7], [14.8, 16.7]]),
    ];
    // Each low side's sources join its shunt's pad 1, which sits over them, through a solid pour.
    let sources: [(&str, [[f64; 2]; 4]); 3] = [
        ("SRCA", [[2.15, 8.75], [4.4, 8.75], [4.4, 12.0], [2.15, 12.0]]),
        ("SRCB", [[15.15, 8.75], [17.3, 8.75], [17.3, 12.0], [15.15, 12.0]]),
        ("SRCC", [[17.9, 8.45], [21.4, 8.45], [21.4, 11.75], [17.9, 11.75]]),
    ];
    for (net, pts) in phases.into_iter().chain(sources) {
        let pts: Vec<[f64; 2]> = pts.iter().map(|[x, y]| [x0 + x, y0 + y]).collect();
        b.pour(&format!("/ch[0]/{net}"), "F.Cu", &pts, "solid");
    }
    // No track crosses the motor current's copper: a track keepout on the top copper covers the
    // gap between each phase node's FETs and its motor pad and between each low side's sources
    // and its shunt, in every channel (board millimetres, channel 0; keepouts don't copy).
    let current: [[f64; 4]; 7] = [
        [-23.45, -20.85, -17.9, -19.15],
        [-23.3, -16.1, -21.45, -15.4],
        [-13.8, -20.85, -8.5, -19.15],
        [-10.25, -16.05, -8.4, -15.45],
        [-7.75, -10.45, -5.0, -9.75],
        [-7.75, -10.9, -6.7, -8.35],
        [-7.5, -15.15, -5.0, -14.55],
    ];
    for k in 0..4 {
        for [ax, ay, bx, by] in current {
            let ring: Vec<[f64; 2]> = [[ax, ay], [bx, ay], [bx, by], [ax, by]].into_iter().map(|p| turn(p, k)).collect();
            b.run("board.keepout", json!({ "points": ring, "applies": "tracks", "layer": "F.Cu" }), &[]);
        }
    }
    // Each gate's track from its FET's gate pin to its gate resistor (board millimetres): A's to
    // vias at the board edge and down to the resistors under the bridge, B's high side around
    // the drain tab into the bottom corridor, B's low side beside the stack hole with its drive
    // dropping through a via clear of the standoff, C's out along the channel's inner edge.
    let gate = |b: &mut Build, net: &str, layer: &str, points: &[[f64; 2]]| {
        b.run("board.route", json!({ "net": format!("net:/ch[0]/{net}"), "layer": layer, "width": 0.25, "points": points }), &[]);
    };
    let via = |b: &mut Build, net: &str, at: [f64; 2]| {
        b.run("board.place-via", json!({ "net": format!("net:/ch[0]/{net}"), "points": [at] }), &[]);
    };
    gate(&mut b, "GHA", "F.Cu", &[[-23.675, -21.2075], [-24.75, -21.2075]]);
    via(&mut b, "GHA", [-24.75, -21.2075]);
    gate(&mut b, "GHA", "B.Cu", &[[-24.75, -21.2075], [-24.75, -22.0725]]);
    gate(&mut b, "GLA", "F.Cu", &[[-23.675, -16.4075], [-24.75, -16.4075]]);
    via(&mut b, "GLA", [-24.75, -16.4075]);
    gate(&mut b, "GLA", "B.Cu", &[[-24.75, -16.4075], [-24.75, -17.2725]]);
    gate(&mut b, "GHB", "F.Cu", &[[-10.695, -21.2075], [-11.7, -21.2075], [-11.7, -22.45], [-12.4725, -22.45]]);
    gate(&mut b, "GLB", "F.Cu", &[[-10.695, -16.4075], [-12.15, -16.4075], [-12.15, -16.5225]]);
    gate(&mut b, "GLB_D", "F.Cu", &[[-12.15, -17.2775], [-12.1, -18.1]]);
    via(&mut b, "GLB_D", [-12.1, -18.1]);
    gate(&mut b, "GHC", "F.Cu", &[[-4.525, -9.3925], [-3.0, -9.3925], [-3.0, -8.7775]]);
    gate(&mut b, "GLC", "F.Cu", &[[-4.525, -14.1925], [-3.0, -14.1925], [-3.0, -11.5775]]);
    // The gate driver's escape (board millimetres). The logic inputs leave the side toward the
    // stack hole as a bus, run north beside it and turn north-west past the standoff toward the
    // MCU; enable and fault join the bus's east side. Phase C's drive leaves the side toward the
    // board's center line: its switch node runs north on the bottom to a via into the phase pour,
    // the gate drives run up the channel's inner edge to vias beside the gate resistors. The
    // bootstrap capacitor sits on the bottom beside the switch node, reached through the top; the
    // supply and drive capacitors sit on the top over the driver's corner.
    let track = |b: &mut Build, net: &str, layer: &str, width: f64, points: &[[f64; 2]]| {
        b.run("board.route", json!({ "net": format!("net:{net}"), "layer": layer, "width": width, "points": points }), &[]);
    };
    let via_at = |b: &mut Build, net: &str, at: [f64; 2]| {
        b.run("board.place-via", json!({ "net": format!("net:{net}"), "points": [at], "size": "0.2/0.45" }), &[]);
    };
    // The inputs to the MCU's bottom row run up the bus and leave it west to east, each
    // stepping west to a via between the shunt filters and the bulk capacitor; they cross to the
    // MCU on In2 and In3 and come up through vias under its pins.
    // (net, bus x, where it leaves the bus, inner layer, inner path, via under the pin)
    let inputs: [(&str, f64, f64, &str, &[[f64; 2]], [f64; 2]); 5] = [
        ("INHA", -9.15, -11.55, "In3.Cu", &[[-14.6, -11.2], [-15.2, -10.6], [-17.65, -10.6]], [-18.25, -10.0]),
        ("INLA", -8.65, -10.9, "In2.Cu", &[[-14.2, -11.0], [-17.75, -11.0]], [-17.75, -9.45]),
        ("INHB", -8.15, -10.3, "In2.Cu", &[[-14.35, -10.5], [-16.75, -10.5]], [-17.25, -10.0]),
        ("INLB", -7.65, -9.6, "In3.Cu", &[[-14.2, -9.45]], [-16.75, -9.45]),
        ("INHC", -7.15, -8.9, "In2.Cu", &[[-15.15, -8.9]], [-16.25, -10.0]),
    ];
    for (name, x, leave, layer, path, pin_via) in inputs {
        let net = format!("/ch[0]/{name}");
        let turn = [x - (leave + 15.5), leave];
        let bus_via = [turn[0] - 0.45, leave];
        track(&mut b, &net, "B.Cu", 0.2, &[[x, -20.5825], [x, -15.5], turn, bus_via]);
        via_at(&mut b, &net, bus_via);
        let points: Vec<[f64; 2]> = [bus_via].into_iter().chain(path.iter().copied()).chain([pin_via]).collect();
        track(&mut b, &net, layer, 0.2, &points);
        via_at(&mut b, &net, pin_via);
        // The low-side inputs' vias sit in their pads; the others step up into theirs.
        if pin_via[1] < -9.6 {
            track(&mut b, &net, "F.Cu", 0.2, &[pin_via, [pin_via[0], -9.3025]]);
        }
    }
    // The driver's west side, phases A and B: each switch-node return leaves between its
    // bootstrap pins and drops through a via beside bridge B's high side; A crosses under the
    // motor pad PB on In3 to PA, B runs on In2 to PB. Bootstrap B runs west on the bottom to its
    // capacitor, whose phase pad reaches PB through In2; GHB_D drops to the top beside it. Phase
    // A's bootstrap capacitor joins PA on the bottom.
    track(&mut b, "/ch[0]/PHA", "B.Cu", 0.3, &[[-9.8175, -22.25], [-11.05, -22.25]]);
    via_at(&mut b, "/ch[0]/PHA", [-11.05, -22.25]);
    track(&mut b, "/ch[0]/PHA", "In3.Cu", 0.3, &[[-11.05, -22.25], [-16.9, -22.25], [-18.0, -21.15]]);
    track(&mut b, "/ch[0]/PHA", "B.Cu", 0.3, &[[-17.17, -21.0775], [-18.2, -21.0775]]);
    track(&mut b, "/ch[0]/PHB", "B.Cu", 0.3, &[[-9.8175, -23.75], [-11.3, -23.8]]);
    via_at(&mut b, "/ch[0]/PHB", [-11.3, -23.8]);
    track(&mut b, "/ch[0]/PHB", "In2.Cu", 0.3, &[[-11.3, -23.8], [-11.6, -24.1], [-12.6, -24.1], [-14.85, -21.85], [-14.85, -21.5]]);
    track(&mut b, "/ch[0]/PHB", "B.Cu", 0.3, &[[-17.17, -22.8775], [-16.7, -23.5]]);
    via_at(&mut b, "/ch[0]/PHB", [-16.7, -23.5]);
    track(&mut b, "/ch[0]/PHB", "In2.Cu", 0.3, &[[-16.7, -23.5], [-16.0, -22.8], [-16.0, -21.6]]);
    track(&mut b, "/ch[0]/BSTB", "B.Cu", 0.2, &[[-9.8175, -22.75], [-10.35, -22.75], [-10.4, -22.8], [-16.2, -22.8], [-16.65, -22.35], [-17.17, -22.1225]]);
    track(&mut b, "/ch[0]/GHB_D", "B.Cu", 0.2, &[[-9.8175, -23.25], [-11.85, -23.25], [-11.9, -23.3]]);
    via_at(&mut b, "/ch[0]/GHB_D", [-11.9, -23.3]);
    track(&mut b, "/ch[0]/GHB_D", "F.Cu", 0.2, &[[-11.9, -23.3], [-12.75, -23.3], [-13.2275, -22.8225], [-13.2275, -22.45]]);
    // INLC, enable and fault run north past the motor pad PC, turn west above it and drop
    // through vias east of the MCU's side row (board millimetres).
    let side = [("INLC", -6.65, -7.95, -8.15), ("DRV_EN", -5.95, -7.35, -7.65), ("FAULT", -5.25, -6.75, -7.15)];
    for (k, (net, x, turn_y, pad_y)) in side.into_iter().enumerate() {
        let net = format!("/ch[0]/{net}");
        let start: &[[f64; 2]] = match k {
            0 => &[[x, -20.5825]],
            1 => &[[-5.9825, -21.25], [x, -21.25]],
            _ => &[[-5.9825, -21.75], [x, -21.75]],
        };
        let via = [-13.3, turn_y];
        let points: Vec<[f64; 2]> = start.iter().copied().chain([[x, turn_y], via]).collect();
        track(&mut b, &net, "B.Cu", 0.2, &points);
        via_at(&mut b, &net, via);
        let jog = pad_y - turn_y;
        track(&mut b, &net, "F.Cu", 0.2, &[via, [-14.0, turn_y], [-14.0 + jog, pad_y], [-15.0975, pad_y]]);
    }
    track(&mut b, "/ch[0]/PHC", "B.Cu", 0.3, &[[-5.9825, -22.25], [-4.08, -22.25], [-4.08, -11.7]]);
    via_at(&mut b, "/ch[0]/PHC", [-4.08, -11.7]);
    track(&mut b, "/ch[0]/PHC", "F.Cu", 0.3, &[[-4.08, -11.7], [-4.525, -11.25]]);
    track(&mut b, "/ch[0]/GHC_D", "B.Cu", 0.2, &[[-5.9825, -22.75], [-2.78, -22.75], [-2.78, -8.0225]]);
    via_at(&mut b, "/ch[0]/GHC_D", [-2.78, -8.0225]);
    track(&mut b, "/ch[0]/GHC_D", "F.Cu", 0.2, &[[-2.78, -8.0225], [-3.0, -8.0225]]);
    track(&mut b, "/ch[0]/GLC_D", "B.Cu", 0.2, &[[-5.9825, -23.25], [-2.43, -23.25], [-2.43, -15.2], [-2.1, -14.87], [-2.1, -10.8225]]);
    via_at(&mut b, "/ch[0]/GLC_D", [-2.1, -10.8225]);
    track(&mut b, "/ch[0]/GLC_D", "F.Cu", 0.2, &[[-2.1, -10.8225], [-3.0, -10.8225]]);
    track(&mut b, "/ch[0]/BSTC", "B.Cu", 0.2, &[[-5.9825, -23.75], [-5.15, -23.75]]);
    via_at(&mut b, "/ch[0]/BSTC", [-5.15, -23.75]);
    track(&mut b, "/ch[0]/BSTC", "F.Cu", 0.2, &[[-5.15, -23.75], [-3.32, -21.92], [-3.32, -21.55]]);
    via_at(&mut b, "/ch[0]/BSTC", [-3.32, -21.55]);
    track(&mut b, "/ch[0]/BSTC", "B.Cu", 0.2, &[[-3.32, -21.55], [-3.32, -21.0]]);
    track(&mut b, "VBAT", "B.Cu", 0.2, &[[-6.65, -24.4175], [-6.65, -25.3]]);
    via_at(&mut b, "VBAT", [-6.65, -25.3]);
    track(&mut b, "VBAT", "F.Cu", 0.3, &[[-6.65, -25.3], [-6.15, -24.9]]);
    track(&mut b, "/ch[0]/VDRV", "B.Cu", 0.2, &[[-8.15, -24.4175], [-8.15, -25.3]]);
    via_at(&mut b, "/ch[0]/VDRV", [-8.15, -25.3]);
    track(&mut b, "/ch[0]/VDRV", "F.Cu", 0.3, &[[-8.15, -25.3], [-7.4, -24.55], [-7.4, -22.6], [-7.03, -22.6]]);
    // The top's ground pads drop to the In1 plane through laser vias in the pads (top to In1),
    // which leave In2, In3 and the bottom free: twelve in each shunt's ground pad for its 20 A,
    // four in the bulk capacitor's, one in each decoupling and filter capacitor's and the MCU's
    // ground pin.
    let micro = |b: &mut Build, at: [f64; 2]| {
        b.run("board.place-via", json!({ "net": "net:GND", "points": [at], "size": "0.1/0.3", "span": "micro" }), &[]);
    };
    for (cx, cy) in [(-19.525, -14.5), (-11.775, -14.5), (-4.075, -16.1)] {
        for dx in [-0.4, 0.0, 0.4] {
            for dy in [-0.6, -0.2, 0.2, 0.6] {
                micro(&mut b, [cx + dx, cy + dy]);
            }
        }
    }
    for at in [
        [-12.3, -8.65], [-11.5, -8.65], [-12.3, -9.1], [-11.5, -9.1],
        [-5.53, -22.6], [-4.65, -24.9], [-17.7, -11.1225], [-15.3, -11.1225], [-11.3, -5.6225],
        [-20.3, -11.1225], [-21.3, -9.5775], [-21.3, -4.3225], [-15.75, -9.5],
    ] {
        micro(&mut b, at);
    }
    // The gate driver's ground pins join its exposed pad, which drops to In1 through three vias
    // between bridge B's high side and the drive capacitor.
    for x in [-7.65, -7.15] {
        track(&mut b, "GND", "B.Cu", 0.2, &[[x, -24.4175], [x, -23.6]]);
    }
    for y in [-23.35, -22.55, -21.85] {
        via_at(&mut b, "GND", [-8.25, y]);
    }
    // The battery feed to the high sides' drains: a top pour along the board edge joins bridges
    // A and B's drain pins, with vias to In4 along it and under bridge A's drain pad (bridge B's
    // has the gate driver under it); bridge C's drain pins face a pour toward the board's
    // center with a grid of vias.
    b.pour("VBAT", "F.Cu", &[[-23.9, -25.6], [-8.6, -25.6], [-8.6, -24.0], [-23.9, -24.0]], "solid");
    b.pour("VBAT", "F.Cu", &[[-7.0, -6.2], [-3.6, -6.2], [-3.6, -2.4], [-7.0, -2.4]], "solid");
    let feed = (0..10)
        .map(|k| [-21.1 + 0.8 * k as f64, -24.45])
        .chain((0..4).map(|k| [-12.6 + 0.8 * k as f64, -25.3]))
        .chain([-23.6, -22.9, -22.2].into_iter().flat_map(|y| [[-23.2, y], [-22.2, y]]))
        .chain([-5.0, -4.1, -3.2].into_iter().flat_map(|y| [[-6.5, y], [-5.6, y], [-4.7, y]]));
    for at in feed {
        via_at(&mut b, "VBAT", at);
    }
    // The channel's plane pads take their vias first, so VBAT and GND reach In4 and In1 straight
    // down; then the gate drive and logic, short loops first; then the phase and shunt copper.
    let parts: Vec<String> = b
        .board()
        .footprints
        .values()
        .filter(|f| f.component.as_deref().is_some_and(|c| c.starts_with("/ch[0]/")))
        .filter(|f| f.package.pads.iter().any(|p| p.kind == runmat_circuit::library::PadKind::Smd))
        .map(|f| format!("component:{}", f.reference))
        .collect();
    let nets = |names: &[&str]| -> Vec<String> { names.iter().map(|n| format!("net:/ch[0]/{n}")).collect() };
    let power = nets(&["PHA", "PHB", "PHC", "SRCA", "SRCB", "SRCC"]);
    let signals = nets(&["GHA", "GLA", "GHB", "GLB", "GHC", "GLC", "GHA_D", "GLA_D", "GHB_D", "GLB_D", "GHC_D", "GLC_D", "BSTA", "BSTB", "BSTC", "VDRV", "INHA", "INLA", "INHB", "INLB", "INHC", "INLC", "DRV_EN", "FAULT", "ISENSE_A", "ISENSE_B", "ISENSE_C", "BOOT0", "NRST"]);
    if stage == Stage::Place {
        b.run("board.copy-room-formats", json!({ "source": ["room:ch[0]"], "rotate": true }), &[]);
        return b;
    }
    // Each MCU's motor-signal pin drops through a via in its pad. Channel 0's router routes
    // around its via, and the copies keep that routing: the via sits at the same spot beside
    // each MCU.
    for k in 0..4 {
        let at = turn([-18.75, -9.45], k);
        b.run("board.place-via", json!({ "net": format!("net:M{}", k + 1), "points": [at], "size": "0.2/0.45" }), &[]);
    }
    b.run("board.fanout", json!({ "component": parts }), &[]);
    b.run("board.autoroute", json!({ "nets": signals }), &[]);
    b.run("board.autoroute", json!({ "nets": power }), &[]);
    if stage == Stage::Channel {
        return b;
    }
    b.run("board.copy-room-formats", json!({ "source": ["room:ch[0]"], "rotate": true }), &[]);
    // The top's free copper is ground, below every other pour, stitched to In1 and the bottom;
    // In3's free copper carries 3V3 from the regulator to the four MCUs.
    b.run("board.pour", json!({ "net": "net:GND", "layer": "F.Cu", "points": all, "connect": "solid", "clearance": 0.3, "priority": -1 }), &[]);
    b.run("board.pour", json!({ "net": "net:+3V3", "layer": "In3.Cu", "points": all, "connect": "relief", "priority": -1 }), &[]);
    // The center column: the bulk capacitor, the regulator and the harness's battery pin reach
    // In4 through vias; the regulator's and the capacitor's ground pads reach In1 through laser
    // vias.
    for at in [[-1.0, -14.2], [-0.2, -14.2], [-1.3, 5.25], [-1.3, 3.35], [1.3, 14.5]] {
        b.run("board.place-via", json!({ "net": "net:VBAT", "points": [at], "size": "0.2/0.45" }), &[]);
    }
    for at in [[-1.0, -8.1], [0.0, -8.1], [1.0, -8.1], [-1.3, 4.3]] {
        b.run("board.place-via", json!({ "net": "net:GND", "points": [at], "size": "0.1/0.3", "span": "micro" }), &[]);
    }
    let board = b.board();
    let top_ground = board
        .zones
        .values()
        .find(|z| z.layers == [0] && z.net.is_some_and(|n| board.net_name(n) == "GND"))
        .map(|z| z.id.0)
        .expect("the top ground pour");
    b.run("board.stitch", json!({ "mode": "stitch", "target": top_ground.to_string(), "pitch": 1.5, "via": "0.2/0.45" }), &[]);
    // Ground and battery reach their planes through vias; the router joins what is left.
    b.finish(&[]);
    b
}
