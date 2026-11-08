//! The flight controller: 36 x 36 mm, four layers (signal, ground, power, signal), the MCU in
//! the middle, the buck and LDO in the lower-left quadrant, the IMU beside the MCU on the
//! board's center line, the barometer away from the regulators, and the connectors at the edges
//! (the GPS port on the right edge under the video transmitter's).

use super::build::Build;
use std::path::Path;

/// (schematic path, x, y, rotation), board millimetres from the board's center.
const PLACE: &[(&str, f64, f64, f64)] = &[
    ("U1", 1.0, 2.0, 0.0),
    ("C3", 1.0, 7.9, 0.0),
    ("C4", 6.9, 2.0, 90.0),
    ("C5", 1.0, -3.9, 0.0),
    ("C6", -4.9, 2.0, 90.0),
    ("C7", 2.6, 7.9, 0.0),
    ("C8", -4.9, 4.0, 90.0),
    ("C9", -5.5, 0.0, 90.0),
    ("C10", 7.2, -1.6, 90.0),
    ("C11", -4.9, -1.8, 90.0),
    ("R13", -6.2, -1.8, 90.0),
    ("Y1", -4.0, 10.4, 0.0),
    ("C1", -7.8, 11.2, 0.0),
    ("C2", -7.8, 9.6, 0.0),
    ("R1", 5.0, 7.9, 0.0),
    ("J1", 4.0, -13.5, 0.0),
    ("R2", -1.8, -8.8, 90.0),
    ("R3", 10.0, -9.0, 90.0),
    ("D3", 13.6, -8.6, 90.0),
    ("D1", 7.0, 10.8, 0.0),
    ("R4", 7.0, 12.6, 0.0),
    ("D2", 10.6, 10.8, 0.0),
    ("R5", 10.6, 12.6, 0.0),
    ("J2", -14.8, 6.5, 270.0),
    ("J3", 0.0, 14.8, 180.0),
    ("J4", 14.8, 8.0, 90.0),
    // GPS and compass on the right edge, under the video transmitter.
    ("J5", 14.8, -1.0, 90.0),
    ("H1", -15.25, -15.25, 0.0),
    ("H2", 15.25, -15.25, 0.0),
    ("H3", 15.25, 15.25, 0.0),
    ("H4", -15.25, 15.25, 0.0),
    // 5 V buck and 3.3 V LDO.
    ("/pwr/U2", -8.0, -8.4, 0.0),
    ("/pwr/L1", -8.3, -13.4, 0.0),
    ("/pwr/C12", -11.6, -7.8, 90.0),
    ("/pwr/C13", -11.6, -2.8, 90.0),
    ("/pwr/C14", -5.4, -10.2, 90.0),
    ("/pwr/C15", -4.6, -13.4, 90.0),
    ("/pwr/C16", -2.4, -13.4, 90.0),
    ("/pwr/R6", -5.2, -5.2, 90.0),
    ("/pwr/R7", -4.0, -5.2, 90.0),
    ("/pwr/C17", -6.4, -5.2, 90.0),
    ("/pwr/R8", -12.6, -11.2, 90.0),
    ("/pwr/U3", -8.0, 1.0, 0.0),
    ("/pwr/C18", -8.0, 3.4, 0.0),
    ("/pwr/C19", -8.0, -1.4, 0.0),
    ("/pwr/R9", -11.4, 1.4, 90.0),
    ("/pwr/R10", -11.4, 3.6, 90.0),
    ("/pwr/C20", -11.4, 5.6, 90.0),
    // IMU on the center line beside the MCU, the barometer by the right edge.
    ("/sens/U4", 10.0, 4.6, 0.0),
    ("/sens/C21", 10.0, 7.0, 0.0),
    ("/sens/C22", 10.0, 2.2, 0.0),
    ("/sens/U5", 10.4, -2.4, 0.0),
    ("/sens/C23", 8.2, -4.3, 90.0),
    ("/sens/R11", 8.6, -5.6, 0.0),
    ("/sens/R12", 10.4, -5.6, 0.0),
];

pub fn build(dir: &Path) -> Build {
    let start = &std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/drone/flight_controller.start.circuit.layout")).unwrap();
    let mut b = Build::new(dir, "flight_controller", start);
    // The checked-in lock keeps each component's identity and schematic place.
    if let Ok(lock) = std::fs::read_to_string(dir.join("flight_controller.circuit.lock")) {
        b.files.insert("/p/flight_controller.circuit.lock".into(), lock);
    }
    b.start(Some(4));
    for (path, x, y, r) in PLACE {
        b.place(path, *x, *y, *r);
    }
    let all = [[-17.6, -16.6], [-16.6, -17.6], [16.6, -17.6], [17.6, -16.6], [17.6, 16.6], [16.6, 17.6], [-16.6, 17.6], [-17.6, 16.6]];
    b.pour("GND", "In1.Cu", &all, "relief");
    b.pour("+3V3", "In2.Cu", &all, "relief");
    b.pour("GND", "B.Cu", &all, "relief-tht");
    // The 5 V island in the power plane under the regulators.
    b.run("board.pour", serde_json::json!({ "net": "net:+5V", "layer": "In2.Cu", "points": [[-17.6, -16.6], [-16.6, -17.6], [0.6, -17.6], [0.6, -5.0], [-17.6, -5.0]], "priority": 1 }), &[]);
    usb_breakout(&mut b);
    // VDDA (pin 9) drops straight to the 3.3 V plane through its own via beside the pin.
    b.run("board.route", serde_json::json!({ "net": "net:+3V3", "points": [[-3.125, 0.75], [-4.6, 0.75]], "width": 0.25, "layer": "Top", "mode": "ignore", "corner": "any" }), &[]);
    b.run("board.place-via", serde_json::json!({ "points": [[-4.6, 0.75]], "net": "net:+3V3", "size": "0.2/0.45" }), &[]);
    // The USB pair first, while the way to the MCU's USB pins is open.
    b.run("board.autoroute", serde_json::json!({ "nets": ["net:USB_DP", "net:USB_DN"] }), &[]);
    // The ESC telemetry line runs from the top edge to the MCU's lower right, past the GPS port,
    // and the I2C clock from the MCU's top side to the barometer and the port: each gets its way
    // before the rest fills the right side.
    b.run("board.autoroute", serde_json::json!({ "nets": ["net:TLM"] }), &[]);
    b.run("board.autoroute", serde_json::json!({ "nets": ["net:I2C_SCL"] }), &[]);
    b.finish(&[]);
    b
}

/// The USB-C receptacle carries each data line twice (one pair per plug orientation), the two
/// pads of a line either side of the other line's pad. Each line's pads join on the bottom layer
/// through a via above its pads, so neither crosses the other's pad.
fn usb_breakout(b: &mut Build) {
    use serde_json::json;
    let r = |b: &mut Build, net: &str, points: serde_json::Value, layer: &str| {
        b.run("board.route", json!({ "net": format!("net:{net}"), "points": points, "width": 0.12, "layer": layer, "mode": "ignore", "corner": "any" }), &[]);
    };
    // Pads 4 (DP), 5 (DN), 6 (DP), 7 (DN) sit at x = 2.75, 3.25, 3.75, 4.25 on y = -17.15.
    r(b, "USB_DP", json!([[2.75, -17.15], [2.75, -15.6]]), "Top");
    r(b, "USB_DP", json!([[3.75, -17.15], [3.75, -14.6]]), "Top");
    b.run("board.place-via", json!({ "points": [[2.75, -15.6]], "net": "net:USB_DP", "size": "0.2/0.45" }), &[]);
    b.run("board.place-via", json!({ "points": [[3.75, -14.6]], "net": "net:USB_DP", "size": "0.2/0.45" }), &[]);
    r(b, "USB_DP", json!([[2.75, -15.6], [2.75, -14.6], [3.75, -14.6]]), "Bottom");
    r(b, "USB_DN", json!([[3.25, -17.15], [3.25, -13.6]]), "Top");
    r(b, "USB_DN", json!([[4.25, -17.15], [4.25, -13.6], [3.25, -13.6]]), "Top");
}
