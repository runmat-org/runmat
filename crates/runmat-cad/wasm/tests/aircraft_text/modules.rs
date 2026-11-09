//! The airliner's folders: which folder holds each design, and the `from` each occurrence needs
//! to reach a design in another folder. The generators write occurrences in the same-folder form
//! (`use part frame as frame`); [`into_folders`] places each file and adds `from "../cabin"` or
//! `from hardware` where the design lives elsewhere.

/// The folder of the design `stem` (`""` is the project root).
fn folder(stem: &str) -> &'static str {
    match stem {
        "airliner" | "airliner_skeleton" => "",
        "rivet" => "+hardware",
        "fuselage"
        | "fuselage_section"
        | "rivet_ring"
        | "nose"
        | "tail_cone"
        | "fairing"
        | "frame"
        | "stringer"
        | "skin_panel"
        | "cabin_window"
        | "aft_pressure_bulkhead"
        | "cockpit_window_front"
        | "cockpit_window_side"
        | "cockpit_window_aft" => "fuselage",
        "wing" | "rib" => "wing",
        s if s.starts_with("wing_") => "wing",
        "empennage" | "fin" | "rudder" | "stabilizer" | "elevator" => "empennage",
        "main_gear" | "nose_gear" | "gear_strut" | "axle" | "tyre" | "hub" => "landing_gear",
        "nacelle" | "cowl" | "fan" | "spinner" | "core_nozzle" | "plug" | "pylon" => "engine",
        "cabin" | "galley" | "lavatory" | "overhead_bin" | "sidewall_lining" | "ceiling_panel"
        | "floor_beam" | "floor_panel" | "seat_track" | "passenger_door" | "passenger_door_aft"
        | "overwing_exit" => "cabin",
        s if s.starts_with("seat_") => "cabin",
        "cockpit" | "cockpit_bulkhead" | "cockpit_floor" | "control_column" | "display_unit"
        | "glareshield" | "instrument_panel" | "overhead_panel" | "pedestal"
        | "pilot_seat_cushion" | "nose_lining" => "cockpit",
        "hold" | "centre_wing_box" | "gear_bay" => "cargo",
        s if s.starts_with("cargo_") || s.starts_with("hold_liner") => "cargo",
        other => panic!("no folder for the airliner design `{other}`"),
    }
}

/// How a file in `here` names the folder `there`: none for its own, the package name for a
/// `+pkg` folder at the root, else a relative path.
fn from(here: &str, there: &str) -> Option<String> {
    if here == there {
        return None;
    }
    if let Some(package) = there.strip_prefix('+') {
        return Some(package.to_string());
    }
    Some(match (here, there) {
        ("", t) => format!("\"./{t}\""),
        (_, "") => "\"..\"".to_string(),
        (_, t) => format!("\"../{t}\""),
    })
}

/// The flat files, each in its folder, with `from` added to every occurrence of a design in
/// another folder (its head `use <kind> <name> as`, and a mirror's opposite-hand source).
pub fn into_folders(flat: Vec<(String, String)>) -> Vec<(String, String)> {
    flat.into_iter()
        .map(|(name, text)| {
            let stem = name.split('.').next().unwrap();
            let here = folder(stem);
            let mut out = String::with_capacity(text.len());
            for line in text.split_inclusive('\n') {
                out += &with_from(line, here);
            }
            let path = if here.is_empty() {
                name
            } else {
                format!("{here}/{name}")
            };
            (path, out)
        })
        .collect()
}

fn with_from(line: &str, here: &str) -> String {
    let t = line.trim_start();
    for kind in ["part", "assembly"] {
        // `use <kind> <name> as <alias>`
        if let Some(rest) = t.strip_prefix(&format!("use {kind} ")) {
            let name = rest.split(' ').next().unwrap();
            if let Some(f) = from(here, folder(name)) {
                return line.replacen(
                    &format!("use {kind} {name} as "),
                    &format!("use {kind} {name} from {f} as "),
                    1,
                );
            }
            return line.to_string();
        }
        // `definition = { kind = <kind>, name = <name> }` in a mirror's hand.
        let key = format!("definition = {{ kind = {kind}, name = ");
        if let Some(at) = line.find(&key) {
            let name = line[at + key.len()..].split(' ').next().unwrap();
            // A mirror's opposite hand is the seed's own design, in the seed's folder.
            assert!(
                from(here, folder(name)).is_none(),
                "a mirror in `{here}` names `{name}` in another folder"
            );
        }
    }
    line.to_string()
}
