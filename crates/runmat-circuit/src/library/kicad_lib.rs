//! KiCad libraries opened in place: a `.kicad_sym` symbol library read as catalog parts whose
//! packages are the `.kicad_mod` files their `Footprint` property names, so the library panel
//! lists, previews and places them like any other part. Nothing is copied: a placed part's
//! declaration names the footprint file where it lies.
//!
//! A symbol's `Footprint` property `Lib:Name` resolves to `<Lib>.pretty/<Name>.kicad_mod`
//! beside the symbol library (the layout of KiCad's own library folders). Every unit's pins
//! become one pin list; pins that share a name (`GND` on three pads) become one pin on those
//! pads, and an unnamed pin (`~`) takes its number.
//!
//! A symbol's simulation fields become the part's SPICE model ([`sim_model`]): KiCad 7 and later
//! write `Sim.Device`, `Sim.Name`, `Sim.Library` and `Sim.Pins`; KiCad 6 and earlier wrote
//! `Spice_Primitive`, `Spice_Model`, `Spice_Lib_File` and `Spice_Node_Sequence`.

use super::catalog::{PartRecord, PartPin, PartSpice};
use crate::kicad::sch::parse_lib_symbol;
use crate::kicad::sexpr;
use crate::library::PinType;

fn property<'a>(node: &'a sexpr::SExpr, key: &str) -> Option<&'a str> {
    node.children("property")
        .find(|p| p.arg(0) == Some(key))
        .and_then(|p| p.arg(1))
        .filter(|v| !v.is_empty() && *v != "~")
}

/// The category a KiCad designator prefix stands for.
fn category(prefix: &str) -> &'static str {
    match prefix {
        "R" => "resistor",
        "C" => "capacitor",
        "L" | "FB" => "inductor",
        "D" => "diode",
        "Q" => "transistor",
        "J" | "P" => "connector",
        "Y" => "crystal",
        "F" => "fuse",
        _ => "",
    }
}

/// The parts of a `.kicad_sym` library: each symbol with a footprint, as a catalog part whose
/// package is that footprint's `.kicad_mod` path relative to the library's folder. Symbols
/// without a footprint (power symbols, generic graphics) are left out; `skipped` names them.
pub fn read(text: &str) -> Result<(Vec<PartRecord>, Vec<String>), String> {
    let root = sexpr::parse(text).map_err(|e| e.to_string())?;
    if root.head() != Some("kicad_symbol_lib") {
        return Err("not a KiCad symbol library".into());
    }
    let mut parts = Vec::new();
    let mut skipped = Vec::new();
    for node in root.children("symbol") {
        let symbol = parse_lib_symbol(node);
        let footprint = property(node, "Footprint").and_then(|f| f.split_once(':'));
        let Some((lib, name)) = footprint.filter(|_| !symbol.power) else {
            skipped.push(symbol.name.clone());
            continue;
        };
        let mut pins: Vec<PartPin> = Vec::new();
        for pin in symbol.units.values().flatten() {
            let pin_name = pin_name_of(&pin.name, &pin.number);
            let kind = if pin.kind == PinType::Unspecified {
                PinType::Passive
            } else {
                pin.kind
            };
            match pins.iter_mut().find(|p| p.name == pin_name) {
                Some(p) if !p.pads.contains(&pin.number) => p.pads.push(pin.number.clone()),
                Some(_) => {}
                None => pins.push(PartPin {
                    name: pin_name,
                    pads: vec![pin.number.clone()],
                    kind: kind.name().into(),
                }),
            }
        }
        if pins.is_empty() {
            skipped.push(symbol.name.clone());
            continue;
        }
        let spice = sim_model(
            node,
            &symbol
                .units
                .values()
                .flatten()
                .map(|p| (p.number.clone(), pin_name_of(&p.name, &p.number)))
                .collect::<Vec<_>>(),
        );
        parts.push(PartRecord {
            mpn: property(node, "Value").unwrap_or(&symbol.name).to_string(),
            manufacturer: String::new(),
            description: property(node, "Description")
                .or_else(|| property(node, "ki_description"))
                .unwrap_or_default()
                .to_string(),
            category: category(&symbol.prefix).into(),
            prefix: symbol.prefix.clone(),
            package: format!("{lib}.pretty/{name}.kicad_mod"),
            pins,
            params: Default::default(),
            datasheet: property(node, "Datasheet")
                .filter(|d| d.starts_with("http"))
                .map(str::to_owned),
            spice,
            ..Default::default()
        });
    }
    Ok((parts, skipped))
}

fn pin_name_of(name: &str, number: &str) -> String {
    if name.is_empty() || name == "~" {
        number.to_string()
    } else {
        name.to_string()
    }
}

/// A symbol's SPICE model from its simulation fields, with the model's nodes as the part's pin
/// names in the model's order; `pins` maps each symbol pin number to its pin name.
///
/// - The device: `SUBCKT` (or the legacy primitive `X`) is a subcircuit; `D`, `NPN`/`PNP`,
///   `NMOS`/`PMOS` and `NJFET`/`PJFET` are the `D`, `Q`, `M` and `J` primitives on a `.model`
///   card. Passives and sources (`R`, `C`, `L`, `V`, `I`) simulate from their value and carry no
///   model, and so does a symbol without a model name.
/// - The node order: `Sim.Pins` pairs (`1=VIN 2=GND`, symbol pin = model pin) in model-pin
///   order when the model pins are numbers, else as written; the legacy `Spice_Node_Sequence`
///   lists symbol pin numbers in node order; with neither, the symbol's pins by number.
pub fn sim_model(node: &sexpr::SExpr, pins: &[(String, String)]) -> Option<PartSpice> {
    let device = property(node, "Sim.Device")
        .or_else(|| property(node, "Spice_Primitive"))
        .unwrap_or("SUBCKT")
        .to_ascii_uppercase();
    let letter = match device.as_str() {
        "SUBCKT" | "X" => None,
        "D" => Some("D"),
        "NPN" | "PNP" | "Q" => Some("Q"),
        "NMOS" | "PMOS" | "M" => Some("M"),
        "NJFET" | "PJFET" | "J" => Some("J"),
        _ => return None,
    };
    let model = property(node, "Sim.Name")
        .or_else(|| property(node, "Spice_Model"))?
        .to_string();
    let library = property(node, "Sim.Library")
        .or_else(|| property(node, "Spice_Lib_File"))
        .map(str::to_owned);
    let name_of = |number: &str| {
        pins.iter()
            .find(|(n, _)| n == number)
            .map(|(_, name)| name.clone())
    };
    let numbers: Vec<String> = if let Some(map) = property(node, "Sim.Pins") {
        let mut pairs: Vec<(&str, &str)> = map
            .split_whitespace()
            .filter_map(|pair| pair.split_once('='))
            .collect();
        if pairs.iter().all(|(_, m)| m.parse::<u32>().is_ok()) {
            pairs.sort_by_key(|(_, m)| m.parse::<u32>().unwrap_or(0));
        }
        pairs.into_iter().map(|(s, _)| s.to_string()).collect()
    } else if let Some(seq) = property(node, "Spice_Node_Sequence") {
        seq.split(|c: char| c.is_whitespace() || c == ',')
            .filter(|t| !t.is_empty())
            .map(str::to_owned)
            .collect()
    } else {
        let mut all: Vec<String> = pins.iter().map(|(n, _)| n.clone()).collect();
        all.sort_by(|a, b| match (a.parse::<u32>(), b.parse::<u32>()) {
            (Ok(x), Ok(y)) => x.cmp(&y),
            _ => a.cmp(b),
        });
        all.dedup();
        all
    };
    let nodes = numbers
        .iter()
        .map(|n| name_of(n).unwrap_or_else(|| n.clone()))
        .collect();
    Some(PartSpice {
        model,
        library,
        nodes,
        device: letter.map(str::to_owned),
    })
}

/// `part` with its package rebased from the symbol library's folder `lib_dir` to `base`, the
/// folder of the circuit that will declare it (both workspace paths, `/`-separated).
pub fn rebase(part: &PartRecord, lib_dir: &str, base: &str) -> PartRecord {
    let from: Vec<&str> = base.split('/').filter(|s| !s.is_empty()).collect();
    let rel = |path: &str| -> String {
        let target: Vec<&str> = lib_dir
            .split('/')
            .filter(|s| !s.is_empty())
            .chain(path.split('/'))
            .collect();
        let common = target.iter().zip(&from).take_while(|(a, b)| a == b).count();
        let mut out: Vec<&str> = std::iter::repeat_n("..", from.len() - common).collect();
        out.extend(&target[common..]);
        out.join("/")
    };
    // A model library beside the symbol library moves with it; an absolute path or one through
    // a KiCad path variable (`${KICAD7_...}`) stays as written.
    let spice = part.spice.as_ref().map(|s| PartSpice {
        library: s.library.as_ref().map(|l| {
            if l.starts_with('/') || l.starts_with("${") {
                l.clone()
            } else {
                rel(l)
            }
        }),
        ..s.clone()
    });
    PartRecord {
        package: rel(&part.package),
        spice,
        ..part.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LIB: &str = r##"(kicad_symbol_lib (version 20231120) (generator "test")
  (symbol "LDO_3V3" (in_bom yes) (on_board yes)
    (property "Reference" "U" (at 0 0 0))
    (property "Value" "LDO_3V3" (at 0 0 0))
    (property "Footprint" "Package_TO_SOT_SMD:SOT-23-5" (at 0 0 0))
    (property "Datasheet" "https://example.com/ldo.pdf" (at 0 0 0))
    (property "Description" "3.3 V 300 mA LDO" (at 0 0 0))
    (symbol "LDO_3V3_0_1"
      (pin power_in line (at -7.62 2.54 0) (length 2.54) (name "VIN" (effects (font (size 1.27 1.27)))) (number "1" (effects (font (size 1.27 1.27)))))
      (pin power_in line (at 0 -7.62 90) (length 2.54) (name "GND" (effects (font (size 1.27 1.27)))) (number "2" (effects (font (size 1.27 1.27)))))
      (pin input line (at -7.62 0 0) (length 2.54) (name "EN" (effects (font (size 1.27 1.27)))) (number "3" (effects (font (size 1.27 1.27)))))
      (pin no_connect line (at 7.62 0 180) (length 2.54) (name "NC" (effects (font (size 1.27 1.27)))) (number "4" (effects (font (size 1.27 1.27)))))
      (pin power_out line (at 7.62 2.54 180) (length 2.54) (name "VOUT" (effects (font (size 1.27 1.27)))) (number "5" (effects (font (size 1.27 1.27)))))
      (pin power_in line (at 0 -7.62 90) (length 2.54) (name "GND" (effects (font (size 1.27 1.27)))) (number "6" (effects (font (size 1.27 1.27)))))
    )
  )
  (symbol "GND" (power) (in_bom yes) (on_board yes)
    (property "Reference" "#PWR" (at 0 0 0))
    (property "Value" "GND" (at 0 0 0))
    (symbol "GND_0_1" (pin power_in line (at 0 0 270) (length 0) hide (name "GND" (effects (font (size 1.27 1.27)))) (number "1" (effects (font (size 1.27 1.27))))))
  )
)"##;

    #[test]
    fn a_symbol_library_reads_as_parts_on_their_footprint_files() {
        let (parts, skipped) = read(LIB).unwrap();
        assert_eq!(skipped, ["GND"]);
        let ldo = &parts[0];
        assert_eq!((ldo.mpn.as_str(), ldo.prefix.as_str()), ("LDO_3V3", "U"));
        assert_eq!(ldo.package, "Package_TO_SOT_SMD.pretty/SOT-23-5.kicad_mod");
        assert_eq!(
            ldo.datasheet.as_deref(),
            Some("https://example.com/ldo.pdf")
        );
        let gnd = ldo.pins.iter().find(|p| p.name == "GND").unwrap();
        assert_eq!(
            (gnd.pads.clone(), gnd.kind.as_str()),
            (vec!["2".to_string(), "6".to_string()], "power_in")
        );
        assert_eq!(ldo.pins.len(), 5);
    }

    const SIM: &str = r##"(kicad_symbol_lib (version 20231120) (generator "test")
  (symbol "BUCK" (in_bom yes) (on_board yes)
    (property "Reference" "U" (at 0 0 0))
    (property "Value" "BUCK" (at 0 0 0))
    (property "Footprint" "Package_TO_SOT_SMD:SOT-23-6" (at 0 0 0))
    (property "Sim.Device" "SUBCKT" (at 0 0 0))
    (property "Sim.Name" "BUCK36V_AVG" (at 0 0 0))
    (property "Sim.Library" "models/buck.lib" (at 0 0 0))
    (property "Sim.Pins" "1=3 2=6 3=5 4=2 5=1 6=4" (at 0 0 0))
    (symbol "BUCK_0_1"
      (pin passive line (at 0 0 0) (length 2.54) (name "BST") (number "1"))
      (pin power_in line (at 0 0 0) (length 2.54) (name "GND") (number "2"))
      (pin input line (at 0 0 0) (length 2.54) (name "FB") (number "3"))
      (pin input line (at 0 0 0) (length 2.54) (name "EN") (number "4"))
      (pin power_in line (at 0 0 0) (length 2.54) (name "VIN") (number "5"))
      (pin power_out line (at 0 0 0) (length 2.54) (name "SW") (number "6"))
    )
  )
  (symbol "BAT54" (in_bom yes) (on_board yes)
    (property "Reference" "D" (at 0 0 0))
    (property "Value" "BAT54" (at 0 0 0))
    (property "Footprint" "Package_TO_SOT_SMD:SOT-23" (at 0 0 0))
    (property "Spice_Primitive" "D" (at 0 0 0))
    (property "Spice_Model" "BAT54" (at 0 0 0))
    (property "Spice_Node_Sequence" "1 3" (at 0 0 0))
    (symbol "BAT54_0_1"
      (pin passive line (at 0 0 0) (length 2.54) (name "A") (number "1"))
      (pin no_connect line (at 0 0 0) (length 2.54) (name "NC") (number "2"))
      (pin passive line (at 0 0 0) (length 2.54) (name "K") (number "3"))
    )
  )
  (symbol "R_10k" (in_bom yes) (on_board yes)
    (property "Reference" "R" (at 0 0 0))
    (property "Value" "R_10k" (at 0 0 0))
    (property "Footprint" "Resistor_SMD:R_0402_1005Metric" (at 0 0 0))
    (property "Sim.Device" "R" (at 0 0 0))
    (symbol "R_10k_0_1"
      (pin passive line (at 0 0 0) (length 2.54) (name "~") (number "1"))
      (pin passive line (at 0 0 0) (length 2.54) (name "~") (number "2"))
    )
  )
)"##;

    #[test]
    fn simulation_fields_become_the_parts_spice_model_in_model_pin_order() {
        let (parts, _) = read(SIM).unwrap();
        let buck = parts[0].spice.clone().unwrap();
        assert_eq!(
            (
                buck.model.as_str(),
                buck.library.as_deref(),
                buck.device.as_deref()
            ),
            ("BUCK36V_AVG", Some("models/buck.lib"), None)
        );
        // Sim.Pins: model pin 1 is symbol pin 5 (VIN), 2 is 4 (EN), 3 is 1 (BST) …
        assert_eq!(buck.nodes, ["VIN", "EN", "BST", "SW", "FB", "GND"]);
        let diode = parts[1].spice.clone().unwrap();
        assert_eq!(
            (diode.model.as_str(), diode.device.as_deref()),
            ("BAT54", Some("D"))
        );
        assert_eq!(diode.nodes, ["A", "K"]);
        assert_eq!(parts[2].spice, None, "a resistor simulates from its value");
        // The model library moves with the symbol library; a path variable stays.
        let moved = rebase(&parts[0], "/proj/libs", "/proj/boards");
        assert_eq!(
            moved.spice.unwrap().library.as_deref(),
            Some("../libs/models/buck.lib")
        );
        let mut var = parts[0].clone();
        var.spice.as_mut().unwrap().library = Some("${KICAD7_3RD_PARTY}/buck.lib".into());
        assert_eq!(
            rebase(&var, "/proj/libs", "/proj/boards")
                .spice
                .unwrap()
                .library
                .as_deref(),
            Some("${KICAD7_3RD_PARTY}/buck.lib")
        );
    }

    #[test]
    fn a_footprint_path_rebases_to_the_circuit_folder() {
        let (parts, _) = read(LIB).unwrap();
        assert_eq!(
            rebase(&parts[0], "/proj/libs/kicad", "/proj/boards").package,
            "../libs/kicad/Package_TO_SOT_SMD.pretty/SOT-23-5.kicad_mod"
        );
        assert_eq!(
            rebase(&parts[0], "/proj", "/proj").package,
            "Package_TO_SOT_SMD.pretty/SOT-23-5.kicad_mod"
        );
    }
}
