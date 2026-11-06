//! KiCad libraries opened in place: a `.kicad_sym` symbol library read as catalog parts whose
//! packages are the `.kicad_mod` files their `Footprint` property names, so the library panel
//! lists, previews and places them like any other part. Nothing is copied: a placed part's
//! declaration names the footprint file where it lies.
//!
//! A symbol's `Footprint` property `Lib:Name` resolves to `<Lib>.pretty/<Name>.kicad_mod`
//! beside the symbol library (the layout of KiCad's own library folders). Every unit's pins
//! become one pin list; pins that share a name (`GND` on three pads) become one pin on those
//! pads, and an unnamed pin (`~`) takes its number.

use super::catalog::{CatalogPart, CatalogPin};
use crate::kicad::sch::parse_lib_symbol;
use crate::kicad::sexpr;
use crate::library::PinType;

fn property<'a>(node: &'a sexpr::SExpr, key: &str) -> Option<&'a str> {
    node.children("property").find(|p| p.arg(0) == Some(key)).and_then(|p| p.arg(1)).filter(|v| !v.is_empty() && *v != "~")
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
pub fn read(text: &str) -> Result<(Vec<CatalogPart>, Vec<String>), String> {
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
        let mut pins: Vec<CatalogPin> = Vec::new();
        for pin in symbol.units.values().flatten() {
            let pin_name = if pin.name.is_empty() || pin.name == "~" { pin.number.clone() } else { pin.name.clone() };
            let kind = if pin.kind == PinType::Unspecified { PinType::Passive } else { pin.kind };
            match pins.iter_mut().find(|p| p.name == pin_name) {
                Some(p) if !p.pads.contains(&pin.number) => p.pads.push(pin.number.clone()),
                Some(_) => {}
                None => pins.push(CatalogPin { name: pin_name, pads: vec![pin.number.clone()], kind: kind.name().into() }),
            }
        }
        if pins.is_empty() {
            skipped.push(symbol.name.clone());
            continue;
        }
        parts.push(CatalogPart {
            mpn: property(node, "Value").unwrap_or(&symbol.name).to_string(),
            manufacturer: String::new(),
            description: property(node, "Description").or_else(|| property(node, "ki_description")).unwrap_or_default().to_string(),
            category: category(&symbol.prefix).into(),
            prefix: symbol.prefix.clone(),
            package: format!("{lib}.pretty/{name}.kicad_mod"),
            pins,
            params: Default::default(),
            datasheet: property(node, "Datasheet").filter(|d| d.starts_with("http")).map(str::to_owned),
            spice: None,
        });
    }
    Ok((parts, skipped))
}

/// `part` with its package rebased from the symbol library's folder `lib_dir` to `base`, the
/// folder of the circuit that will declare it (both workspace paths, `/`-separated).
pub fn rebase(part: &CatalogPart, lib_dir: &str, base: &str) -> CatalogPart {
    let target: Vec<&str> = lib_dir.split('/').filter(|s| !s.is_empty()).chain(part.package.split('/')).collect();
    let from: Vec<&str> = base.split('/').filter(|s| !s.is_empty()).collect();
    let common = target.iter().zip(&from).take_while(|(a, b)| a == b).count();
    let mut rel: Vec<&str> = std::iter::repeat("..").take(from.len() - common).collect();
    rel.extend(&target[common..]);
    CatalogPart { package: rel.join("/"), ..part.clone() }
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
        assert_eq!(ldo.datasheet.as_deref(), Some("https://example.com/ldo.pdf"));
        let gnd = ldo.pins.iter().find(|p| p.name == "GND").unwrap();
        assert_eq!((gnd.pads.clone(), gnd.kind.as_str()), (vec!["2".to_string(), "6".to_string()], "power_in"));
        assert_eq!(ldo.pins.len(), 5);
    }

    #[test]
    fn a_footprint_path_rebases_to_the_circuit_folder() {
        let (parts, _) = read(LIB).unwrap();
        assert_eq!(rebase(&parts[0], "/proj/libs/kicad", "/proj/boards").package, "../libs/kicad/Package_TO_SOT_SMD.pretty/SOT-23-5.kicad_mod");
        assert_eq!(rebase(&parts[0], "/proj", "/proj").package, "Package_TO_SOT_SMD.pretty/SOT-23-5.kicad_mod");
    }
}
