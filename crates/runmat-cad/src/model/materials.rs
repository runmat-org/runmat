//! Material densities for mass properties, matched by the words in a material's name
//! (`material.aluminum_7075`, `steel_300m`, `prepreg_carbon`), most specific first.
//!
//! A part can state its own density as a property (`density = 1450 kg/m^3` on its output) for a
//! material this table doesn't know or a lumped body (a battery, a potted module); that value
//! wins ([`part_density`]). Values are nominal room-temperature densities in kg/m³.

use crate::model::{PartOutput, PropertyValue};
use crate::value::ValueExpr;

/// (words, density): the first entry with a word in the material name wins.
const DENSITIES: &[(&[&str], f64)] = &[
    (&["7075", "al_7075"], 2810.0),
    (&["2024", "al_2024"], 2780.0),
    (&["6061", "al_6061", "6063"], 2700.0),
    (&["aluminum", "aluminium"], 2700.0),
    (&["ti_6al_4v", "ti6al4v"], 4430.0),
    (&["titanium"], 4500.0),
    (&["stainless", "304", "316", "17_4"], 7950.0),
    (&["inconel"], 8190.0),
    // Structural steel grades by name (EN 10025, ASTM A36).
    (&["steel", "iron", "s235", "s275", "s355", "a36"], 7850.0),
    (&["nickel"], 8900.0),
    (&["copper", "c110", "c101"], 8940.0),
    (&["brass"], 8500.0),
    (&["bronze"], 8800.0),
    (&["gold"], 19300.0),
    (&["magnesium", "az31"], 1770.0),
    (
        &["carbon_fiber", "carbon_fibre", "cfrp", "prepreg", "carbon"],
        1600.0,
    ),
    (&["fiberglass", "fibreglass", "gfrp", "composite"], 1850.0),
    (&["fr4", "fr_4", "board"], 1850.0),
    // Glass-filled polyamide (PA6/PA66 with 30 % glass) before plain glass and nylon.
    (
        &[
            "glass_filled_nylon",
            "pa6_gf30",
            "pa66_gf30",
            "nylon_gf",
            "glass_filled",
        ],
        1360.0,
    ),
    (&["glass", "glazing"], 2500.0),
    (&["acrylic", "pmma"], 1180.0),
    (&["polycarbonate"], 1200.0),
    (&["abs"], 1050.0),
    (&["nylon", "pa6", "pa12"], 1140.0),
    (&["pom", "acetal", "delrin"], 1410.0),
    (&["peek"], 1320.0),
    (&["pla"], 1240.0),
    (&["petg"], 1270.0),
    (&["polypropylene"], 905.0),
    (&["hdpe", "polyethylene"], 950.0),
    (&["pvc"], 1380.0),
    (&["epoxy", "mold_compound", "mould_compound"], 1850.0),
    (&["silicone", "gap_pad"], 2400.0),
    (
        &["rubber", "elastomer", "neoprene", "epdm", "nitrile"],
        1200.0,
    ),
    (&["wood", "pine"], 500.0),
    (&["oak"], 750.0),
    (&["plywood", "birch"], 650.0),
];

/// The density (kg/m³) of a material by its name, when this table knows it.
pub fn material_density(material: &str) -> Option<f64> {
    let name = material
        .strip_prefix("material.")
        .unwrap_or(material)
        .to_lowercase();
    let words: Vec<&str> = name
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect();
    DENSITIES.iter().find_map(|(keys, density)| {
        keys.iter()
            .any(|k| {
                if k.contains('_') {
                    name.contains(k)
                } else {
                    words.contains(k)
                }
            })
            .then_some(*density)
    })
}

/// Where a part's density came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DensitySource {
    /// The part's own `density` property.
    Property,
    /// Its material, from the table.
    Material,
}

/// A part output's density (kg/m³) and where it came from: its `density` property (a quantity
/// in kg/m³, g/cm³ or a bare number in kg/m³), else its material.
pub fn part_density(output: &PartOutput) -> Option<(f64, DensitySource)> {
    let stated = output
        .properties
        .iter()
        .find(|p| p.name.as_str() == "density")
        .and_then(|p| match &p.value {
            PropertyValue::Quantity(ValueExpr::Quantity { value, unit }) => {
                match unit.replace(' ', "").as_str() {
                    "kg/m^3" | "kg/m3" => Some(*value),
                    "g/cm^3" | "g/cm3" => Some(value * 1000.0),
                    _ => None,
                }
            }
            PropertyValue::Quantity(ValueExpr::Number { value }) => Some(*value),
            _ => None,
        });
    if let Some(d) = stated {
        return Some((d, DensitySource::Property));
    }
    output
        .material
        .as_ref()
        .and_then(|m| material_density(m.as_str()))
        .map(|d| (d, DensitySource::Material))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn densities_match_the_most_specific_name() {
        assert_eq!(material_density("material.aluminum_7075"), Some(2810.0));
        assert_eq!(material_density("material.aluminum"), Some(2700.0));
        assert_eq!(material_density("steel_300m"), Some(7850.0));
        assert_eq!(material_density("material.prepreg_carbon"), Some(1600.0));
        assert_eq!(material_density("glass_filled_nylon"), Some(1360.0));
        assert_eq!(material_density("glass"), Some(2500.0));
        assert_eq!(material_density("S235"), Some(7850.0));
        assert_eq!(material_density("material.unobtainium"), None);
    }
}
