//! Material densities for mass properties, matched by the words in a material's name
//! (`material.aluminum_7075`, `steel_300m`, `prepreg_carbon`), most specific first.
//!
//! A part can state its own density as a property (`properties = [{ name = density, value =
//! { type = quantity, value = 1450 kg/m^3 } }]` on its output) for a
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
    // Building insulation before the glass-fibre composites: a fibreglass batt is mostly air.
    (&["batt", "mineral_wool", "glass_wool", "rock_wool"], 20.0),
    (&["xps", "extruded_polystyrene"], 35.0),
    (
        &["eps", "expanded_polystyrene", "polystyrene", "foam_board"],
        20.0,
    ),
    // Construction materials (residential practice): normal-weight concrete, compacted
    // crushed-stone base, cement plaster, gypsum board, asphalt shingles, sheathing and lumber;
    // before the board and composite entries their names share words with (`gypsum_board`).
    (&["concrete", "grout"], 2400.0),
    (&["gravel", "crushed_stone", "aggregate"], 1600.0),
    (&["stucco", "cement_plaster", "mortar"], 1860.0),
    (&["gypsum", "drywall"], 700.0),
    (&["asphalt", "shingle"], 1100.0),
    (&["osb", "oriented_strand"], 620.0),
    (&["lvl", "lsl", "glulam", "engineered_lumber"], 600.0),
    (&["douglas_fir", "hem_fir", "spf", "lumber"], 530.0),
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

/// The material library a part or body picks from: `(name, label)`, the name as the Source
/// writes it (`material = "material.steel_1018"`). Every entry has a density here and a display family in
/// [`crate::appearance`], so mass properties and the canvas color follow the pick.
pub const LIBRARY: &[(&str, &str)] = &[
    ("material.aluminum_6061_t6", "Aluminum 6061-T6"),
    ("material.aluminum_7075_t6", "Aluminum 7075-T6"),
    ("material.aluminum_2024_t3", "Aluminum 2024-T3"),
    ("material.steel_1018", "Steel 1018"),
    ("material.steel_4140", "Steel 4140"),
    ("material.stainless_304", "Stainless steel 304"),
    ("material.stainless_316", "Stainless steel 316"),
    ("material.titanium_ti_6al_4v", "Titanium Ti-6Al-4V"),
    ("material.copper_c110", "Copper C110"),
    ("material.brass_c360", "Brass C360"),
    ("material.abs", "ABS"),
    ("material.nylon_pa12", "Nylon PA12"),
    ("material.glass_filled_nylon", "Nylon PA66 GF30"),
    ("material.pom_acetal", "Acetal (POM)"),
    ("material.peek", "PEEK"),
    ("material.polycarbonate", "Polycarbonate"),
    ("material.acrylic", "Acrylic (PMMA)"),
    ("material.carbon_fiber", "Carbon fiber laminate"),
    ("material.fr4", "FR-4"),
    ("material.silicone_rubber", "Silicone rubber"),
    ("material.oak", "Oak"),
    ("material.concrete", "Concrete, normal weight"),
    ("material.douglas_fir_lumber", "Douglas fir-larch lumber"),
    ("material.osb", "Oriented strand board (OSB)"),
    ("material.gypsum_board", "Gypsum board"),
    ("material.xps_foam", "Extruded polystyrene (XPS)"),
];

/// A library material's label (`material.steel_1018` → `Steel 1018`), when it is in [`LIBRARY`].
/// Whether `material` is compressible fill: fibrous batt or loose-fill insulation. A cable, pipe
/// or box set in an insulated cavity displaces the fill (the installer splits or cuts the batt
/// around it), so interference checks leave fill bodies out.
pub fn is_compressible_fill(material: &str) -> bool {
    let m = material.to_ascii_lowercase();
    [
        "batt",
        "loose_fill",
        "glass_wool",
        "mineral_wool",
        "rock_wool",
        "cellulose",
    ]
    .iter()
    .any(|k| m.contains(k))
}

pub fn library_label(material: &str) -> Option<&'static str> {
    let name = material.strip_prefix("material.").unwrap_or(material);
    LIBRARY
        .iter()
        .find(|(n, _)| n.strip_prefix("material.") == Some(name))
        .map(|(_, label)| *label)
}

/// The density (kg/m³) of a material by its name, when this table knows it.
pub fn material_density(material: &str) -> Option<f64> {
    first_match(material, DENSITIES)
}

/// The value of the first table entry with a key in `material`'s name: a key with an underscore
/// anywhere in the name, any other key as a whole word (`eps` is not in `steps`).
fn first_match<T: Copy>(material: &str, table: &[(&[&str], T)]) -> Option<T> {
    let name = material
        .strip_prefix("material.")
        .unwrap_or(material)
        .to_lowercase();
    let words: Vec<&str> = name
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect();
    table.iter().find_map(|(keys, value)| {
        keys.iter()
            .any(|k| {
                if k.contains('_') {
                    name.contains(k)
                } else {
                    words.contains(k)
                }
            })
            .then_some(*value)
    })
}

/// The section lining a cut through a material draws (ASME Y14.2 section-lining symbols, and
/// the architectural material symbols for building materials).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SectionSymbol {
    /// 45° lines, evenly spaced: cast iron, and any material without its own symbol.
    #[default]
    General,
    /// 45° lines in pairs.
    Steel,
    /// Stippling with small triangles of aggregate.
    Concrete,
    /// Wood across the grain: diagonals corner to corner.
    Wood,
    /// Batt and loose-fill insulation: a row of loops across the cavity.
    Insulation,
    /// Rigid board insulation: 45° lines both ways.
    RigidInsulation,
    /// Gypsum board and plaster: fine stippling.
    Gypsum,
    /// Earth and compacted fill: short strokes in groups of three, alternating direction.
    Earth,
}

/// (keys, symbol): the first entry with a key in the material name wins; insulation before the
/// fibre and foam words it shares, plaster before the cement words.
const SECTION_SYMBOLS: &[(&[&str], SectionSymbol)] = &[
    (
        &[
            "batt",
            "loose_fill",
            "glass_wool",
            "mineral_wool",
            "rock_wool",
            "cellulose",
        ],
        SectionSymbol::Insulation,
    ),
    (
        &["xps", "eps", "polystyrene", "polyiso", "foam_board"],
        SectionSymbol::RigidInsulation,
    ),
    (
        &["gypsum", "drywall", "stucco", "plaster", "cement_plaster"],
        SectionSymbol::Gypsum,
    ),
    (&["concrete", "grout", "mortar"], SectionSymbol::Concrete),
    (
        &["earth", "soil", "gravel", "crushed_stone", "aggregate", "backfill"],
        SectionSymbol::Earth,
    ),
    (
        &[
            "lumber", "wood", "douglas_fir", "hem_fir", "spf", "pine", "oak", "birch", "plywood",
            "osb", "oriented_strand", "lvl", "lsl", "glulam", "i_joist", "timber",
        ],
        SectionSymbol::Wood,
    ),
    (
        &["steel", "stainless", "iron", "rebar", "s235", "s275", "s355", "a36"],
        SectionSymbol::Steel,
    ),
];

/// The section lining for a cut through `material` (a library id such as
/// `material.douglas_fir_lumber`); [`SectionSymbol::General`] for a material without its own.
pub fn section_symbol(material: Option<&str>) -> SectionSymbol {
    material
        .and_then(|m| first_match(m, SECTION_SYMBOLS))
        .unwrap_or(SectionSymbol::General)
}

/// Where a part's density came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DensitySource {
    /// The part's own `density` property.
    Property,
    /// Its material, from the table.
    Material,
}

/// A part output's density (kg/m³) and where it came from: its `density` property (a quantity
/// in kg/m³, g/cm³ or a bare number in kg/m³), else its material from the table.
///
/// There is no default density. When neither gives one, the `Err` says why, naming the material
/// it looked for, so mass properties report the mass as unknown instead of guessing it.
pub fn part_density(output: &PartOutput) -> Result<(f64, DensitySource), String> {
    if let Some(property) = output
        .properties
        .iter()
        .find(|p| p.name.as_str() == "density")
    {
        let stated = match &property.value {
            PropertyValue::Quantity(ValueExpr::Quantity { value, unit }) => {
                match unit.replace(' ', "").as_str() {
                    "kg/m^3" | "kg/m3" => Ok(*value),
                    "g/cm^3" | "g/cm3" => Ok(value * 1000.0),
                    other => Err(format!(
                        "its `density` property is in {other}; write it in kg/m^3 or g/cm^3"
                    )),
                }
            }
            PropertyValue::Quantity(ValueExpr::Number { value }) => Ok(*value),
            _ => Err("its `density` property is not a number in kg/m^3 or g/cm^3".to_owned()),
        }?;
        if !(stated.is_finite() && stated > 0.0) {
            return Err(format!(
                "its `density` property is {stated} kg/m^3; a density is more than zero"
            ));
        }
        return Ok((stated, DensitySource::Property));
    }
    let material = output.material.as_ref().map(|m| m.as_str());
    match material.and_then(material_density) {
        Some(d) => Ok((d, DensitySource::Material)),
        None => Err(no_density_reason(material)),
    }
}

/// Why a part with `material` (or none) has no density.
pub fn no_density_reason(material: Option<&str>) -> String {
    match material {
        Some(m) => format!(
            "no density is known for material `{m}`; pick a library material or give the part a `density` property"
        ),
        None => "it has no material and no `density` property".to_owned(),
    }
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
        // Building materials: a gypsum board is not a circuit board, a batt is not a laminate.
        assert_eq!(material_density("material.gypsum_board"), Some(700.0));
        assert_eq!(material_density("material.fiberglass_batt"), Some(20.0));
        assert_eq!(material_density("material.concrete_4000psi"), Some(2400.0));
        assert_eq!(material_density("material.douglas_fir_lumber"), Some(530.0));
        assert_eq!(material_density("material.xps_foam"), Some(35.0));
    }

    /// The first output of a one-line part with `output` written as given.
    fn output(fields: &str) -> PartOutput {
        let text = format!(
            "runmat 0.7\nstd 1\n\npart p {{\n    sketch s on origin.xy {{\n        c = circle(center = [0 mm, 0 mm], radius = 5 mm)\n    }}\n    feature b = extrude(profile = sketch.s, distance = 10 mm)\n    output p = {{ body = feature.b.body{fields} }}\n}}\n"
        );
        let parsed = crate::syntax::parse_part("p.part", &text);
        let document = parsed.document.expect("parses");
        document.parts.into_iter().next().expect("one output")
    }

    #[test]
    fn a_part_density_is_never_assumed() {
        assert_eq!(
            part_density(&output(", material = \"material.aluminum\"")),
            Ok((2700.0, DensitySource::Material))
        );
        assert_eq!(
            part_density(&output(", material = \"material.unobtainium\"")),
            Err("no density is known for material `material.unobtainium`; pick a library material or give the part a `density` property".to_owned())
        );
        assert_eq!(
            part_density(&output("")),
            Err("it has no material and no `density` property".to_owned())
        );
        // A stated density wins over the material, in kg/m³ or g/cm³.
        assert_eq!(
            part_density(&output(", material = \"material.unobtainium\", properties = [{ name = density, value = { type = quantity, value = 1.5 g/cm^3 } }]")),
            Ok((1500.0, DensitySource::Property))
        );
        // A density in a unit mass properties don't read is unknown, not skipped for the
        // material's.
        let odd = part_density(&output(", material = \"material.steel\", properties = [{ name = density, value = { type = quantity, value = 0.28 lb/in^3 } }]"));
        assert!(odd.as_ref().is_err_and(|e| e.contains("lb/in^3")), "{odd:?}");
        let zero = part_density(&output(", properties = [{ name = density, value = { type = quantity, value = 0 kg/m^3 } }]"));
        assert!(zero.is_err(), "{zero:?}");
    }

    #[test]
    fn every_library_material_has_a_density_and_a_color() {
        for (name, label) in LIBRARY {
            let written = name.to_string();
            assert!(
                material_density(&written).is_some(),
                "{name} has no density"
            );
            assert_ne!(
                crate::appearance::material_appearance(Some(&written)).0,
                "default",
                "{name} has no color"
            );
            assert_eq!(library_label(&written), Some(*label));
        }
        assert_eq!(material_density("material.stainless_304"), Some(7950.0));
        assert_eq!(library_label("unobtainium"), None);
    }

    #[test]
    fn a_cut_draws_its_materials_section_symbol() {
        use super::{section_symbol, SectionSymbol::*};
        for (material, symbol) in [
            ("material.concrete", Concrete),
            ("material.douglas_fir_lumber", Wood),
            ("material.osb", Wood),
            ("material.engineered_lumber_i_joist", Wood),
            ("material.fiberglass_batt", Insulation),
            ("material.glass_wool_loose_fill", Insulation),
            ("material.xps_foam", RigidInsulation),
            ("material.gypsum_board", Gypsum),
            ("material.stucco_brown_coat", Gypsum),
            ("material.steel_1018", Steel),
            ("material.galvanized_steel", Steel),
            ("material.crushed_stone_gravel", Earth),
            ("material.asphalt_saturated_kraft_wrb", General),
            ("material.copper_insulated", General),
            ("material.aluminum_6061_t6", General),
        ] {
            assert_eq!(section_symbol(Some(material)), symbol, "{material}");
        }
        assert_eq!(section_symbol(None), General);
    }
}
