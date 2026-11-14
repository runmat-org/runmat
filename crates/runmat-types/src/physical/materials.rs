//! The shared material library: engineering materials with an id, a display label and the
//! nominal room-temperature properties a study, a part's mass and a board's package bodies read,
//! each row with its source.
//!
//! Values are stored in SI base units (kg/m³, Pa, 1/K, W/(m·K), J/(kg·K)). [`Material::fea_properties`]
//! writes them in the `.fea` material block's property names and units, so a study that picks a
//! library material declares the same numbers the library holds.
//!
//! A material the library doesn't hold is described by its properties, never by a name alone.

/// One library material.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Material {
    /// Stable id, also the `.fea` material declaration name (`material steel_s275 { ... }`).
    pub id: &'static str,
    /// The name people read on a card or in a parts list.
    pub label: &'static str,
    /// kg/m³.
    pub mass_density: f64,
    /// Pa.
    pub youngs_modulus: f64,
    pub poisson_ratio: f64,
    /// Linear expansion coefficient, 1/K.
    pub expansion_coefficient: f64,
    /// W/(m·K).
    pub conductivity: f64,
    /// J/(kg·K).
    pub specific_heat: f64,
    /// Minimum or typical yield strength, Pa, when the source gives one for this condition.
    pub yield_strength: Option<f64>,
    /// Where the values come from.
    pub source: &'static str,
}

const GPA: f64 = 1.0e9;
const MPA: f64 = 1.0e6;

/// The library, in the order a card offers them.
pub const MATERIALS: &[Material] = &[
    Material {
        id: "steel_s275",
        label: "Structural steel S275",
        mass_density: 7850.0,
        youngs_modulus: 210.0 * GPA,
        poisson_ratio: 0.3,
        expansion_coefficient: 12.0e-6,
        conductivity: 45.0,
        specific_heat: 600.0,
        yield_strength: Some(275.0 * MPA),
        source: "EN 1993-1-1 3.2.6 (E, nu, expansion) and Table 3.1 (fy, t <= 40 mm); EN 1991-1-1 Table A.4 (density); EN 1993-1-2 3.4.1 (conductivity and specific heat for simple calculations)",
    },
    Material {
        id: "steel_s355",
        label: "Structural steel S355",
        mass_density: 7850.0,
        youngs_modulus: 210.0 * GPA,
        poisson_ratio: 0.3,
        expansion_coefficient: 12.0e-6,
        conductivity: 45.0,
        specific_heat: 600.0,
        yield_strength: Some(355.0 * MPA),
        source: "EN 1993-1-1 3.2.6 (E, nu, expansion) and Table 3.1 (fy, t <= 40 mm); EN 1991-1-1 Table A.4 (density); EN 1993-1-2 3.4.1 (conductivity and specific heat for simple calculations)",
    },
    Material {
        id: "aluminum_6061_t6",
        label: "Aluminum 6061-T6",
        mass_density: 2700.0,
        youngs_modulus: 68.9 * GPA,
        poisson_ratio: 0.33,
        expansion_coefficient: 23.6e-6,
        conductivity: 167.0,
        specific_heat: 896.0,
        yield_strength: Some(276.0 * MPA),
        source: "ASM Handbook Vol. 2, typical room-temperature properties of wrought 6061-T6",
    },
    Material {
        id: "stainless_304",
        label: "Stainless steel 304 (annealed)",
        mass_density: 8000.0,
        youngs_modulus: 193.0 * GPA,
        poisson_ratio: 0.29,
        expansion_coefficient: 17.3e-6,
        conductivity: 16.2,
        specific_heat: 500.0,
        yield_strength: Some(205.0 * MPA),
        source: "ASM Handbook Vol. 1, typical properties of annealed 304 (expansion 0-100 C); yield is the ASTM A240 minimum",
    },
    Material {
        id: "titanium_ti6al4v",
        label: "Titanium Ti-6Al-4V (annealed)",
        mass_density: 4430.0,
        youngs_modulus: 113.8 * GPA,
        poisson_ratio: 0.342,
        expansion_coefficient: 8.6e-6,
        conductivity: 6.7,
        specific_heat: 526.3,
        yield_strength: Some(880.0 * MPA),
        source: "ASM Handbook Vol. 2, typical room-temperature properties of annealed Ti-6Al-4V",
    },
    Material {
        id: "copper_c110",
        label: "Copper C110 (annealed)",
        mass_density: 8890.0,
        youngs_modulus: 115.0 * GPA,
        poisson_ratio: 0.31,
        expansion_coefficient: 17.0e-6,
        conductivity: 388.0,
        specific_heat: 385.0,
        yield_strength: None,
        source: "ASM Handbook Vol. 2, typical room-temperature properties of C11000",
    },
];

/// The library material with `id`.
pub fn material(id: &str) -> Option<&'static Material> {
    MATERIALS.iter().find(|material| material.id == id)
}

/// The library material a written name means: its id, its label, or a name whose words all
/// appear in one label (`"Aluminum 6061"` → `aluminum_6061_t6`). `None` when no material or
/// more than one matches.
pub fn material_named(name: &str) -> Option<&'static Material> {
    let words = |text: &str| -> Vec<String> {
        text.split(|c: char| !c.is_ascii_alphanumeric())
            .filter(|w| !w.is_empty())
            .map(str::to_ascii_lowercase)
            .collect()
    };
    if let Some(found) = MATERIALS
        .iter()
        .find(|m| m.id == name || m.label.eq_ignore_ascii_case(name))
    {
        return Some(found);
    }
    let wanted = words(name);
    if wanted.is_empty() {
        return None;
    }
    let mut matches = MATERIALS.iter().filter(|m| {
        let label = words(m.label);
        wanted.iter().all(|w| label.contains(w))
    });
    let first = matches.next()?;
    matches.next().is_none().then_some(first)
}

/// The library's material ids, in card order.
pub fn material_ids() -> impl Iterator<Item = &'static str> {
    MATERIALS.iter().map(|material| material.id)
}

impl Material {
    /// The properties as a `.fea` material block names them, each with its value and unit:
    /// `("youngs_modulus", "210 GPa")`. `yield_strength` is left out: a study reads it from an
    /// acceptance criterion, not from the material block.
    pub fn fea_properties(&self) -> Vec<(&'static str, String)> {
        vec![
            ("mass_density", format!("{} kg/m^3", number(self.mass_density))),
            ("youngs_modulus", format!("{} GPa", number(self.youngs_modulus / GPA))),
            ("poisson_ratio", number(self.poisson_ratio)),
            ("expansion_coefficient", format!("{} K^-1", number(self.expansion_coefficient))),
            ("conductivity", format!("{} W/m/K", number(self.conductivity))),
            ("specific_heat", format!("{} J/kg/K", number(self.specific_heat))),
        ]
    }
}

/// A plain decimal with no float noise (`0.000012`, `68.9`, `7850`).
fn number(value: f64) -> String {
    let text = format!("{value:.9}");
    let text = text.trim_end_matches('0').trim_end_matches('.');
    text.to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_find_one_material() {
        assert_eq!(material_named("Aluminum 6061").map(|m| m.id), Some("aluminum_6061_t6"));
        assert_eq!(material_named("steel_s355").map(|m| m.id), Some("steel_s355"));
        assert_eq!(material_named("Structural steel").map(|m| m.id), None);
        assert_eq!(material_named("unobtainium"), None);
    }

    #[test]
    fn ids_are_unique_fea_names_and_every_row_has_a_source() {
        let mut seen = std::collections::HashSet::new();
        for material in MATERIALS {
            assert!(seen.insert(material.id), "{}", material.id);
            assert!(material.id.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_'), "{}", material.id);
            assert!(!material.source.is_empty() && !material.label.is_empty());
            assert!(material.poisson_ratio > 0.0 && material.poisson_ratio < 0.5);
            assert!(material.mass_density > 0.0 && material.youngs_modulus > 0.0);
        }
        // A choice card offers at most eight choices, one of them "not in the library".
        assert!(MATERIALS.len() < 8);
    }

    #[test]
    fn fea_properties_write_the_library_values_in_block_units() {
        let steel = material("steel_s275").unwrap();
        let properties = steel.fea_properties();
        assert!(properties.contains(&("youngs_modulus", "210 GPa".to_owned())));
        assert!(properties.contains(&("expansion_coefficient", "0.000012 K^-1".to_owned())));
        assert!(properties.contains(&("mass_density", "7850 kg/m^3".to_owned())));
        assert_eq!(material("aluminum_6061_t6").unwrap().fea_properties()[1].1, "68.9 GPa");
        assert!(material("unobtainium").is_none());
    }
}
