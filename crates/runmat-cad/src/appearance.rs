//! A material's display color, matched by family words in its name (`aluminum_7075`,
//! `material.steel_300m`, `ABS plastic`): one flat sRGB display color and opacity per family, as
//! the Desktop canvas shades parts and as the exchange formats (3MF base materials, glTF
//! materials) carry them. A material this table doesn't know, or no material, takes the CAD
//! face color.

/// The default part grey, readable on light and dark backgrounds.
pub const CAD_FACE_RGBA: [f32; 4] = [0.8, 0.82, 0.85, 1.0];

/// Families in match order: the first whose words appear in the material name wins.
const FAMILIES: &[(&str, &[&str], [f32; 4])] = &[
    ("glass", &["glass", "glazing", "acrylic", "pmma", "polycarbonate_clear", "clear", "sapphire", "quartz"], [0.55, 0.68, 0.78, 0.35]),
    ("rubber", &["rubber", "elastomer", "silicone", "neoprene", "epdm", "nitrile", "viton", "tyre", "tire", "urethane_rubber"], [0.14, 0.14, 0.15, 1.0]),
    ("copper", &["copper", "c110", "c101"], [0.78, 0.5, 0.36, 1.0]),
    ("brass", &["brass", "bronze"], [0.8, 0.68, 0.4, 1.0]),
    ("gold", &["gold", "enig"], [0.86, 0.72, 0.38, 1.0]),
    ("titanium", &["titanium", "ti_6al_4v", "ti6al4v"], [0.6, 0.6, 0.62, 1.0]),
    ("steel", &["steel", "stainless", "inconel", "iron", "nickel"], [0.62, 0.64, 0.67, 1.0]),
    ("aluminum", &["aluminum", "aluminium", "al_6061", "al_7075", "al_2024"], [0.78, 0.8, 0.83, 1.0]),
    ("carbon", &["carbon_fiber", "carbon_fibre", "cfrp"], [0.22, 0.23, 0.25, 1.0]),
    ("composite", &["composite", "gfrp", "fiberglass", "fibreglass", "laminate"], [0.88, 0.89, 0.9, 1.0]),
    // Moulded package bodies (ICs, regulators): near-black epoxy.
    ("epoxy", &["epoxy", "mold_compound", "mould_compound"], [0.16, 0.16, 0.17, 1.0]),
    ("fr4", &["fr4", "fr_4"], [0.3, 0.42, 0.24, 1.0]),
    ("plastic", &["plastic", "abs", "nylon", "pom", "acetal", "delrin", "pla", "petg", "polypropylene", "polyethylene", "hdpe", "peek", "pvc", "polycarbonate"], [0.9, 0.9, 0.88, 1.0]),
    ("wood", &["wood", "oak", "pine", "plywood", "birch"], [0.72, 0.56, 0.38, 1.0]),
];

/// A material's family (`default` when none matches) and its color.
pub fn material_appearance(material: Option<&str>) -> (&'static str, [f32; 4]) {
    let Some(m) = material else {
        return ("default", CAD_FACE_RGBA);
    };
    let name: String = m
        .strip_prefix("material.")
        .unwrap_or(m)
        .to_lowercase()
        .split(|c: char| c.is_whitespace() || c == '-')
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("_");
    let tokens: Vec<&str> = name.split('_').collect();
    for (family, words, color) in FAMILIES {
        if words
            .iter()
            .any(|w| if w.contains('_') { name.contains(w) } else { tokens.contains(w) })
        {
            return (family, *color);
        }
    }
    ("default", CAD_FACE_RGBA)
}

/// `#RRGGBBAA` of a display color, as 3MF's `displaycolor` takes it.
pub fn srgb_hex(c: [f32; 4]) -> String {
    let b = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    format!("#{:02X}{:02X}{:02X}{:02X}", b(c[0]), b(c[1]), b(c[2]), b(c[3]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn families_match_by_words() {
        assert_eq!(material_appearance(Some("material.aluminum_6061")).0, "aluminum");
        assert_eq!(material_appearance(Some("ABS plastic")).0, "plastic");
        assert_eq!(material_appearance(Some("steel_300m")).0, "steel");
        assert_eq!(material_appearance(Some("unobtainium")).0, "default");
        assert_eq!(material_appearance(None).1, CAD_FACE_RGBA);
        assert_eq!(material_appearance(Some("material.epoxy")).0, "epoxy");
        assert!(material_appearance(Some("glass")).1[3] < 0.5, "glass is see-through");
        assert_eq!(srgb_hex([1.0, 0.0, 0.0, 1.0]), "#FF0000FF");
    }
}
