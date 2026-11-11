//! A material's display color, matched by family words in its name (`aluminum_7075`,
//! `material.steel_300m`, `ABS plastic`): one flat sRGB display color and opacity per family, as
//! the Desktop canvas shades parts and as the exchange formats (3MF base materials, glTF
//! materials) carry them. A material this table doesn't know, or no material, takes the CAD
//! face color.

/// The default part grey, readable on light and dark backgrounds.
pub const CAD_FACE_RGBA: [f32; 4] = [0.8, 0.82, 0.85, 1.0];

/// Families in match order: the first whose words appear in the material name wins.
const FAMILIES: &[(&str, &[&str], [f32; 4])] = &[
    // Glass-filled polymers (PA6/PA66 GF30) are opaque plastics: matched before plain glass.
    ("plastic", &["glass_filled", "glass_fibre_filled", "glass_fiber_filled", "nylon_gf", "pa6_gf", "pa66_gf", "gf30"], [0.9, 0.9, 0.88, 1.0]),
    ("glass", &["glass", "glazing", "acrylic", "pmma", "polycarbonate_clear", "clear", "sapphire", "quartz"], [0.55, 0.68, 0.78, 0.35]),
    ("rubber", &["rubber", "elastomer", "silicone", "neoprene", "epdm", "nitrile", "viton", "tyre", "tire", "urethane_rubber"], [0.14, 0.14, 0.15, 1.0]),
    ("copper", &["copper", "c110", "c101"], [0.78, 0.5, 0.36, 1.0]),
    ("brass", &["brass", "bronze"], [0.8, 0.68, 0.4, 1.0]),
    ("gold", &["gold", "enig"], [0.86, 0.72, 0.38, 1.0]),
    ("titanium", &["titanium", "ti_6al_4v", "ti6al4v"], [0.6, 0.6, 0.62, 1.0]),
    ("steel", &["steel", "stainless", "inconel", "iron", "nickel"], [0.62, 0.64, 0.67, 1.0]),
    ("aluminum", &["aluminum", "aluminium", "al_6061", "al_7075", "al_2024"], [0.78, 0.8, 0.83, 1.0]),
    // Building materials, before the composites and plastics their names overlap with
    // (a fibreglass batt, polyethylene sheet stays plastic).
    ("insulation", &["batt", "mineral_wool", "glass_wool", "rock_wool"], [0.93, 0.8, 0.5, 1.0]),
    ("foam", &["xps", "eps", "polystyrene", "extruded_polystyrene", "expanded_polystyrene", "foam_board"], [0.58, 0.76, 0.88, 1.0]),
    ("concrete", &["concrete", "grout", "mortar"], [0.66, 0.66, 0.64, 1.0]),
    ("gravel", &["gravel", "crushed_stone", "aggregate"], [0.52, 0.5, 0.47, 1.0]),
    ("plaster", &["stucco", "cement_plaster", "gypsum", "drywall"], [0.9, 0.88, 0.82, 1.0]),
    ("asphalt", &["asphalt", "shingle"], [0.3, 0.3, 0.32, 1.0]),
    ("osb", &["osb", "oriented_strand"], [0.8, 0.66, 0.42, 1.0]),
    ("carbon", &["carbon_fiber", "carbon_fibre", "cfrp"], [0.22, 0.23, 0.25, 1.0]),
    ("composite", &["composite", "gfrp", "fiberglass", "fibreglass", "laminate"], [0.88, 0.89, 0.9, 1.0]),
    // Moulded package bodies (ICs, regulators): near-black epoxy.
    ("epoxy", &["epoxy", "mold_compound", "mould_compound"], [0.16, 0.16, 0.17, 1.0]),
    ("fr4", &["fr4", "fr_4"], [0.3, 0.42, 0.24, 1.0]),
    ("plastic", &["plastic", "abs", "nylon", "pom", "acetal", "delrin", "pla", "petg", "polypropylene", "polyethylene", "hdpe", "peek", "pvc", "polycarbonate"], [0.9, 0.9, 0.88, 1.0]),
    ("wood", &["wood", "oak", "pine", "plywood", "birch", "lumber", "douglas_fir", "hem_fir", "spf", "lvl", "lsl", "glulam"], [0.72, 0.56, 0.38, 1.0]),
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

/// A display color from sRGB hex: `#RRGGBB` (opaque) or `#RRGGBBAA`, the `#` optional.
pub fn parse_srgb_hex(text: &str) -> Option<[f32; 4]> {
    let hex = text.trim().strip_prefix('#').unwrap_or(text.trim());
    if !(hex.len() == 6 || hex.len() == 8) || !hex.is_ascii() {
        return None;
    }
    let byte = |k: usize| {
        u8::from_str_radix(&hex[k..k + 2], 16)
            .ok()
            .map(|b| b as f32 / 255.0)
    };
    Some([byte(0)?, byte(2)?, byte(4)?, if hex.len() == 8 { byte(6)? } else { 1.0 }])
}

/// The color a part draws in: its own `color` when it is valid hex, else its material's
/// appearance.
pub fn part_appearance(material: Option<&str>, color: Option<&str>) -> [f32; 4] {
    color
        .and_then(parse_srgb_hex)
        .unwrap_or_else(|| material_appearance(material).1)
}

/// Sets `display`'s appearance from the part document: its first output's color, else that
/// output's material appearance, the CAD grey without either; and colors each output's body
/// when the part has several ([`color_output_bodies`]). Every regeneration that draws a part,
/// in its default or any other configuration, takes its colors from here.
pub fn apply_document_appearance(
    doc: &crate::model::PartDocument,
    result: &crate::regen::engine::RegenResult,
    display: &mut crate::display::PartDisplay,
) {
    let output = doc.parts.first();
    let material = output
        .and_then(|p| p.material.as_ref())
        .or_else(|| doc.parts.iter().find_map(|p| p.material.as_ref()));
    display.appearance = Some(part_appearance(
        material.map(|m| m.0.as_str()),
        output.and_then(|p| p.color.as_deref()),
    ));
    color_output_bodies(doc, result, display);
}

/// Gives each output's body its own color (the output's `color`, else its material's
/// appearance) when the part has more than one output, so a study part's solid, fluid and air
/// bodies read apart; then gives every body a `material = { body, material }` statement names
/// that material's appearance, so a wall's batts, plates and foam read apart in one part. A body
/// that already carries a color keeps it; a one-output part with no material statements draws in
/// the part's appearance as before.
pub fn color_output_bodies(
    doc: &crate::model::PartDocument,
    result: &crate::regen::engine::RegenResult,
    display: &mut crate::display::PartDisplay,
) {
    color_outputs(doc, result, display);
    for (m, o) in doc.materials.iter().zip(&result.materials) {
        let rgba = material_appearance(Some(m.material.0.as_str())).1;
        for b in assigned_bodies(result, o) {
            if let Some(body) = display.bodies.iter_mut().find(|d| d.index as usize == b) {
                body.color.get_or_insert(rgba);
            }
        }
    }
}

/// Every live body a material statement resolved to: each body carrying all the names of one
/// resolved entity, so a feature that makes several bodies (a multi-loop extrude, a split) colors
/// them all.
pub fn assigned_bodies(
    result: &crate::regen::engine::RegenResult,
    o: &crate::regen::engine::ItemOutcome,
) -> Vec<usize> {
    if !o.ok {
        return Vec::new();
    }
    let mut out = Vec::new();
    for names in &o.entities {
        let want: std::collections::BTreeSet<_> = names.iter().collect();
        if want.is_empty() {
            continue;
        }
        for (b, body) in result.state.bodies.iter().enumerate() {
            if want.iter().all(|n| body.names.body.contains(*n)) && !out.contains(&b) {
                out.push(b);
            }
        }
    }
    out
}

fn color_outputs(
    doc: &crate::model::PartDocument,
    result: &crate::regen::engine::RegenResult,
    display: &mut crate::display::PartDisplay,
) {
    if doc.parts.len() < 2 {
        return;
    }
    for p in &doc.parts {
        if p.color.is_none() && p.material.is_none() {
            continue;
        }
        let Some(b) = result
            .parts
            .get(&p.name)
            .and_then(|o| crate::regen::slots::resolved_body(result, o))
        else {
            continue;
        };
        let rgba = part_appearance(p.material.as_ref().map(|m| m.0.as_str()), p.color.as_deref());
        if let Some(body) = display.bodies.iter_mut().find(|d| d.index as usize == b) {
            body.color.get_or_insert(rgba);
        }
    }
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
        // Glass-filled nylon is an opaque plastic, not glass.
        assert_eq!(material_appearance(Some("glass_filled_nylon")).0, "plastic");
        assert_eq!(material_appearance(Some("PA66-GF30")).0, "plastic");
        assert_eq!(material_appearance(Some("glass_filled_nylon")).1[3], 1.0);
        assert_eq!(material_appearance(Some("material.fiberglass_batt")).0, "insulation");
        assert_eq!(material_appearance(Some("material.concrete")).0, "concrete");
        assert_eq!(material_appearance(Some("material.douglas_fir_lumber")).0, "wood");
        assert_eq!(material_appearance(Some("material.polyethylene_sheet")).0, "plastic");
        assert_eq!(srgb_hex([1.0, 0.0, 0.0, 1.0]), "#FF0000FF");
    }

    #[test]
    fn a_part_color_overrides_its_material() {
        assert_eq!(parse_srgb_hex("#FF000080"), Some([1.0, 0.0, 0.0, 128.0 / 255.0]));
        assert_eq!(parse_srgb_hex("00ff00"), Some([0.0, 1.0, 0.0, 1.0]));
        assert_eq!(parse_srgb_hex("#12345"), None);
        assert_eq!(parse_srgb_hex("#GG0000"), None);
        assert_eq!(
            part_appearance(Some("glass"), Some("#1B232C")),
            [27.0 / 255.0, 35.0 / 255.0, 44.0 / 255.0, 1.0]
        );
        // An invalid color falls back to the material.
        assert_eq!(
            part_appearance(Some("glass"), Some("navy")),
            material_appearance(Some("glass")).1
        );
    }
}
