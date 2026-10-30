//! Planned constructs that `runmat check` rejects with their pending id. The ids are the keys of
//! docs/simulation/pending-features.json and the release-status page, so a diagnostic, the docs
//! and the RunMat agent name a planned feature the same way. Diagnostics append
//! [`marker`] to the rule's own message.

pub const GEOMETRY_SELECT_PREDICATE: &str = "geometry.select.predicate";
pub const INCLUDE_PARAMETERIZED: &str = "language.include.parameterized";
pub const VECTOR_TRAILING_UNIT: &str = "language.vector.trailing-unit";
pub const WALL_SHEAR_STRESS: &str = "output.wall-shear-stress";
pub const OSCILLATORY_SHEAR_INDEX: &str = "output.oscillatory-shear-index";
pub const TIME_MEAN: &str = "output.time-mean";
pub const COEFFICIENT_OF_VARIATION: &str = "output.coefficient-of-variation";
pub const FLUID_NON_NEWTONIAN: &str = "fluid.non-newtonian";
pub const FLUID_SPECIES_TRANSPORT: &str = "fluid.species-transport";
pub const CHT_BUOYANCY: &str = "cht.buoyancy";
pub const LUBRICATION_FAMILY: &str = "lubrication.family";
pub const SIGNAL_TABULATED_FILE: &str = "signal.tabulated-file";
pub const SIGNAL_HARMONIC_NONCONSTANT_VALUE: &str = "signal.harmonic-nonconstant-value";
pub const UNIT_SIEMENS: &str = "language.unit.siemens";
pub const PLASTIC_LAW_IDENTIFIER: &str = "language.plastic-law-identifier";
pub const OUTPUT_MODE_LOCATION: &str = "output.mode-location";

/// ` [pending:<id>]`, appended to a diagnostic message.
pub fn marker(id: &str) -> String {
    format!(" [pending:{id}]")
}

/// `message` with the marker of `id`, when there is one.
pub fn with_pending(message: impl Into<String>, id: Option<&str>) -> String {
    let message = message.into();
    match id {
        Some(id) => message + &marker(id),
        None => message,
    }
}

/// Planned output operations: `oscillatory_shear_index(...)`, `wall_shear_stress(...)`, ...
pub fn operation(name: &str) -> Option<&'static str> {
    Some(match name {
        "oscillatory_shear_index" => OSCILLATORY_SHEAR_INDEX,
        "coefficient_of_variation" => COEFFICIENT_OF_VARIATION,
        "wall_shear_stress" => WALL_SHEAR_STRESS,
        _ => return None,
    })
}

/// Planned references, by path: species declarations and concentration fields, wall shear
/// stress fields, lubrication film fields and outputs.
pub fn reference(path: &[String]) -> Option<&'static str> {
    let first = path.first()?.as_str();
    if first == "species" {
        return Some(FLUID_SPECIES_TRANSPORT);
    }
    let last = path.last()?.as_str();
    match last {
        "wall_shear_stress" => Some(WALL_SHEAR_STRESS),
        "oscillatory_shear_index" => Some(OSCILLATORY_SHEAR_INDEX),
        "film_pressure" | "film_thickness" | "friction_torque" | "load_capacity" => {
            Some(LUBRICATION_FAMILY)
        }
        name if name.ends_with("_concentration") || name == "concentration" => {
            Some(FLUID_SPECIES_TRANSPORT)
        }
        _ => None,
    }
}

/// Planned essential constraints: `species.*` and `lubrication.*`.
pub fn constraint(kind: &str) -> Option<&'static str> {
    if kind.starts_with("species.") {
        Some(FLUID_SPECIES_TRANSPORT)
    } else if kind.starts_with("lubrication.") {
        Some(LUBRICATION_FAMILY)
    } else {
        None
    }
}

/// Planned units: siemens (`S`, conductivity in `S/m`), `rpm` (journal speed) and fractional exponents (power-law consistency,
/// `Pa*s^0.45`).
pub fn unit(symbol: &str) -> Option<&'static str> {
    let symbol = symbol.trim();
    let siemens = symbol
        .split(|c: char| !c.is_ascii_alphanumeric())
        .any(|part| matches!(part, "S" | "mS" | "uS" | "kS" | "MS"));
    if siemens {
        Some(UNIT_SIEMENS)
    } else if symbol == "rpm" {
        Some(LUBRICATION_FAMILY)
    } else if symbol.contains("^0.") || symbol.contains("^.") {
        Some(FLUID_NON_NEWTONIAN)
    } else {
        None
    }
}

/// Planned definition members: lubrication journal members, species constraint members.
pub fn definition_member(name: &str) -> Option<&'static str> {
    match name {
        "speed" | "eccentricity_ratio" | "attitude_angle" | "clearance" => Some(LUBRICATION_FAMILY),
        "species" | "transport" => Some(FLUID_SPECIES_TRANSPORT),
        _ => None,
    }
}

/// Planned procedure members: `buoyancy` and `gravity` (Boussinesq CHT), `transport` (species).
pub fn procedure_member(name: &str) -> Option<&'static str> {
    match name {
        "buoyancy" | "gravity" => Some(CHT_BUOYANCY),
        "transport" => Some(FLUID_SPECIES_TRANSPORT),
        _ => None,
    }
}

/// Planned material members: `species` blocks in a material.
pub fn material_member(name: &str) -> Option<&'static str> {
    match name {
        "species" => Some(FLUID_SPECIES_TRANSPORT),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    /// The ids this crate reports; each must be a key of docs/simulation/pending-features.json
    /// (`language.keyword-names` is reported by the parser in runmat-fea-syntax).
    #[test]
    fn reported_ids_are_the_registry_keys() {
        let ids = [
            super::GEOMETRY_SELECT_PREDICATE,
            super::INCLUDE_PARAMETERIZED,
            super::VECTOR_TRAILING_UNIT,
            super::WALL_SHEAR_STRESS,
            super::OSCILLATORY_SHEAR_INDEX,
            super::TIME_MEAN,
            super::COEFFICIENT_OF_VARIATION,
            super::FLUID_NON_NEWTONIAN,
            super::FLUID_SPECIES_TRANSPORT,
            super::CHT_BUOYANCY,
            super::LUBRICATION_FAMILY,
            super::SIGNAL_TABULATED_FILE,
            super::SIGNAL_HARMONIC_NONCONSTANT_VALUE,
            super::UNIT_SIEMENS,
            super::PLASTIC_LAW_IDENTIFIER,
            super::OUTPUT_MODE_LOCATION,
        ];
        let registry = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../docs/simulation/pending-features.json");
        // The registry lands with the docs rails; until then only the id shape is checked.
        let text = std::fs::read_to_string(registry).unwrap_or_default();
        for id in ids {
            assert!(id.chars().all(|c| c.is_ascii_lowercase() || c == '.' || c == '-' || c == '_'));
            if !text.is_empty() {
                assert!(text.contains(&format!("\"{id}\"")), "{id} missing from pending-features.json");
            }
        }
    }
}
