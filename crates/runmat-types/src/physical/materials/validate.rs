//! The checks every library row passes: a `.fea` name for an id, physical ranges, and a source for
//! every value.
use super::{citations, Material};

/// What a row gets wrong.
#[derive(Debug, Clone, PartialEq)]
pub enum MaterialDefect {
    /// The id isn't a `.fea` declaration name (lowercase ASCII, digits and `_`).
    Id,
    /// The label is empty.
    Label,
    /// A property is outside its physical range; the text names it.
    Range(&'static str),
    /// A property names a citation the library doesn't list.
    Citation(&'static str),
}

/// Checks one row.
pub fn validate_material(material: &Material) -> Result<(), MaterialDefect> {
    if material.id.is_empty()
        || !material.id.starts_with(|c: char| c.is_ascii_lowercase())
        || !material.id.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
    {
        return Err(MaterialDefect::Id);
    }
    if material.label.trim().is_empty() {
        return Err(MaterialDefect::Label);
    }
    let positive = [
        ("mass_density", material.mass_density.si()),
        ("youngs_modulus", material.youngs_modulus.si()),
        ("expansion_coefficient", material.expansion_coefficient.si()),
        ("conductivity", material.conductivity.si()),
        ("specific_heat", material.specific_heat.si()),
        ("reference_temperature", material.property_temperature.si()),
    ];
    for (name, value) in positive {
        if !(value.is_finite() && value > 0.0) {
            return Err(MaterialDefect::Range(name));
        }
    }
    let nu = material.poisson_ratio.si();
    if !(nu > 0.0 && nu < 0.5) {
        return Err(MaterialDefect::Range("poisson_ratio"));
    }
    // Stiffness falls as metals warm; a coefficient above zero or a loss of more than 0.5 % per
    // kelvin is a units slip.
    let coefficient = material.modulus_temperature_coefficient.si();
    if !(coefficient.is_finite() && (-5.0e-3..=0.0).contains(&coefficient)) {
        return Err(MaterialDefect::Range("modulus_temperature_coefficient"));
    }
    if let Some(yield_strength) = &material.yield_strength {
        if !(yield_strength.si().is_finite() && yield_strength.si() > 0.0) {
            return Err(MaterialDefect::Range("yield_strength"));
        }
    }
    for (name, citation) in material.property_citations() {
        if citation.source.trim().is_empty() || !citations::ALL.iter().any(|listed| *listed == citation) {
            return Err(MaterialDefect::Citation(name));
        }
    }
    Ok(())
}
