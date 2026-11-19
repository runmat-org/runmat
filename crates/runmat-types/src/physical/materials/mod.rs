//! The shared material library: engineering materials with an id, a display label and the
//! nominal properties a study, a part's mass and a board's package bodies read. Every property
//! is a typed [`Quantity`] in SI canonical units with the [`Citation`] it comes from.
//!
//! [`Material::fea_properties`] writes a material in the `.fea` material block's property names
//! and units, so a study that picks a library material declares the numbers the library holds.
//! A material the library doesn't hold is described by its properties, never by a name alone.
//!
//! Layout: this file holds the record and the lookup; [`catalog`] holds the materials by
//! category, [`citations`] the sources, and `validate.rs` the checks every row passes.

pub mod catalog;
pub mod citations;
mod validate;
#[cfg(test)]
mod tests;

pub use citations::Citation;
pub use validate::{validate_material, MaterialDefect};

use super::{kind, Quantity, QuantityMarker};

/// A property value and the source it comes from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cited<K: QuantityMarker> {
    pub value: Quantity<K>,
    pub citation: &'static Citation,
}

impl<K: QuantityMarker> Cited<K> {
    pub const fn new(value: f64, citation: &'static Citation) -> Self {
        Self {
            value: Quantity::new(value),
            citation,
        }
    }

    /// The value in the kind's SI canonical unit.
    pub const fn si(&self) -> f64 {
        self.value.canonical_value()
    }
}

/// The library's material groups, one catalog file each.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MaterialCategory {
    Metal,
    Polymer,
    Composite,
    Ceramic,
    Electronic,
    Construction,
    Fluid,
}

/// One library material, one exact grade. Thermal properties are given at
/// [`Material::property_temperature`]; [`Material::modulus_temperature_coefficient`] is the
/// relative change of Young's modulus per kelvin from there, `E(T) = E (1 + c (T - T_ref))`,
/// which a thermal preload study reads.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Material {
    /// Stable id, also the `.fea` material declaration name (`material steel_s275 { ... }`).
    pub id: &'static str,
    /// The name people read on a card or in a parts list.
    pub label: &'static str,
    pub category: MaterialCategory,
    pub mass_density: Cited<kind::MassDensity>,
    pub youngs_modulus: Cited<kind::YoungsModulus>,
    pub poisson_ratio: Cited<kind::PoissonRatio>,
    /// Linear expansion coefficient.
    pub expansion_coefficient: Cited<kind::ThermalExpansion>,
    pub conductivity: Cited<kind::ThermalConductivity>,
    pub specific_heat: Cited<kind::SpecificHeat>,
    /// The temperature the properties are given at: the `.fea` thermal block's
    /// `reference_temperature`.
    pub property_temperature: Cited<kind::Temperature>,
    /// Only from a primary source for this exact grade (ASME BPVC Section II Part D's TM tables,
    /// or MMPDS); `None` until one is read and cited. A study then asks the person for it.
    pub modulus_temperature_coefficient: Option<Cited<kind::InverseTemperature>>,
    /// Minimum or typical yield strength, when the source gives one for this condition.
    pub yield_strength: Option<Cited<kind::YieldStress>>,
}

/// The library, by category, in the order a card offers them.
pub fn materials() -> impl Iterator<Item = &'static Material> {
    catalog::CATEGORIES.iter().flat_map(|(_, rows)| rows.iter())
}

/// The library material with `id`.
pub fn material(id: &str) -> Option<&'static Material> {
    materials().find(|material| material.id == id)
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
    if let Some(found) = materials().find(|m| m.id == name || m.label.eq_ignore_ascii_case(name)) {
        return Some(found);
    }
    let wanted = words(name);
    if wanted.is_empty() {
        return None;
    }
    let mut matches = materials().filter(|m| {
        let label = words(m.label);
        wanted.iter().all(|w| label.contains(w))
    });
    let first = matches.next()?;
    matches.next().is_none().then_some(first)
}

/// The library's material ids, in card order.
pub fn material_ids() -> impl Iterator<Item = &'static str> {
    materials().map(|material| material.id)
}

const GPA: f64 = 1.0e9;

impl Material {
    /// The properties as a `.fea` material block names them, each with its value and unit:
    /// `("youngs_modulus", "210 GPa")`. The thermal block's `reference_temperature` is
    /// [`Material::property_temperature`]. `yield_strength` is left out: a study reads it from an
    /// acceptance criterion, not from the material block. `modulus_temperature_coefficient` is
    /// written only when the library holds a cited one.
    pub fn fea_properties(&self) -> Vec<(&'static str, String)> {
        let coefficient = self.modulus_temperature_coefficient.map(|coefficient| {
            (
                "modulus_temperature_coefficient",
                format!("{} K^-1", number(coefficient.si())),
            )
        });
        [
            (
                "mass_density",
                format!("{} kg/m^3", number(self.mass_density.si())),
            ),
            (
                "youngs_modulus",
                format!("{} GPa", number(self.youngs_modulus.si() / GPA)),
            ),
            ("poisson_ratio", number(self.poisson_ratio.si())),
            (
                "reference_temperature",
                format!("{} K", number(self.property_temperature.si())),
            ),
        ]
        .into_iter()
        .chain(coefficient)
        .chain([
            (
                "expansion_coefficient",
                format!("{} K^-1", number(self.expansion_coefficient.si())),
            ),
            (
                "conductivity",
                format!("{} W/m/K", number(self.conductivity.si())),
            ),
            (
                "specific_heat",
                format!("{} J/kg/K", number(self.specific_heat.si())),
            ),
        ])
        .collect()
    }

    /// Each property's citation, by `.fea` property name (and `yield_strength`), for the agent
    /// and the docs to show where a number comes from.
    pub fn property_citations(&self) -> Vec<(&'static str, &'static Citation)> {
        let mut cited = vec![
            ("mass_density", self.mass_density.citation),
            ("youngs_modulus", self.youngs_modulus.citation),
            ("poisson_ratio", self.poisson_ratio.citation),
            ("reference_temperature", self.property_temperature.citation),
            ("expansion_coefficient", self.expansion_coefficient.citation),
            ("conductivity", self.conductivity.citation),
            ("specific_heat", self.specific_heat.citation),
        ];
        if let Some(coefficient) = &self.modulus_temperature_coefficient {
            cited.push(("modulus_temperature_coefficient", coefficient.citation));
        }
        if let Some(yield_strength) = &self.yield_strength {
            cited.push(("yield_strength", yield_strength.citation));
        }
        cited
    }
}

/// A plain decimal with no float noise (`0.000012`, `68.9`, `7850`, `-0.00033`).
fn number(value: f64) -> String {
    let text = format!("{value:.9}");
    let text = text.trim_end_matches('0').trim_end_matches('.');
    if text == "-0" {
        "0".to_owned()
    } else {
        text.to_owned()
    }
}
