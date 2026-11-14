//! Metals: structural steels, aluminum, stainless steel, titanium and copper.
use super::super::citations::{self as c, secant_coefficient, Citation};
use super::super::{Cited, Material, MaterialCategory};

const GPA: f64 = 1.0e9;
const MPA: f64 = 1.0e6;
const ROOM: Cited<crate::physical::kind::Temperature> = Cited::new(293.15, &c::ROOM_TEMPERATURE_20_C);

const fn at(value: f64, citation: &'static Citation) -> Cited<crate::physical::kind::InverseTemperature> {
    Cited::new(value, citation)
}

/// The secant 70-400 F coefficients of each ASME B31.1 row ([`c::ASME_B31_1_MODULUS_CARBON_STEEL`] and the rest).
pub const CARBON_STEEL_MODULUS_COEFFICIENT: f64 = secant_coefficient(29.5, 27.7, 70.0, 400.0);
pub const AUSTENITIC_MODULUS_COEFFICIENT: f64 = secant_coefficient(28.3, 26.5, 70.0, 400.0);
pub const ALUMINUM_MODULUS_COEFFICIENT: f64 = secant_coefficient(10.0, 8.7, 70.0, 400.0);
pub const TITANIUM_MODULUS_COEFFICIENT: f64 = secant_coefficient(14.0, 11.9, 70.0, 400.0);
pub const COPPER_MODULUS_COEFFICIENT: f64 = secant_coefficient(16.0, 15.0, 70.0, 400.0);

const fn structural_steel(id: &'static str, label: &'static str, yield_mpa: f64) -> Material {
    Material {
        id,
        label,
        category: MaterialCategory::Metal,
        mass_density: Cited::new(7850.0, &c::EN_1991_1_1_TABLE_A_4),
        youngs_modulus: Cited::new(210.0 * GPA, &c::EN_1993_1_1_3_2_6),
        poisson_ratio: Cited::new(0.3, &c::EN_1993_1_1_3_2_6),
        expansion_coefficient: Cited::new(12.0e-6, &c::EN_1993_1_1_3_2_6),
        conductivity: Cited::new(45.0, &c::EN_1993_1_2_3_4_1),
        specific_heat: Cited::new(600.0, &c::EN_1993_1_2_3_4_1),
        property_temperature: ROOM,
        modulus_temperature_coefficient: at(CARBON_STEEL_MODULUS_COEFFICIENT, &c::ASME_B31_1_MODULUS_CARBON_STEEL),
        yield_strength: Some(Cited::new(yield_mpa * MPA, &c::EN_1993_1_1_TABLE_3_1)),
    }
}

pub const MATERIALS: &[Material] = &[
    structural_steel("steel_s275", "Structural steel S275", 275.0),
    structural_steel("steel_s355", "Structural steel S355", 355.0),
    Material {
        id: "aluminum_6061_t6",
        label: "Aluminum 6061-T6",
        category: MaterialCategory::Metal,
        mass_density: Cited::new(2700.0, &c::ASM_HANDBOOK_VOL_2),
        youngs_modulus: Cited::new(68.9 * GPA, &c::ASM_HANDBOOK_VOL_2),
        poisson_ratio: Cited::new(0.33, &c::ASM_HANDBOOK_VOL_2),
        expansion_coefficient: Cited::new(23.6e-6, &c::ASM_HANDBOOK_VOL_2),
        conductivity: Cited::new(167.0, &c::ASM_HANDBOOK_VOL_2),
        specific_heat: Cited::new(896.0, &c::ASM_HANDBOOK_VOL_2),
        property_temperature: ROOM,
        modulus_temperature_coefficient: at(ALUMINUM_MODULUS_COEFFICIENT, &c::ASME_B31_1_MODULUS_ALUMINUM),
        yield_strength: Some(Cited::new(276.0 * MPA, &c::ASM_HANDBOOK_VOL_2)),
    },
    Material {
        id: "stainless_304",
        label: "Stainless steel 304 (annealed)",
        category: MaterialCategory::Metal,
        mass_density: Cited::new(8000.0, &c::ASM_HANDBOOK_VOL_1),
        youngs_modulus: Cited::new(193.0 * GPA, &c::ASM_HANDBOOK_VOL_1),
        poisson_ratio: Cited::new(0.29, &c::ASM_HANDBOOK_VOL_1),
        expansion_coefficient: Cited::new(17.3e-6, &c::ASM_HANDBOOK_VOL_1),
        conductivity: Cited::new(16.2, &c::ASM_HANDBOOK_VOL_1),
        specific_heat: Cited::new(500.0, &c::ASM_HANDBOOK_VOL_1),
        property_temperature: ROOM,
        modulus_temperature_coefficient: at(AUSTENITIC_MODULUS_COEFFICIENT, &c::ASME_B31_1_MODULUS_AUSTENITIC),
        yield_strength: Some(Cited::new(205.0 * MPA, &c::ASTM_A240)),
    },
    Material {
        id: "titanium_ti6al4v",
        label: "Titanium Ti-6Al-4V (annealed)",
        category: MaterialCategory::Metal,
        mass_density: Cited::new(4430.0, &c::ASM_HANDBOOK_VOL_2),
        youngs_modulus: Cited::new(113.8 * GPA, &c::ASM_HANDBOOK_VOL_2),
        poisson_ratio: Cited::new(0.342, &c::ASM_HANDBOOK_VOL_2),
        expansion_coefficient: Cited::new(8.6e-6, &c::ASM_HANDBOOK_VOL_2),
        conductivity: Cited::new(6.7, &c::ASM_HANDBOOK_VOL_2),
        specific_heat: Cited::new(526.3, &c::ASM_HANDBOOK_VOL_2),
        property_temperature: ROOM,
        modulus_temperature_coefficient: at(TITANIUM_MODULUS_COEFFICIENT, &c::ASME_B31_1_MODULUS_TITANIUM),
        yield_strength: Some(Cited::new(880.0 * MPA, &c::ASM_HANDBOOK_VOL_2)),
    },
    Material {
        id: "copper_c110",
        label: "Copper C110 (annealed)",
        category: MaterialCategory::Metal,
        mass_density: Cited::new(8890.0, &c::ASM_HANDBOOK_VOL_2),
        youngs_modulus: Cited::new(115.0 * GPA, &c::ASM_HANDBOOK_VOL_2),
        poisson_ratio: Cited::new(0.31, &c::ASM_HANDBOOK_VOL_2),
        expansion_coefficient: Cited::new(17.0e-6, &c::ASM_HANDBOOK_VOL_2),
        conductivity: Cited::new(388.0, &c::ASM_HANDBOOK_VOL_2),
        specific_heat: Cited::new(385.0, &c::ASM_HANDBOOK_VOL_2),
        property_temperature: ROOM,
        modulus_temperature_coefficient: at(COPPER_MODULUS_COEFFICIENT, &c::ASME_B31_1_MODULUS_COPPER),
        yield_strength: None,
    },
];
