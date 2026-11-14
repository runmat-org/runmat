//! The sources the library's values come from. A property names one of these; a row with no
//! source isn't admitted (`validate.rs`).

/// One source: a standard's clause or table, or a handbook's table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Citation {
    /// Stable id, used in tests and the docs.
    pub id: &'static str,
    /// The document and the clause or table the value comes from.
    pub source: &'static str,
}

const fn cite(id: &'static str, source: &'static str) -> Citation {
    Citation { id, source }
}

/// E, nu and the expansion coefficient of structural steel.
pub const EN_1993_1_1_3_2_6: Citation = cite("en1993-1-1-3.2.6", "EN 1993-1-1, 3.2.6: design values of material coefficients for structural steel");
/// Yield strength of hot-rolled structural steel, t <= 40 mm.
pub const EN_1993_1_1_TABLE_3_1: Citation = cite("en1993-1-1-table-3.1", "EN 1993-1-1, Table 3.1: nominal yield strength for hot-rolled structural steel (t <= 40 mm)");
/// Density of steel.
pub const EN_1991_1_1_TABLE_A_4: Citation = cite("en1991-1-1-table-a.4", "EN 1991-1-1, Table A.4: nominal density of construction materials (metals)");
/// The constant conductivity and specific heat EN 1993-1-2 allows for simple calculations.
pub const EN_1993_1_2_3_4_1: Citation = cite("en1993-1-2-3.4.1", "EN 1993-1-2, 3.4.1: thermal properties of carbon steel for simple calculation models");
/// The temperature room-temperature handbook and standard values are given at.
pub const ROOM_TEMPERATURE_20_C: Citation = cite("room-temperature-20c", "Room-temperature values of the cited sources, taken at 20 C (293.15 K)");
/// Typical room-temperature properties of wrought aluminum, titanium and copper alloys.
pub const ASM_HANDBOOK_VOL_2: Citation = cite("asm-vol2", "ASM Handbook Vol. 2, Properties and Selection: Nonferrous Alloys and Special-Purpose Materials, typical room-temperature properties");
/// Typical properties of annealed austenitic stainless steel; expansion over 0-100 C.
pub const ASM_HANDBOOK_VOL_1: Citation = cite("asm-vol1", "ASM Handbook Vol. 1, Properties and Selection: Irons, Steels, and High-Performance Alloys, typical properties of annealed type 304");
/// Minimum yield strength of annealed 304 plate, sheet and strip.
pub const ASTM_A240: Citation = cite("astm-a240", "ASTM A240/A240M, Table 2: minimum yield strength of type 304");

/// Modulus of elasticity against temperature in ASME B31.1 (1995 edition), as tabulated by
/// The Engineering ToolBox ("Young's Modulus of Elasticity for Metals and Alloys"). The library's
/// modulus temperature coefficients are the secant of that table from 70 F (21 C) to 400 F
/// (204 C): `(E_400F / E_70F - 1) / 183.3 K`.
pub const ASME_B31_1_MODULUS_CARBON_STEEL: Citation = cite("asme-b31.1-e-carbon-steel", "ASME B31.1-1995 modulus of elasticity table, carbon steel C <= 0.3 %: 29.5 and 27.7 x 10^6 psi at 70 F and 400 F (secant 70-400 F)");
pub const ASME_B31_1_MODULUS_AUSTENITIC: Citation = cite("asme-b31.1-e-austenitic", "ASME B31.1-1995 modulus of elasticity table, austenitic steels (TP304 and others): 28.3 and 26.5 x 10^6 psi at 70 F and 400 F (secant 70-400 F)");
pub const ASME_B31_1_MODULUS_ALUMINUM: Citation = cite("asme-b31.1-e-aluminum", "ASME B31.1-1995 modulus of elasticity table, aluminum grades (6063 and others; 6061 is not listed): 10.0 and 8.7 x 10^6 psi at 70 F and 400 F (secant 70-400 F)");
pub const ASME_B31_1_MODULUS_TITANIUM: Citation = cite("asme-b31.1-e-titanium", "ASME B31.1-1995 modulus of elasticity table, unalloyed titanium grades 1, 2, 3 and 7 (Ti-6Al-4V is not listed): 14.0 and 11.9 x 10^6 psi at 70 F and 400 F (secant 70-400 F)");
pub const ASME_B31_1_MODULUS_COPPER: Citation = cite("asme-b31.1-e-copper", "ASME B31.1-1995 modulus of elasticity table, copper C11000: 16.0 and 15.0 x 10^6 psi at 70 F and 400 F (secant 70-400 F)");

/// Every citation the library uses, for the docs and the uniqueness check.
pub const ALL: &[&Citation] = &[
    &EN_1993_1_1_3_2_6,
    &EN_1993_1_1_TABLE_3_1,
    &EN_1991_1_1_TABLE_A_4,
    &EN_1993_1_2_3_4_1,
    &ROOM_TEMPERATURE_20_C,
    &ASM_HANDBOOK_VOL_2,
    &ASM_HANDBOOK_VOL_1,
    &ASTM_A240,
    &ASME_B31_1_MODULUS_CARBON_STEEL,
    &ASME_B31_1_MODULUS_AUSTENITIC,
    &ASME_B31_1_MODULUS_ALUMINUM,
    &ASME_B31_1_MODULUS_TITANIUM,
    &ASME_B31_1_MODULUS_COPPER,
];

/// The secant coefficient `(e_hot / e_cold - 1) / (t_hot - t_cold)` per kelvin, from two table
/// values in any one unit and their temperatures in Fahrenheit.
pub const fn secant_coefficient(e_cold: f64, e_hot: f64, t_cold_f: f64, t_hot_f: f64) -> f64 {
    (e_hot / e_cold - 1.0) / ((t_hot_f - t_cold_f) * 5.0 / 9.0)
}
