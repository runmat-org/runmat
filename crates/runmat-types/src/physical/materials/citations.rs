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

/// Density of type 304 (UNS S30400), from the producer's datasheet (read 2026-11-14).
pub const ATI_304_DATASHEET: Citation = cite("ati-304", "ATI 304 Austenitic Stainless Steel (UNS S30400) datasheet, ATI, atimaterials.com/Products/ati-304: density 0.285 lb/in3 (7.90 g/cm3)");
/// Density of annealed copper: the International Annealed Copper Standard's density (read
/// 2026-11-14 on NIST's site, nvlpubs.nist.gov/nistpubs/Legacy/circ/nbscircular31e4.pdf).
pub const NBS_CIRCULAR_31_COPPER_DENSITY: Citation = cite("nbs-circ31-1.1", "NBS Circular 31, 4th ed. (1956), Copper Wire Tables, 1.1 Standard Values for Copper, pp. 2-3: the density of the International Annealed Copper Standard, 8.89 g/cm3 at 20 C (IEC)");

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
    &ATI_304_DATASHEET,
    &NBS_CIRCULAR_31_COPPER_DENSITY,
];
