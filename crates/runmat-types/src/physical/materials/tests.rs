use super::*;

#[test]
fn every_row_passes_validation_with_a_unique_id() {
    let mut seen = std::collections::HashSet::new();
    for material in materials() {
        assert!(seen.insert(material.id), "{}", material.id);
        assert_eq!(validate_material(material), Ok(()), "{}", material.id);
    }
    // A choice card offers at most eight choices, one of them "not in the library".
    assert!(material_ids().count() < 8);
    let mut ids = std::collections::HashSet::new();
    assert!(citations::ALL
        .iter()
        .all(|citation| ids.insert(citation.id)));
}

#[test]
fn a_bad_row_is_refused() {
    let mut row = *material("steel_s275").unwrap();
    row.poisson_ratio = Cited::new(0.6, &citations::EN_1993_1_1_3_2_6);
    assert_eq!(
        validate_material(&row),
        Err(MaterialDefect::Range("poisson_ratio"))
    );
    static TEST_TABLE: Citation = Citation {
        id: "test-table",
        source: "a test table",
    };
    let mut row = *material("steel_s275").unwrap();
    row.modulus_temperature_coefficient = Some(Cited::new(3.3e-4, &TEST_TABLE));
    assert_eq!(
        validate_material(&row),
        Err(MaterialDefect::Range("modulus_temperature_coefficient"))
    );
    static UNLISTED: Citation = Citation {
        id: "x",
        source: "somewhere",
    };
    let mut row = *material("steel_s275").unwrap();
    row.conductivity = Cited::new(45.0, &UNLISTED);
    assert_eq!(
        validate_material(&row),
        Err(MaterialDefect::Citation("conductivity"))
    );
    let mut row = *material("steel_s275").unwrap();
    row.id = "Steel S275";
    assert_eq!(validate_material(&row), Err(MaterialDefect::Id));
}

/// A thermal preload study's material block needs conductivity, specific heat, expansion and the
/// temperature they're given at from every library material. No row holds a modulus temperature
/// coefficient yet (none read from ASME BPVC II-D or MMPDS for its exact grade), so the block
/// leaves it out and the study's check asks the person for it.
#[test]
fn every_material_carries_what_a_thermal_preload_study_needs() {
    for material in materials() {
        let names: Vec<&str> = material
            .fea_properties()
            .into_iter()
            .map(|(name, _)| name)
            .collect();
        for needed in [
            "mass_density",
            "youngs_modulus",
            "poisson_ratio",
            "reference_temperature",
            "conductivity",
            "specific_heat",
            "expansion_coefficient",
        ] {
            assert!(names.contains(&needed), "{} lacks {needed}", material.id);
        }
        assert!(
            material.modulus_temperature_coefficient.is_none(),
            "{}",
            material.id
        );
        assert!(
            !names.contains(&"modulus_temperature_coefficient"),
            "{}",
            material.id
        );
    }
}

#[test]
fn fea_properties_write_the_library_values_in_block_units() {
    let steel = material("steel_s275").unwrap();
    let properties = steel.fea_properties();
    assert!(properties.contains(&("youngs_modulus", "210 GPa".to_owned())));
    assert!(properties.contains(&("expansion_coefficient", "0.000012 K^-1".to_owned())));
    assert!(properties.contains(&("mass_density", "7850 kg/m^3".to_owned())));
    assert!(properties.contains(&("reference_temperature", "293.15 K".to_owned())));
    assert_eq!(
        material("aluminum_6061_t6").unwrap().fea_properties()[1].1,
        "68.9 GPa"
    );
    assert!(material("unobtainium").is_none());
    assert_eq!(steel.youngs_modulus.citation, &citations::EN_1993_1_1_3_2_6);
    assert!(
        steel
            .property_citations()
            .iter()
            .any(|(name, citation)| *name == "yield_strength"
                && citation.id == "en1993-1-1-table-3.1")
    );
}

/// Each row names one exact grade, and the 304 and C11000 densities come from the sources read for
/// them: the ATI 304 datasheet (7.90 g/cm3) and the International Annealed Copper Standard density
/// in NBS Circular 31 (8.89 g/cm3 at 20 C). runmat-cad's part materials still differ on these two
/// (stainless 7950, copper 8940) and fold in later.
#[test]
fn rows_name_exact_grades_with_their_density_sources() {
    let stainless = material("stainless_304").unwrap();
    assert!(stainless.label.contains("UNS S30400"));
    assert_eq!(stainless.mass_density.si(), 7900.0);
    assert_eq!(
        stainless.mass_density.citation,
        &citations::ATI_304_DATASHEET
    );
    let copper = material("copper_c11000").unwrap();
    assert!(copper.label.contains("C11000"));
    assert_eq!(copper.mass_density.si(), 8890.0);
    assert_eq!(
        copper.mass_density.citation,
        &citations::NBS_CIRCULAR_31_COPPER_DENSITY
    );
    assert!(citations::ALL
        .iter()
        .all(|citation| !citation.source.starts_with("Pending:")));
    assert!(material("copper_c110").is_none());
}

#[test]
fn names_find_one_material() {
    assert_eq!(
        material_named("Aluminum 6061").map(|m| m.id),
        Some("aluminum_6061_t6")
    );
    assert_eq!(
        material_named("steel_s355").map(|m| m.id),
        Some("steel_s355")
    );
    assert_eq!(material_named("Structural steel").map(|m| m.id), None);
    assert_eq!(material_named("unobtainium"), None);
}
