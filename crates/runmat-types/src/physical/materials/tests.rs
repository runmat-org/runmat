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
    assert!(citations::ALL.iter().all(|citation| ids.insert(citation.id)));
}

#[test]
fn a_bad_row_is_refused() {
    let mut row = *material("steel_s275").unwrap();
    row.poisson_ratio = Cited::new(0.6, &citations::EN_1993_1_1_3_2_6);
    assert_eq!(validate_material(&row), Err(MaterialDefect::Range("poisson_ratio")));
    let mut row = *material("steel_s275").unwrap();
    row.modulus_temperature_coefficient = Cited::new(3.3e-4, &citations::ASME_B31_1_MODULUS_CARBON_STEEL);
    assert_eq!(validate_material(&row), Err(MaterialDefect::Range("modulus_temperature_coefficient")));
    static UNLISTED: Citation = Citation { id: "x", source: "somewhere" };
    let mut row = *material("steel_s275").unwrap();
    row.conductivity = Cited::new(45.0, &UNLISTED);
    assert_eq!(validate_material(&row), Err(MaterialDefect::Citation("conductivity")));
    let mut row = *material("steel_s275").unwrap();
    row.id = "Steel S275";
    assert_eq!(validate_material(&row), Err(MaterialDefect::Id));
}

/// A thermal preload study's material block needs the thermal block's five values on every
/// library material: conductivity, specific heat, expansion, the modulus temperature
/// coefficient and the temperature they're given at.
#[test]
fn every_material_carries_what_a_thermal_preload_study_needs() {
    for material in materials() {
        let names: Vec<&str> = material.fea_properties().into_iter().map(|(name, _)| name).collect();
        for needed in ["mass_density", "youngs_modulus", "poisson_ratio", "reference_temperature",
            "modulus_temperature_coefficient", "conductivity", "specific_heat", "expansion_coefficient"] {
            assert!(names.contains(&needed), "{} lacks {needed}", material.id);
        }
        assert!(material.modulus_temperature_coefficient.si() < 0.0, "{}", material.id);
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
    // ASME B31.1 carbon steel: 29.5 to 27.7 x 10^6 psi from 70 F to 400 F.
    assert!(properties.contains(&("modulus_temperature_coefficient", "-0.00033282 K^-1".to_owned())), "{properties:?}");
    assert_eq!(material("aluminum_6061_t6").unwrap().fea_properties()[1].1, "68.9 GPa");
    assert!(material("unobtainium").is_none());
    assert_eq!(steel.youngs_modulus.citation, &citations::EN_1993_1_1_3_2_6);
    assert!(steel.property_citations().iter().any(|(name, citation)| *name == "yield_strength" && citation.id == "en1993-1-1-table-3.1"));
}

/// The library keeps its cited densities; runmat-cad's part materials still differ on two
/// (stainless 7950, copper 8940), which FEA reviews before the MCAD list folds in.
#[test]
fn densities_are_the_cited_ones() {
    assert_eq!(material("stainless_304").unwrap().mass_density.si(), 8000.0);
    assert_eq!(material("copper_c110").unwrap().mass_density.si(), 8890.0);
}
