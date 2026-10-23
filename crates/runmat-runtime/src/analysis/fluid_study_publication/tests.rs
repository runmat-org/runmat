use super::*;

#[test]
fn publication_cannot_substitute_precision_or_independent_numerical_controls() {
    let original = AnalysisCfdRunOptions::default();
    let validate = |options: &AnalysisCfdRunOptions| {
        validate_options(options, original.numerical, original.geometry)
    };
    assert!(validate(&original).is_ok());
    let mut changed = original.clone();
    changed.precision_mode = crate::analysis::PrecisionMode::Fp32;
    assert!(validate(&changed).is_err());
    changed = original.clone();
    changed.numerical.linear_relative_tolerance *= 0.5;
    assert!(validate(&changed).is_err());
    changed = original.clone();
    changed.numerical.physical_relative_tolerance *= 0.5;
    assert!(validate(&changed).is_err());
    changed = original.clone();
    changed.geometry.max_face_incidences = Some(1);
    assert!(validate(&changed).is_err());
}
