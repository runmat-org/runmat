use super::*;

#[test]
fn preserves_matching_tabular_identity() {
    let entry = builtin_catalog_entry_by_name("atan2").expect("atan2 catalog entry");
    for class in [standard::TABLE, standard::TIMETABLE] {
        let inferred = infer_catalog_call(
            entry,
            &request(vec![
                object(class),
                numeric(
                    NumericClass::Double,
                    NumericDomain::Real,
                    vec![Some(1), Some(1)],
                ),
            ]),
        );
        assert!(inferred.diagnostics.is_empty(), "{class}");
        let ValueKindFact::Object(output) = &inferred.outputs[0].kind else {
            panic!("atan2 must retain {class} identity");
        };
        assert_eq!(output.runtime_class, Some(class.owned()));
        assert!(output.properties.is_empty());
        assert!(!output.properties_complete);
    }
    let incompatible = infer_catalog_call(
        entry,
        &request(vec![object(standard::TABLE), object(standard::TIMETABLE)]),
    );
    assert!(incompatible
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-TABULAR-BINARY-PAIR"));
}
