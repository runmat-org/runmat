use super::*;
use crate::results::observation::tests::{scalar_definition, spec};

#[test]
fn original_declaration_binding_borrows_and_exact_inventory_admission_precedes_allocation() {
    let definition = scalar_definition();
    let fields = [definition];
    let request = spec("borrowed", ObservationTarget::WholeModel);
    let bound = bind_field(&request, fields.iter()).unwrap();
    assert!(std::ptr::eq(bound, &fields[0]));
    let bytes = declaration_inventory_resident_bytes(fields.len()).unwrap();
    let (_, _, peak) = crate::topology::measure_test_allocations(|| {
        validate_field_catalog(fields.iter(), bytes).unwrap();
    });
    assert_eq!(peak as u64, bytes);
    let (_, _, peak) = crate::topology::measure_test_allocations(|| {
        assert!(matches!(
            validate_field_catalog(fields.iter(), bytes - 1),
            Err(ObservationError::Limit(
                "recovered-field declaration inventory"
            ))
        ));
    });
    assert_eq!(peak, 0);
    let duplicates = [&fields[0], &fields[0]];
    assert!(matches!(
        validate_field_catalog(duplicates.into_iter(), 2 * bytes),
        Err(ObservationError::Invalid(_))
    ));
    assert!(matches!(
        bind_field(&request, duplicates.into_iter()),
        Err(ObservationError::Field { .. })
    ));
    validate_field_catalog(std::iter::empty(), 0).unwrap();
    assert!(declaration_inventory_resident_bytes(usize::MAX).is_err());
}
