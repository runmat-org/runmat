use super::*;

fn object(namespace: ObjectNamespace, name: &str) -> LogicalObject {
    LogicalObject::new(namespace, name, "application/test", vec![1, 2, 3]).unwrap()
}

#[test]
fn empty_and_arbitrarily_ordered_names_in_distinct_namespaces_are_valid() {
    let limits = ObjectInventoryLimits::default();
    validate_inventory(&[], limits).unwrap();
    let objects = [
        object(ObjectNamespace::ResultValue, "z"),
        object(ObjectNamespace::InputValue, "a"),
        object(ObjectNamespace::ResultValue, "a"),
        object(ObjectNamespace::ResultValue, "m"),
    ];
    let before = objects.clone();
    validate_inventory(&objects, limits).unwrap();
    assert_eq!(objects, before);
}

#[test]
fn duplicate_names_are_rejected_only_within_the_same_namespace() {
    let objects = [
        object(ObjectNamespace::ResultValue, "duplicate"),
        object(ObjectNamespace::InputValue, "other"),
        object(ObjectNamespace::ResultValue, "duplicate"),
    ];
    assert!(matches!(
        validate_inventory(&objects, ObjectInventoryLimits::default()),
        Err(ArtifactError::Invalid(message))
            if message == "duplicate object logical name in one namespace"
    ));
}

#[test]
fn malformed_objects_and_inventory_limits_keep_their_error_categories() {
    let valid = object(ObjectNamespace::ResultValue, "valid");
    let mut malformed = valid.clone();
    malformed.descriptor.logical_name = "../invalid".into();
    assert!(matches!(
        validate_inventory(&[malformed], ObjectInventoryLimits::default()),
        Err(ArtifactError::Invalid(_))
    ));
    let mut stale = valid.clone();
    stale.bytes.push(4);
    assert!(matches!(
        validate_inventory(&[stale], ObjectInventoryLimits::default()),
        Err(ArtifactError::Identity(_))
    ));
    for limits in [
        ObjectInventoryLimits {
            max_objects: 0,
            ..ObjectInventoryLimits::default()
        },
        ObjectInventoryLimits {
            max_object_bytes: 2,
            ..ObjectInventoryLimits::default()
        },
        ObjectInventoryLimits {
            max_total_bytes: 2,
            ..ObjectInventoryLimits::default()
        },
    ] {
        assert!(matches!(
            validate_inventory(std::slice::from_ref(&valid), limits),
            Err(ArtifactError::Limit(_))
        ));
    }
}

#[test]
fn deferred_duplicate_check_preserves_first_error_precedence() {
    let valid = object(ObjectNamespace::ResultValue, "same");
    let mut stale = valid.clone();
    stale.bytes.push(4);
    assert!(matches!(
        validate_inventory(&[valid.clone(), valid.clone(), stale.clone()], ObjectInventoryLimits::default()),
        Err(ArtifactError::Invalid(message)) if message.contains("duplicate")
    ));
    assert!(matches!(
        validate_inventory(
            &[stale, valid.clone(), valid],
            ObjectInventoryLimits::default()
        ),
        Err(ArtifactError::Identity(_))
    ));
}

#[test]
fn scratch_slots_are_checked_and_empty_inventory_allocates_no_slots() {
    assert_eq!(inventory_validation_scratch_bytes(0).unwrap(), 0);
    assert_eq!(
        inventory_validation_scratch_bytes(7).unwrap(),
        (7 * std::mem::size_of::<InventoryName<'_>>()) as u64
    );
    assert!(matches!(
        inventory_validation_scratch_bytes(usize::MAX),
        Err(ArtifactError::Limit(_))
    ));
}

#[test]
fn admitted_inventory_follows_the_run_budget_within_platform_ceilings() {
    let budget = 14 * 1024 * 1024 * 1024_u64;
    let limits = ObjectInventoryLimits::admitted(budget);
    #[cfg(target_pointer_width = "64")]
    let expected_total = budget;
    #[cfg(not(target_pointer_width = "64"))]
    let expected_total = 1_u64 << usize::BITS;
    assert_eq!(limits.max_total_bytes, expected_total);
    assert_eq!(
        limits.max_object_bytes,
        limits
            .max_total_bytes
            .min(super::PLATFORM_MAXIMUM_OBJECT_BYTES)
    );
    // A thermal projection past the former fixed 4 GiB / 512 MiB ceilings fits a 14 GiB run.
    #[cfg(target_pointer_width = "64")]
    limits
        .require_projection(
            "projection",
            2_000,
            1024 * 1024 * 1024,
            6 * 1024 * 1024 * 1024,
        )
        .unwrap();
    let small = ObjectInventoryLimits::admitted(1024);
    assert_eq!(small.max_objects, 1024);
    assert_eq!(ObjectInventoryLimits::admitted(0).max_total_bytes, 1);
}

#[test]
fn projection_beyond_the_admitted_budget_is_a_typed_refusal_naming_both_sizes() {
    let limits = ObjectInventoryLimits::admitted(1_000);
    let error = limits
        .require_projection("canonical payload projection", 2, 100, 1_001)
        .unwrap_err();
    assert_eq!(
        error,
        super::InventoryCapacityError {
            subject: "canonical payload projection",
            resource: "output bytes",
            required: 1_001,
            maximum: 1_000,
        }
    );
    assert_eq!(
        error.to_string(),
        "canonical payload projection requires 1001 output bytes, the admitted inventory allows 1000"
    );
    let objects = limits.require_projection("p", 1_001, 1, 1_000).unwrap_err();
    assert_eq!(objects.resource, "objects");
    limits.require_projection("p", 1_000, 1_000, 1_000).unwrap();
}
