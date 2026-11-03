use super::*;

#[test]
fn counts_keep_exact_integer_units_and_checked_arithmetic() {
    let large = OperationCount::from_operations((1 << 53) + 1);
    assert_eq!(large.operations(), 9_007_199_254_740_993);
    assert_eq!(OperationCount::PER_GFLOP, 1_000_000_000);
    assert_eq!(large.checked_mul(0), Ok(OperationCount::ZERO));
    let max = OperationCount::from_operations(u64::MAX);
    assert_eq!(
        max.checked_add(OperationCount::from_operations(1)),
        Err(CountOverflow)
    );
    assert_eq!(max.checked_mul(2), Err(CountOverflow));
}

#[test]
fn direct_dot_schedule_handles_empty_and_full_range() {
    for (length, expected) in [(0, 0), (1, 1), (4, 7), (1 << 63, u64::MAX)] {
        assert_eq!(
            OperationCount::direct_real_dot(length),
            Ok(OperationCount::from_operations(expected))
        );
    }
    assert_eq!(
        OperationCount::direct_real_dot((1 << 63) + 1),
        Err(CountOverflow)
    );
    assert_eq!(
        RealArithmetic::FusedMultiplyAdd.count(4),
        Ok(OperationCount::from_operations(8))
    );
    for operation in [
        RealArithmetic::Add,
        RealArithmetic::Subtract,
        RealArithmetic::Multiply,
        RealArithmetic::Divide,
    ] {
        assert_eq!(operation.count(4), Ok(OperationCount::from_operations(4)));
    }
}

#[test]
fn wire_count_round_trips_without_javascript_number_conversion() {
    for operations in [0, 1, (1 << 53) + 1, u64::MAX] {
        let count = OperationCount::from_operations(operations);
        let encoded = serde_json::to_string(&count).unwrap();
        assert_eq!(encoded, format!("\"{operations}\""));
        assert_eq!(
            serde_json::from_str::<OperationCount>(&encoded).unwrap(),
            count
        );
    }
    for invalid in [
        "0",
        "1.0",
        "null",
        "\"\"",
        "\"01\"",
        "\"+1\"",
        "\"-1\"",
        "\"1.0\"",
        "\" 1\"",
        "\"18446744073709551616\"",
    ] {
        assert!(
            serde_json::from_str::<OperationCount>(invalid).is_err(),
            "{invalid}"
        );
    }
}
