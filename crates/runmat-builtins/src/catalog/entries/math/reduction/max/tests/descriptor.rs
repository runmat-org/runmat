use super::*;

#[test]
fn descriptor_preserves_reduction_pairwise_output_and_error_contracts() {
    let labels: Vec<_> = MAX_DESCRIPTOR
        .signatures
        .iter()
        .map(|signature| signature.label)
        .collect();
    for form in [
        "M = max(A)",
        "[M, I] = max(A)",
        "M = max(A, B)",
        "M = max(A, B, nanflag)",
        "M = max(A, [], dim)",
        "[M, I] = max(A, [], vecdim)",
        "M = max(A, [], \"all\")",
        "[M, I] = max(A, [], \"linear\")",
        "M = max(A, [], nanflag)",
        "M = max(A, [], \"ComparisonMethod\", method)",
        "M = max(A, B, \"ComparisonMethod\", method)",
    ] {
        assert!(labels.contains(&form), "missing {form}");
    }
    assert!(!labels.contains(&"[M, I] = max(A, B)"));
    assert_eq!(MAX_DESCRIPTOR.errors.len(), 6);
    for code in [
        MAX_ERROR_INVALID_ARGUMENT.code,
        MAX_ERROR_INVALID_INPUT.code,
        MAX_ERROR_SIZE_MISMATCH.code,
        MAX_ERROR_INTERNAL.code,
        TABLE_ERROR_INVALID_ARGUMENT.code,
        TABLE_ERROR_INVALID_VARIABLE.code,
    ] {
        assert!(MAX_DESCRIPTOR.errors.iter().any(|error| error.code == code));
    }
    assert_eq!(MAX_INTEGER_CAPABILITIES.len(), 2);
}

#[test]
fn value_output_descriptor_admits_the_existing_ordinal_categorical_route() {
    for signature in MAX_DESCRIPTOR.signatures {
        assert_eq!(signature.outputs[0].ty, BuiltinParamType::Any);
        if signature.outputs.len() == 2 {
            assert_eq!(signature.outputs[1].ty, BuiltinParamType::NumericArray);
        }
    }
}
