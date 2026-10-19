use super::*;

#[test]
fn interpolation_uploads_and_execution_keep_original_cause_without_retry() {
    for (hook, trace) in [
        ("upload_x", vec!["upload_x"]),
        ("upload_query", vec!["upload_x", "upload_query"]),
        ("interp1", vec!["upload_x", "upload_query", "interp1"]),
    ] {
        check_case(
            "interp1",
            hook,
            &trace,
            vec![2, 1],
            GpuTensorStorage::Real,
            |left, _| vec![Value::GpuTensor(left.clone()), Value::Num(1.5)],
        );
    }
    // A borrowed resident query remains live if the subsequent hook fails;
    // only the uploaded X intermediate is owned by this invocation.
    check_case(
        "interp1",
        "interp1",
        &["upload_x", "interp1"],
        vec![2, 1],
        GpuTensorStorage::Real,
        |left, right| {
            vec![
                Value::GpuTensor(left.clone()),
                Value::GpuTensor(right.clone()),
            ]
        },
    );
}
