use super::*;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, GpuTensorStorage, NumericElementType,
};

#[test]
fn write_only_file_rejects_resident_size_before_download() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fread_write_only_preflight");
    let opened = run_fopen(&[
        Value::from(path.to_string_lossy().to_string()),
        Value::from("w"),
    ])
    .expect("write-only fopen");
    let fid = opened.as_open().expect("file form").fid;
    let resident = TestAllocationOwner::new()
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .expect("resident size");
    let _no_provider = runmat_accelerate_api::ThreadProviderGuard::set(None);
    let error = run_evaluate(
        &Value::Num(fid),
        &[Value::GpuTensor(resident.handle.clone())],
    )
    .expect_err("write-only target precedes resident size");
    assert_eq!(error.identifier(), Some("RunMat:fread:InvalidIdentifier"));
    assert_eq!(resident.probe.retirements(), 0);
    run_fclose(&[Value::Num(fid)]).expect("close");
    test_support::fs::remove_file(path).expect("cleanup");
}
