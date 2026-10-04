use super::*;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, GpuTensorStorage, NumericElementType,
};

#[test]
fn read_only_file_rejects_resident_payload_before_download() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fwrite_read_only_preflight");
    test_support::fs::write(&path, "unchanged").expect("fixture");
    let opened = run_fopen(&[
        Value::from(path.to_string_lossy().to_string()),
        Value::from("r"),
    ])
    .expect("read-only fopen");
    let fid = opened.as_open().expect("file form").fid;
    let resident = TestAllocationOwner::new()
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .expect("resident data");
    let _no_provider = runmat_accelerate_api::ThreadProviderGuard::set(None);
    let error = run_evaluate(
        &Value::Num(fid),
        &Value::GpuTensor(resident.handle.clone()),
        &[],
    )
    .expect_err("read-only target precedes resident payload");
    assert_eq!(error.identifier(), Some("RunMat:fwrite:InvalidIdentifier"));
    assert_eq!(resident.probe.retirements(), 0);
    assert_eq!(
        test_support::fs::read_to_string(&path).unwrap(),
        "unchanged"
    );
    run_fclose(&[Value::Num(fid)]).expect("close");
    test_support::fs::remove_file(path).expect("cleanup");
}
