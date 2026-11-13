use super::*;
use crate::foreign::{foreign_error_with_source, ForeignErrorKind};
use std::error::Error;

#[test]
fn missing_prepared_library_retains_real_path_and_os_error_through_wire() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("missing-prepared-library.so");
    let original_io = std::fs::read(&path).unwrap_err();
    let raw_os_error = original_io.raw_os_error().expect("actual file OS error");
    let kind = original_io.kind();
    let original = foreign_error_with_source(
        ForeignErrorKind::LoadFailed,
        PreparedLibraryReadError {
            path: path.clone(),
            source: original_io,
        },
    );
    let wire = NativeWireError::from_runtime(original);
    wire.validate().unwrap();
    let encoded = serde_json::to_vec(&wire).unwrap();
    let decoded: NativeWireError = serde_json::from_slice(&encoded).unwrap();
    decoded.validate().unwrap();
    let restored = decoded.into_runtime();
    assert_eq!(restored.identifier(), Some("RunMat:Foreign:LoadFailed"));
    let cause = restored.source().expect("path-qualified cause");
    let typed = cause.downcast_ref::<PreparedLibraryReadError>().unwrap();
    assert_eq!(typed.path, path);
    assert!(cause.to_string().contains("missing-prepared-library.so"));
    let io = cause
        .source()
        .unwrap()
        .downcast_ref::<std::io::Error>()
        .unwrap();
    assert_eq!(io.kind(), kind);
    assert_eq!(io.raw_os_error(), Some(raw_os_error));
    // A public identifier or matching spelling is never cause authority.
    let unrelated = crate::build_runtime_error("missing-prepared-library.so")
        .with_identifier("RunMat:Foreign:LoadFailed")
        .build();
    assert!(NativeWireError::from_runtime(unrelated).cause.is_none());
}
