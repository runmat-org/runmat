use std::convert::TryFrom;

use runmat_value::{IntegerStorage, Tensor, Value};

use super::support::{call, PathGuard};

fn codes() -> Value {
    Value::Tensor(
        Tensor::new_integer(
            IntegerStorage::U16("typed-path".encode_utf16().collect()),
            vec![1, 10],
        )
        .expect("numeric character-code row"),
    )
}

#[test]
fn runmat_mode_accepts_typed_numeric_character_codes() {
    let _lock = runmat_filesystem::provider_override_lock();
    let _guard = PathGuard::new();
    let _compatibility = crate::compatibility::push_runmat_extensions_enabled(true);
    call(vec![codes()]).expect("numeric path");
    assert_eq!(
        String::try_from(&call(Vec::new()).expect("query")).expect("current path"),
        "typed-path"
    );
}

#[test]
fn matlab_mode_rejects_numeric_codes_before_mutation() {
    let _lock = runmat_filesystem::provider_override_lock();
    let guard = PathGuard::new();
    let _compatibility = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = call(vec![codes()]).expect_err("numeric extension");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:PathNumericCharacterCodesExtension")
    );
    assert_eq!(
        String::try_from(&call(Vec::new()).expect("query")).expect("current path"),
        guard.previous
    );
}

#[test]
fn matlab_mode_rejects_resident_codes_before_provider_lookup() {
    let _lock = runmat_filesystem::provider_override_lock();
    let guard = PathGuard::new();
    let _compatibility = crate::compatibility::push_runmat_extensions_enabled(false);
    let allocation =
        crate::builtins::io::repl_fs::test_support::unregistered_numeric_allocation(vec![1, 1]);

    let error = call(vec![Value::GpuTensor(allocation.handle.clone())])
        .expect_err("resident numeric extension");

    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:PathNumericCharacterCodesExtension")
    );
    assert_eq!(
        String::try_from(&call(Vec::new()).expect("query")).expect("current path"),
        guard.previous
    );
    allocation
        .handle
        .allocation_lease()
        .validate_live()
        .unwrap();
    assert_eq!(allocation.probe.retirements(), 0);
}

#[test]
fn resident_numeric_path_uses_its_exact_provider_and_retains_the_source() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let _lock = runmat_filesystem::provider_override_lock();
        let _guard = PathGuard::new();
        let text = "resident-path";
        let tensor = Tensor::new(
            text.chars()
                .map(|character| u32::from(character) as f64)
                .collect(),
            vec![1, text.len()],
        )
        .expect("numeric path");
        let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor)
            .expect("resident numeric path");
        let lifetime = handle.allocation_lease().downgrade();
        let owner = crate::builtins::common::gpu_helpers::exact_provider_for_handle(&handle)
            .expect("exact resident owner");
        assert!(runmat_accelerate_api::same_provider_instance(
            owner, provider
        ));
        let before = provider.telemetry_snapshot();
        let _compatibility = crate::compatibility::push_runmat_extensions_enabled(true);

        call(vec![Value::GpuTensor(handle.clone())]).expect("resident path");
        assert_eq!(
            String::try_from(&call(Vec::new()).expect("query")).expect("current path"),
            text
        );

        let after = provider.telemetry_snapshot();
        assert!(after.download_bytes > before.download_bytes);
        assert_eq!(after.kernel_launches, before.kernel_launches);
        assert!(lifetime.acquire().is_some());
        handle.release().expect("release resident path");
        assert!(lifetime.acquire().is_none());
    });
}
