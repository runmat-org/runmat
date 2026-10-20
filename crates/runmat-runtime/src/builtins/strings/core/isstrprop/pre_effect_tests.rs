use super::*;
use crate::builtins::common::test_support;
use runmat_accelerate_api::{GpuHandleProvenance, HostIntegerDataView, HostIntegerTensorView};
use runmat_value::LogicalArray;
use std::error::Error;

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn known_invalid_controls_reject_before_resident_download_and_keep_borrowed_input() {
    test_support::with_test_provider(|provider| {
        let handle = provider
            .upload_integer(&HostIntegerTensorView {
                data: HostIntegerDataView::U16(&[u16::from(b'A'), u16::from(b'1')]),
                shape: &[1, 2],
            })
            .expect("upload exact character codes")
            .with_provenance(GpuHandleProvenance::Explicit)
            .expect("explicit resident input");
        let weak = handle.allocation_lease().downgrade();
        let descriptor = handle.wire_descriptor();
        let download_before = provider.telemetry_snapshot().download_bytes;

        {
            let _matlab = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = futures::executor::block_on(isstrprop_builtin(
                None,
                Value::GpuTensor(handle.clone()),
                Value::String("unknown-category".into()),
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::String("not-an-option".into()),
                    Value::Num(2.0),
                ]),
            ))
            .expect_err("explicit resident extension owns the strict-mode rejection");
            assert_eq!(
                error.identifier(),
                ISSTRPROP_RESIDENT_INPUT_EXTENSION.error_identifier
            );
            assert_eq!(
                provider.telemetry_snapshot().download_bytes,
                download_before
            );
        }

        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        for (category, rest, diagnostic) in [
            ("unknown-category", Vec::new(), "unknown character category"),
            (
                "alpha",
                vec![Value::String("not-an-option".into()), Value::Bool(true)],
                "unsupported option",
            ),
            (
                "alpha",
                vec![Value::String("ForceCellOutput".into()), Value::Num(2.0)],
                "ForceCellOutput must be",
            ),
        ] {
            let error = futures::executor::block_on(isstrprop_builtin(
                None,
                Value::GpuTensor(handle.clone()),
                Value::String(category.into()),
                crate::builtins::common::test_support::builtin_arguments(rest),
            ))
            .expect_err("invalid host-known control");
            assert!(error.message().contains(diagnostic));
            assert!(
                Error::source(&error).is_none(),
                "no provider work/source occurred"
            );
            assert_eq!(
                provider.telemetry_snapshot().download_bytes,
                download_before
            );
            assert!(weak.acquire().is_some());
            assert_eq!(handle.wire_descriptor(), descriptor);
        }

        let result = futures::executor::block_on(isstrprop_builtin(
            None,
            Value::GpuTensor(handle.clone()),
            Value::String("alpha".into()),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect("valid resident character-code classification");
        assert_eq!(
            result,
            Value::LogicalArray(LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap())
        );
        assert_eq!(
            provider.telemetry_snapshot().download_bytes,
            download_before + 4
        );
        assert!(weak.acquire().is_some());
        assert_eq!(handle.wire_descriptor(), descriptor);
        drop(handle);
        assert!(weak.acquire().is_none());
    });
}
