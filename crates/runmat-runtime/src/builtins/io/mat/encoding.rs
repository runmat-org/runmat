//! Shared Level-5 MAT value conversion and byte encoding.

use super::cooperation::MatCodecCooperation;
use super::format::{MatArray, MatClass, MatData};

mod diagnostics;
use crate::{gather_if_needed_async, BuiltinResult};
use diagnostics::MatEncodingFailure;
pub(crate) use diagnostics::MatEncodingOwner;
use futures::future::LocalBoxFuture;
use runmat_value::{
    CharArray, ComplexStorage, IntValue, IntegerStorage, NumericDType, NumericStorage, Value,
};

pub(super) struct MatVar {
    pub(super) name: String,
    pub(super) array: MatArray,
}

mod value;
#[cfg(test)]
use value::convert_value;
pub(super) use value::convert_value_with_cooperation;

async fn completed(
    cooperation: &mut Option<&mut MatCodecCooperation<'_>>,
    work: usize,
) -> BuiltinResult<()> {
    if let Some(cooperation) = cooperation.as_deref_mut() {
        cooperation.completed(work).await?;
    }
    Ok(())
}

mod scalars;
use scalars::tensor_dtype_mat_class;

mod writer;
pub(in crate::builtins::io::mat) use writer::write_mat_bytes_cooperative;
pub(super) use writer::write_mat_file;
pub async fn encode_workspace_to_mat_bytes(entries: &[(String, Value)]) -> BuiltinResult<Vec<u8>> {
    encode_workspace_to_mat_bytes_for_owner(entries, MatEncodingOwner::Save, None).await
}

pub(crate) async fn encode_workspace_to_mat_bytes_for_owner(
    entries: &[(String, Value)],
    owner: MatEncodingOwner,
    mut cooperation: Option<&mut MatCodecCooperation<'_>>,
) -> BuiltinResult<Vec<u8>> {
    let mut mat_vars = Vec::with_capacity(entries.len());
    for (name, value) in entries {
        completed(&mut cooperation, 0).await?;
        let array =
            convert_value_with_cooperation(value.clone(), owner, cooperation.as_deref_mut())
                .await?;
        mat_vars.push(MatVar {
            name: name.clone(),
            array,
        });
    }
    write_mat_bytes_cooperative(&mat_vars, owner, cooperation).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy, ExecutionServiceError};
    use futures::executor::block_on;
    use runmat_execution::cancellation::CancellationSource;
    use runmat_value::StringArray;
    use std::num::NonZeroU64;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn utf16_char_geometry_and_typed_missing_admission_are_explicit() {
        let array = block_on(convert_value(Value::String("A😀".into()))).expect("encode text");
        assert_eq!(array.dims, vec![1, 3]);
        let MatData::Char { data } = array.data else {
            panic!("text must become MAT char data");
        };
        assert_eq!(data, vec![0x0041, 0xd83d, 0xde00]);
        for text in ["<missing>", "<MISSING>", ""] {
            assert!(block_on(convert_value(Value::String(text.into()))).is_ok());
        }
        assert!(block_on(convert_value(Value::String(
            runmat_value::StringElement::Missing
        )))
        .is_err());
        let array = StringArray::from_elements_unbounded(
            vec!["literal".into(), runmat_value::StringElement::Missing],
            vec![1, 2],
        )
        .unwrap();
        assert!(block_on(convert_value(Value::StringArray(array))).is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn unicode_char_row_expands_mat_unit_geometry_without_truncation() {
        let source = CharArray::new_row("A😀");
        let array = block_on(convert_value(Value::CharArray(source)))
            .expect("Unicode char row has representable MAT geometry");
        assert_eq!(array.dims, vec![1, 3]);
        let MatData::Char { data } = array.data else {
            panic!("expected character payload");
        };
        assert_eq!(data, vec![0x0041, 0xd83d, 0xde00]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn multidimensional_char_array_preserves_utf16_units_and_column_major_layout() {
        let source = CharArray::from_code_units(
            vec![
                runmat_value::Utf16CodeUnit::from_raw(0xD83D),
                runmat_value::Utf16CodeUnit::from_raw(0xDE00),
                runmat_value::Utf16CodeUnit::from_raw(b'x' as u16),
                runmat_value::Utf16CodeUnit::from_raw(0xD800),
            ],
            2,
            2,
        )
        .expect("rectangular UTF-16 units");
        let array = block_on(convert_value(Value::CharArray(source)))
            .expect("raw UTF-16 geometry is directly representable in MAT");
        assert_eq!(array.dims, vec![2, 2]);
        let MatData::Char { data } = array.data else {
            panic!("expected character payload");
        };
        assert_eq!(data, vec![0xD83D, b'x' as u16, 0xDE00, 0xD800]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn matfile_encoding_uses_owning_error_identity() {
        let error = block_on(convert_value_with_cooperation(
            Value::output_list(Vec::new()),
            MatEncodingOwner::MatFile,
            None,
        ))
        .expect_err("MAT-file assignment rejects unsupported payload");
        assert_eq!(error.identifier(), Some("RunMat:matfile:Unsupported"));
        assert!(error.message().starts_with("matfile:"));

        for sparse in [
            runmat_value::SparseTensor::new_complex(1, 1, vec![0, 1], vec![0], vec![(1.0, 2.0)])
                .unwrap(),
            runmat_value::SparseTensor::new_complex_f32(
                1,
                1,
                vec![0, 1],
                vec![0],
                vec![(1.0, 2.0)],
            )
            .unwrap(),
        ] {
            for (owner, expected) in [
                (MatEncodingOwner::Save, "RunMat:save:Unsupported"),
                (MatEncodingOwner::MatFile, "RunMat:matfile:Unsupported"),
            ] {
                let error = block_on(convert_value_with_cooperation(
                    Value::SparseTensor(sparse.clone()),
                    owner,
                    None,
                ))
                .expect_err("unsupported sparse complex storage must not panic");
                assert_eq!(error.identifier(), Some(expected));
            }
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn numeric_low_rank_shapes_use_mat_canonical_dimensions_without_losing_storage_class() {
        let rank_zero =
            runmat_value::Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), Vec::new())
                .expect("rank-zero uint64");
        let encoded = block_on(convert_value(Value::Tensor(rank_zero))).expect("encode uint64");
        assert_eq!(encoded.class, MatClass::UInt64);
        assert_eq!(encoded.dims, vec![1, 1]);
        let MatData::Numeric { real, imag: None } = encoded.data else {
            panic!("expected real numeric MAT payload");
        };
        assert_eq!(real, NumericStorage::U64(vec![u64::MAX]));

        let rank_one =
            runmat_value::ComplexTensor::from_f32(vec![(1.25, -2.5), (3.5, 4.75)], vec![2])
                .expect("rank-one complex single");
        let encoded =
            block_on(convert_value(Value::ComplexTensor(rank_one))).expect("encode complex single");
        assert_eq!(encoded.class, MatClass::Single);
        assert_eq!(encoded.dims, vec![1, 2]);
        let MatData::Numeric {
            real,
            imag: Some(imag),
        } = encoded.data
        else {
            panic!("expected complex numeric MAT payload");
        };
        assert_eq!(real, NumericStorage::F32(vec![1.25, 3.5]));
        assert_eq!(imag, NumericStorage::F32(vec![-2.5, 4.75]));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn nested_string_cells_cancel_between_elements_without_result() {
        let cancellation = CancellationSource::new();
        let turns = Arc::new(AtomicUsize::new(0));
        let observed = turns.clone();
        let budget = CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
            cancellation.token(),
            move || {
                let turn = observed.fetch_add(1, Ordering::SeqCst);
                Box::pin(async move {
                    if turn == 0 {
                        Ok(())
                    } else {
                        Err(ExecutionServiceError::Cancelled)
                    }
                })
            },
        );
        let mut cooperation =
            MatCodecCooperation::with_budget(budget, "save", &runmat_builtins::SAVE_ERROR_IO);
        let strings = Value::StringArray(
            StringArray::new(
                vec!["first".into(), "second".into(), "third".into()],
                vec![1, 3],
            )
            .expect("string row fixture"),
        );
        let error = block_on(convert_value_with_cooperation(
            strings,
            MatEncodingOwner::Save,
            Some(&mut cooperation),
        ))
        .expect_err("cancellation before later nested entries");
        assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
        assert_eq!(turns.load(Ordering::SeqCst), 2);
        assert!(!cancellation.token().is_cancelled());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn nested_resident_gather_is_cancelled_while_provider_download_is_pending() {
        use crate::builtins::io::mat::tests::pending_provider::PendingDownloadProvider;
        use runmat_execution::CancellationReason;

        let _state = crate::builtins::io::mat::tests::state_guard();
        let owner = Box::leak(Box::new(PendingDownloadProvider::new()));
        let _gpu = runmat_accelerate_api::ThreadProviderGuard::set(Some(owner));
        let value = Value::Cell(
            runmat_value::CellArray::new(vec![Value::GpuTensor(owner.resident())], 1, 1)
                .expect("nested resident payload"),
        );
        let cancellation = CancellationSource::new();
        let budget = CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(4096).unwrap()),
            cancellation.token(),
            || Box::pin(async { Ok(()) }),
        );
        let mut cooperation =
            MatCodecCooperation::with_budget(budget, "save", &runmat_builtins::SAVE_ERROR_IO);
        let mut future = Box::pin(convert_value_with_cooperation(
            value,
            MatEncodingOwner::Save,
            Some(&mut cooperation),
        ));
        assert!(matches!(
            crate::builtins::io::mat::tests::receipt::poll_once(future.as_mut()),
            std::task::Poll::Pending
        ));
        assert_eq!(owner.requests(), 1);
        cancellation.cancel(CancellationReason::User);
        let error = block_on(future).expect_err("nested provider gather must cancel");
        assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
        assert_eq!(owner.abandoned(), 1);
    }
}
