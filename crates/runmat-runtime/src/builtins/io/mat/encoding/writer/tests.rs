use super::super::super::format::MatClass;
use super::*;
use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy, ExecutionServiceError};
use futures::executor::block_on;
use runmat_execution::cancellation::CancellationSource;
use runmat_filesystem::FsProvider;
use runmat_value::NumericStorage;
use std::num::NonZeroU64;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

fn virtual_directory() -> &'static str {
    "/mat"
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn a_large_dimension_rejects_instead_of_wrapping_its_mat_tag() {
    let vars = [MatVar {
        name: "x".into(),
        array: MatArray {
            class: MatClass::Double,
            dims: vec![i32::MAX as usize + 1, 1],
            data: MatData::Numeric {
                real: NumericStorage::F64(Vec::new()),
                imag: None,
            },
        },
    }];
    let error = write_mat_bytes(&vars).expect_err("MAT dimensions are signed 32-bit");
    assert_eq!(error.identifier(), Some("RunMat:save:Io"));
    assert!(error.message().contains("int32 range"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn matfile_writer_errors_retain_matfile_identity() {
    let vars = [MatVar {
        name: "x".into(),
        array: MatArray {
            class: MatClass::Double,
            dims: vec![i32::MAX as usize + 1, 1],
            data: MatData::Numeric {
                real: NumericStorage::F64(Vec::new()),
                imag: None,
            },
        },
    }];
    let error = block_on(write_mat_bytes_cooperative(
        &vars,
        MatEncodingOwner::MatFile,
        None,
    ))
    .expect_err("dimension does not fit MAT tag");
    assert_eq!(error.identifier(), Some("RunMat:matfile:Io"));
    assert!(error.message().starts_with("matfile:"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn malformed_shape_and_complex_storage_reject_before_encoded_publication() {
    let malformed = [
        MatArray {
            class: MatClass::Double,
            dims: vec![2, 2],
            data: MatData::Numeric {
                real: NumericStorage::F64(vec![1.0]),
                imag: None,
            },
        },
        MatArray {
            class: MatClass::Double,
            dims: vec![1, 1],
            data: MatData::Numeric {
                real: NumericStorage::F64(vec![1.0]),
                imag: Some(NumericStorage::F32(vec![1.0])),
            },
        },
    ];
    for array in malformed {
        let error = write_mat_bytes(&[MatVar {
            name: "bad".into(),
            array,
        }])
        .expect_err("invalid matrix never yields MAT bytes");
        assert_eq!(error.identifier(), Some("RunMat:save:Io"));
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn later_zero_extent_keeps_empty_mat_payload_valid() {
    let bytes = write_mat_bytes(&[MatVar {
        name: "empty".into(),
        array: MatArray {
            class: MatClass::Double,
            dims: vec![i32::MAX as usize, i32::MAX as usize, 0],
            data: MatData::Numeric {
                real: NumericStorage::F64(Vec::new()),
                imag: None,
            },
        },
    }])
    .expect("zero cardinality despite earlier large dimensions");
    assert!(bytes.len() > MAT_HEADER_LEN);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn struct_field_width_includes_nul_and_nested_cardinality_is_checked() {
    let scalar = MatArray {
        class: MatClass::Double,
        dims: vec![1, 1],
        data: MatData::Numeric {
            real: NumericStorage::F64(vec![1.0]),
            imag: None,
        },
    };
    let make_struct = |field_values: Vec<MatArray>| MatVar {
        name: "record".into(),
        array: MatArray {
            class: MatClass::Struct,
            dims: vec![1, 1],
            data: MatData::Struct {
                field_names: vec!["abc".into()],
                field_values,
            },
        },
    };
    let bytes = write_mat_bytes(&[make_struct(vec![scalar])]).expect("valid MAT struct");
    assert!(bytes
        .windows(12)
        .any(|window| window == [5, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0]));
    assert!(bytes.windows(4).any(|window| window == b"abc\0"));
    let error =
        write_mat_bytes(&[make_struct(Vec::new())]).expect_err("missing nested field value");
    assert_eq!(error.identifier(), Some("RunMat:save:Io"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn sparse_row_and_column_metadata_reject_before_mat_bytes() {
    let malformed = MatVar {
        name: "s".into(),
        array: MatArray {
            class: MatClass::Sparse,
            dims: vec![2, 2],
            data: MatData::Sparse {
                rows: 2,
                cols: 2,
                col_ptrs: vec![0, 2, 1],
                row_indices: vec![0],
                values: NumericStorage::F64(vec![1.0]),
                logical: false,
            },
        },
    };
    let error = write_mat_bytes(&[malformed]).expect_err("invalid sparse pointer geometry");
    assert_eq!(error.identifier(), Some("RunMat:save:Io"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn sparse_order_and_logical_value_metadata_reject_before_mat_bytes() {
    let sparse = |row_indices: Vec<usize>, values: NumericStorage, logical| MatVar {
        name: "s".into(),
        array: MatArray {
            class: MatClass::Sparse,
            dims: vec![3, 1],
            data: MatData::Sparse {
                rows: 3,
                cols: 1,
                col_ptrs: vec![0, 2],
                row_indices,
                values,
                logical,
            },
        },
    };
    let duplicate = sparse(vec![1, 1], NumericStorage::F64(vec![1.0, 2.0]), false);
    let error = write_mat_bytes(&[duplicate]).expect_err("duplicate CSC row indices");
    assert_eq!(error.identifier(), Some("RunMat:save:Io"));
    assert!(error.message().contains("sorted and unique"));

    let wrong_class = sparse(vec![0, 1], NumericStorage::F64(vec![1.0, 1.0]), true);
    let error = write_mat_bytes(&[wrong_class]).expect_err("logical CSC storage class");
    assert_eq!(error.identifier(), Some("RunMat:save:Io"));
    assert!(error.message().contains("uint8 storage"));

    let false_entry = sparse(vec![0, 1], NumericStorage::U8(vec![1, 0]), true);
    let error = write_mat_bytes(&[false_entry])
        .expect_err("logical CSC pattern contains only true entries");
    assert_eq!(error.identifier(), Some("RunMat:save:Io"));
    assert!(error.message().contains("true at every stored index"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn matrix_rank_rejects_before_mat_bytes() {
    let short_rank = MatVar {
        name: "x".into(),
        array: MatArray {
            class: MatClass::Logical,
            dims: vec![1],
            data: MatData::Logical { data: vec![1] },
        },
    };
    let error = write_mat_bytes(&[short_rank]).expect_err("MAT array rank");
    assert_eq!(error.identifier(), Some("RunMat:save:Io"));
    assert!(error.message().contains("at least two dimensions"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn checked_plan_matches_encoded_bytes_across_nested_mat_classes() {
    let scalar = MatArray {
        class: MatClass::Single,
        dims: vec![1, 1],
        data: MatData::Numeric {
            real: NumericStorage::F32(vec![1.25]),
            imag: Some(NumericStorage::F32(vec![-2.5])),
        },
    };
    let vars = vec![
        MatVar {
            name: "numeric".into(),
            array: scalar.clone(),
        },
        MatVar {
            name: "logical".into(),
            array: MatArray {
                class: MatClass::Logical,
                dims: vec![1, 2],
                data: MatData::Logical { data: vec![1, 0] },
            },
        },
        MatVar {
            name: "unicode".into(),
            array: MatArray {
                class: MatClass::Char,
                dims: vec![1, 2],
                data: MatData::Char {
                    data: vec![0xd83d, 0xde00],
                },
            },
        },
        MatVar {
            name: "nested".into(),
            array: MatArray {
                class: MatClass::Cell,
                dims: vec![1, 1],
                data: MatData::Cell {
                    elements: vec![MatArray {
                        class: MatClass::Struct,
                        dims: vec![1, 1],
                        data: MatData::Struct {
                            field_names: vec!["value".into()],
                            field_values: vec![scalar],
                        },
                    }],
                },
            },
        },
        MatVar {
            name: "sparse".into(),
            array: MatArray {
                class: MatClass::Sparse,
                dims: vec![3, 2],
                data: MatData::Sparse {
                    rows: 3,
                    cols: 2,
                    col_ptrs: vec![0, 1, 2],
                    row_indices: vec![0, 2],
                    values: NumericStorage::F64(vec![4.0, 5.0]),
                    logical: false,
                },
            },
        },
    ];
    let mut cooperation = None;
    let planned = block_on(plan::plan_workspace(
        &vars,
        MatEncodingOwner::Save,
        &mut cooperation,
    ))
    .expect("checked MAT layout");
    let bytes = write_mat_bytes(&vars).expect("same values encode");
    assert_eq!(planned.total_bytes, bytes.len());
    assert_eq!(planned.matrices.len(), vars.len());
    assert_eq!(planned.matrices[3].children[0].children.len(), 1);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn cancelled_encoding_never_publishes_to_captured_owner() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    let provider = Arc::new(runmat_filesystem::MemoryFsProvider::with_current_dir(
        virtual_directory(),
    ));
    let _selected = runmat_filesystem::replace_provider(provider.clone());
    runmat_filesystem::set_current_dir(virtual_directory()).expect("selected virtual cwd");
    let path = MatFsPath::capture(PathBuf::from("cancelled.mat")).expect("capture owner");
    let cancellation = CancellationSource::new();
    let budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
        cancellation.token(),
        || Box::pin(async { Err(ExecutionServiceError::Cancelled) }),
    );
    let mut cooperation =
        MatCodecCooperation::with_budget(budget, "save", &runmat_builtins::SAVE_ERROR_IO);
    let vars = [MatVar {
        name: "x".into(),
        array: MatArray {
            class: MatClass::Double,
            dims: vec![1, 1],
            data: MatData::Numeric {
                real: NumericStorage::F64(vec![42.0]),
                imag: None,
            },
        },
    }];
    let error = block_on(write_mat_file(
        &path,
        &vars,
        MatEncodingOwner::Save,
        &mut cooperation,
    ))
    .expect_err("cancel before publication");
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert_eq!(
        error
            .source
            .as_deref()
            .and_then(|cause| cause.downcast_ref::<ExecutionServiceError>()),
        Some(&ExecutionServiceError::Cancelled)
    );
    let error = block_on(provider.read(path.path())).expect_err("no published MAT file");
    assert_eq!(error.kind(), std::io::ErrorKind::NotFound);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn large_numeric_encoding_cancels_inside_payload_before_write() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    let provider = Arc::new(runmat_filesystem::MemoryFsProvider::with_current_dir(
        virtual_directory(),
    ));
    let _selected = runmat_filesystem::replace_provider(provider.clone());
    runmat_filesystem::set_current_dir(virtual_directory()).expect("selected virtual cwd");
    let path =
        MatFsPath::capture(PathBuf::from("large-cancelled.mat")).expect("capture original owner");
    let cancellation = CancellationSource::new();
    let turns = Arc::new(AtomicUsize::new(0));
    let observed = turns.clone();
    let budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(ENCODE_WORK_CHUNK as u64).unwrap()),
        cancellation.token(),
        move || {
            observed.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { Err(ExecutionServiceError::Cancelled) })
        },
    );
    let mut cooperation =
        MatCodecCooperation::with_budget(budget, "save", &runmat_builtins::SAVE_ERROR_IO);
    let vars = [MatVar {
        name: "large".into(),
        array: MatArray {
            class: MatClass::UInt64,
            dims: vec![1, 20_000],
            data: MatData::Numeric {
                real: NumericStorage::U64(vec![u64::MAX; 20_000]),
                imag: None,
            },
        },
    }];
    let error = block_on(write_mat_file(
        &path,
        &vars,
        MatEncodingOwner::Save,
        &mut cooperation,
    ))
    .expect_err("service cancels during numeric payload encoding");
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert_eq!(turns.load(Ordering::SeqCst), 1);
    assert_eq!(
        block_on(provider.read(path.path()))
            .expect_err("unpublished MAT bytes")
            .kind(),
        std::io::ErrorKind::NotFound
    );
}
