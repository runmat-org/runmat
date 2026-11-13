use super::*;
use runmat_accelerate_api::AccelProvider;
use runmat_builtins::{
    BuiltinIntegerAuditKind, APPEND_INTEGER_AUDIT, ERASE_PUNCTUATION_DESCRIPTOR,
    ERASE_URLS_DESCRIPTOR, EXTRACT_BEFORE_RESIDENT_POSITION_EXTENSION,
};
use runmat_value::{CellArray, IntValue, IntegerStorage, ObjectInstance, StructValue, Tensor};
use std::sync::atomic::{AtomicUsize, Ordering};

struct CountingBoundaryProvider {
    inner: runmat_accelerate::simple_provider::InProcessProvider,
    downloads: AtomicUsize,
}

impl CountingBoundaryProvider {
    fn new() -> Self {
        Self {
            inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
            downloads: AtomicUsize::new(0),
        }
    }

    fn downloads(&self) -> usize {
        self.downloads.load(Ordering::SeqCst)
    }
}

impl runmat_accelerate_api::AccelProvider for CountingBoundaryProvider {
    fn device_info(&self) -> String {
        self.inner.device_info()
    }

    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }

    fn upload(
        &self,
        host: &runmat_accelerate_api::HostTensorView,
    ) -> anyhow::Result<runmat_accelerate_api::GpuTensorHandle> {
        self.inner.upload(host)
    }

    fn download<'a>(
        &'a self,
        handle: &'a runmat_accelerate_api::GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        self.inner.download(handle)
    }

    fn download_numeric<'a>(
        &'a self,
        handle: &'a runmat_accelerate_api::GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelNumericDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        self.inner.download_numeric(handle)
    }
}

fn counted_boundary_provider() -> &'static CountingBoundaryProvider {
    Box::leak(Box::new(CountingBoundaryProvider::new()))
}

fn resident_position(provider: &CountingBoundaryProvider) -> Value {
    Value::GpuTensor(
        provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &[2.0],
                shape: &[1, 1],
            })
            .expect("resident position"),
    )
}

fn block(value: impl std::future::Future<Output = BuiltinResult<Value>>) -> BuiltinResult<Value> {
    futures::executor::block_on(value)
}

#[test]
fn append_extract_insert_and_replace_work() {
    assert_eq!(
        block(append_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("run".into()),
                Value::String("mat".into())
            ])
        ))
        .unwrap(),
        Value::String("runmat".into())
    );
    assert_eq!(
        block(extract_before_builtin(
            Value::String("alpha.beta".into()),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                ".".into()
            )]),
        ))
        .unwrap(),
        Value::String("alpha".into())
    );
    assert_eq!(
        block(insert_after_builtin(
            Value::String("run".into()),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(3.0),
                Value::String("mat".into())
            ]),
        ))
        .unwrap(),
        Value::String("runmat".into())
    );
    assert_eq!(
        block(extract_after_builtin(
            Value::String("alpha.beta".into()),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                ".".into()
            )]),
        ))
        .unwrap(),
        Value::String("beta".into())
    );
    assert_eq!(
        block(insert_before_builtin(
            Value::String("run".into()),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(1.0),
                Value::String("pre".into())
            ]),
        ))
        .unwrap(),
        Value::String("prerun".into())
    );
    assert_eq!(
        block(replace_between_builtin(
            Value::String("a[old]b".into()),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("[".into()),
                Value::String("]".into()),
                Value::String("new".into()),
            ])
        ))
        .unwrap(),
        Value::String("a[new]b".into())
    );
}

#[test]
fn replace_between_rejects_trailing_arguments_before_provider_access() {
    let provider = counted_boundary_provider();
    let position = resident_position(provider);
    let error = block(replace_between_builtin(
        Value::string("abc"),
        crate::builtins::common::test_support::builtin_arguments(vec![
            position.clone(),
            Value::Num(2.0),
            Value::string("X"),
            Value::string("ignored"),
        ]),
    ))
    .expect_err("extra control must not be silently ignored");
    assert_eq!(
        error.identifier(),
        runmat_builtins::REPLACE_BETWEEN_ERROR_ARGUMENT_COUNT.identifier
    );
    assert_eq!(provider.downloads(), 0);
    if let Value::GpuTensor(handle) = position {
        handle.release().expect("release position");
    }
}

#[test]
fn append_integer_audit_is_explicitly_inapplicable() {
    assert_eq!(
        APPEND_INTEGER_AUDIT.kind,
        BuiltinIntegerAuditKind::NotApplicable
    );
    assert!(APPEND_INTEGER_AUDIT.canonical_builtin.is_none());
    assert!(APPEND_INTEGER_AUDIT.notes.contains("all eight integer"));
}

#[test]
fn append_preserves_documented_text_output_precedence_and_whitespace() {
    assert_eq!(
        block(append_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::CharArray(CharArray::new_row("Hello ")),
                Value::CharArray(CharArray::new_row("World")),
            ])
        ))
        .expect("all-char append"),
        Value::CharArray(CharArray::new_row("Hello World"))
    );

    let cell = CellArray::new(
        vec![
            Value::CharArray(CharArray::new_row("alpha")),
            Value::CharArray(CharArray::new_row("beta")),
        ],
        1,
        2,
    )
    .expect("cellstr");
    let result = block(append_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Cell(cell.clone()),
            Value::CharArray(CharArray::new_row(" ")),
        ]),
    ))
    .expect("cell append");
    let Value::Cell(cell_result) = result else {
        panic!("expected cellstr output");
    };
    assert_eq!(cell_result.shape, vec![1, 2]);
    assert_eq!(
        cell_result.data(),
        vec![
            Value::CharArray(CharArray::new_row("alpha ")),
            Value::CharArray(CharArray::new_row("beta ")),
        ]
    );

    let strings = StringArray::new(vec!["A".into(), "B".into()], vec![2, 1]).expect("strings");
    let result = block(append_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::StringArray(strings),
            Value::Cell(cell),
        ]),
    ))
    .expect("string-cell append");
    let Value::StringArray(string_result) = result else {
        panic!("expected string-array output");
    };
    assert_eq!(string_result.shape, vec![2, 2]);
    assert_eq!(
        string_result.data(),
        vec!["Aalpha", "Balpha", "Abeta", "Bbeta"]
    );
}

#[test]
fn append_rejects_every_integer_class_without_numeric_conversion() {
    let cases = [
        ("int8", IntValue::I8(-1), IntegerStorage::I8(vec![-1, 2])),
        ("int16", IntValue::I16(-2), IntegerStorage::I16(vec![-2, 3])),
        ("int32", IntValue::I32(-3), IntegerStorage::I32(vec![-3, 4])),
        (
            "int64",
            IntValue::I64(i64::MIN),
            IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
        ),
        (
            "uint8",
            IntValue::U8(5),
            IntegerStorage::U8(vec![5, u8::MAX]),
        ),
        (
            "uint16",
            IntValue::U16(6),
            IntegerStorage::U16(vec![6, u16::MAX]),
        ),
        (
            "uint32",
            IntValue::U32(7),
            IntegerStorage::U32(vec![7, u32::MAX]),
        ),
        (
            "uint64",
            IntValue::U64(u64::MAX),
            IntegerStorage::U64(vec![(1_u64 << 53) + 1, u64::MAX]),
        ),
    ];
    for (class, scalar, storage) in cases {
        for value in [
            Value::Int(scalar),
            Value::Tensor(Tensor::new_integer(storage, vec![1, 2]).expect("integer tensor")),
        ] {
            let error = block(append_builtin(
                crate::builtins::common::test_support::builtin_arguments(vec![
                    value,
                    Value::CharArray(CharArray::new_row("suffix")),
                ]),
            ))
            .expect_err("numeric append input must reject");
            assert!(
                error.message().contains(
                    "expected string, character vector, or cell array of character vectors"
                ),
                "{class}: {error}"
            );
        }
    }

    for value in [Value::Num(1.0), Value::Bool(true)] {
        let error = block(append_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                value,
                Value::CharArray(CharArray::new_row("suffix")),
            ]),
        ))
        .expect_err("nontext append input must reject");
        assert!(error.message().contains("expected string"), "{error}");
    }

    let invalid_cell =
        CellArray::new(vec![Value::String("not a character vector".into())], 1, 1).expect("cell");
    let error = block(append_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Cell(invalid_cell)]),
    ))
    .expect_err("non-cellstr input must reject");
    assert!(
        error
            .message()
            .contains("must contain only character vectors"),
        "{error}"
    );
}

#[test]
fn append_rejects_resident_integer_handles_before_provider_gather() {
    for element_type in [
        runmat_accelerate_api::IntegerElementType::I8,
        runmat_accelerate_api::IntegerElementType::I16,
        runmat_accelerate_api::IntegerElementType::I32,
        runmat_accelerate_api::IntegerElementType::I64,
        runmat_accelerate_api::IntegerElementType::U8,
        runmat_accelerate_api::IntegerElementType::U16,
        runmat_accelerate_api::IntegerElementType::U32,
        runmat_accelerate_api::IntegerElementType::U64,
    ] {
        let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 1],
                element_type.into(),
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .unwrap();
        let result = block(append_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::GpuTensor(allocation.handle.clone()),
                Value::String("suffix".into()),
            ]),
        ));
        let error = result.expect_err("resident numeric append input must reject");
        assert!(error.message().contains("expected string"), "{error}");
        allocation
            .handle
            .allocation_lease()
            .validate_live()
            .unwrap();
        assert_eq!(allocation.probe.retirements(), 0);
    }
}

#[test]
fn text_boundary_positions_read_typed_integer_storage_exactly() {
    let boundary = Tensor::new_integer(IntegerStorage::U16(vec![3]), vec![1, 1]).expect("boundary");

    assert_eq!(
        block(insert_after_builtin(
            Value::String("run".into()),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Tensor(boundary),
                Value::String("mat".into())
            ]),
        ))
        .unwrap(),
        Value::String("runmat".into())
    );

    let start = Tensor::new_integer(IntegerStorage::U16(vec![2]), vec![1, 1]).expect("start");
    let stop = Tensor::new_integer(IntegerStorage::U16(vec![4]), vec![1, 1]).expect("stop");

    assert_eq!(
        block(replace_between_builtin(
            Value::String("abcde".into()),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Tensor(start),
                Value::Tensor(stop),
                Value::String("X".into()),
            ])
        ))
        .unwrap(),
        Value::String("aXe".into())
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn numeric_boundary_positions_preserve_rank_zero_and_rank_one_shapes() {
    assert_eq!(
        block(insert_after_builtin(
            Value::String("ab".into()),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Tensor(Tensor::new(vec![1.0], vec![]).unwrap()),
                Value::String("X".into()),
            ]),
        ))
        .expect("rank-zero scalar boundary"),
        Value::String("aXb".into())
    );

    let text =
        Value::StringArray(StringArray::new(vec!["ab".into(), "cd".into()], vec![2]).unwrap());
    let value = block(insert_after_builtin(
        text,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2]).unwrap()),
            Value::String("X".into()),
        ]),
    ))
    .expect("rank-one matching boundaries");
    let Value::StringArray(output) = value else {
        panic!("expected string array")
    };
    assert_eq!(output.shape, vec![2]);
    assert_eq!(output.data(), vec!["aXb", "cdX"]);
}

#[test]
fn replace_between_accepts_all_integer_position_classes_and_same_size_arrays() {
    for position in [
        IntValue::I8(2),
        IntValue::I16(2),
        IntValue::I32(2),
        IntValue::I64(2),
        IntValue::U8(2),
        IntValue::U16(2),
        IntValue::U32(2),
        IntValue::U64(2),
    ] {
        assert_eq!(
            block(replace_between_builtin(
                Value::String("abc".into()),
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::Int(position.clone()),
                    Value::Int(position),
                    Value::String("X".into()),
                ])
            ))
            .expect("integer positions"),
            Value::String("aXc".into())
        );
    }

    let text = Value::StringArray(
        StringArray::new(vec!["abcde".into(), "vwxyz".into()], vec![2, 1]).unwrap(),
    );
    let starts =
        Value::Tensor(Tensor::new_integer(IntegerStorage::U64(vec![2, 3]), vec![2, 1]).unwrap());
    let stops =
        Value::Tensor(Tensor::new_integer(IntegerStorage::U64(vec![4, 5]), vec![2, 1]).unwrap());
    let replacements =
        Value::StringArray(StringArray::new(vec!["X".into(), "Q".into()], vec![2, 1]).unwrap());
    let Value::StringArray(output) = block(replace_between_builtin(
        text,
        crate::builtins::common::test_support::builtin_arguments(vec![starts, stops, replacements]),
    ))
    .expect("same-size positions") else {
        panic!("string array")
    };
    assert_eq!(output.data(), vec!["aXe", "vwQ"]);
}

#[test]
fn text_boundary_positions_reject_invalid_integer_and_double_values() {
    let zero = Tensor::new_integer(IntegerStorage::U16(vec![0]), vec![1, 1]).expect("boundary");
    assert!(block(insert_after_builtin(
        Value::String("run".into()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Tensor(zero),
            Value::String("mat".into())
        ]),
    ))
    .is_err());

    assert!(block(insert_after_builtin(
        Value::String("run".into()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(1.0e300),
            Value::String("mat".into())
        ]),
    ))
    .is_err());
}

#[test]
fn insert_before_and_after_accept_all_integer_position_classes() {
    for position in [
        IntValue::I8(2),
        IntValue::I16(2),
        IntValue::I32(2),
        IntValue::I64(2),
        IntValue::U8(2),
        IntValue::U16(2),
        IntValue::U32(2),
        IntValue::U64(2),
    ] {
        assert_eq!(
            block(insert_before_builtin(
                Value::String("abc".into()),
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::Int(position.clone()),
                    Value::String("X".into())
                ]),
            ))
            .unwrap(),
            Value::String("aXbc".into())
        );
        assert_eq!(
            block(insert_after_builtin(
                Value::String("abc".into()),
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::Int(position),
                    Value::String("X".into())
                ]),
            ))
            .unwrap(),
            Value::String("abXc".into())
        );
    }
    for storage in [
        IntegerStorage::I8(vec![2]),
        IntegerStorage::I16(vec![2]),
        IntegerStorage::I32(vec![2]),
        IntegerStorage::I64(vec![2]),
        IntegerStorage::U8(vec![2]),
        IntegerStorage::U16(vec![2]),
        IntegerStorage::U32(vec![2]),
        IntegerStorage::U64(vec![2]),
    ] {
        let position = Value::Tensor(Tensor::new_integer(storage, vec![1, 1]).unwrap());
        assert_eq!(
            block(insert_after_builtin(
                Value::String("abc".into()),
                crate::builtins::common::test_support::builtin_arguments(vec![
                    position,
                    Value::String("X".into())
                ]),
            ))
            .unwrap(),
            Value::String("abXc".into())
        );
    }
    for position in [
        Value::Tensor(Tensor::from_f32(vec![2.0], vec![1, 1]).unwrap()),
        Value::Tensor(Tensor::new(vec![2.0], vec![1, 1]).unwrap()),
    ] {
        assert_eq!(
            block(insert_before_builtin(
                Value::String("abc".into()),
                crate::builtins::common::test_support::builtin_arguments(vec![
                    position,
                    Value::String("X".into())
                ]),
            ))
            .unwrap(),
            Value::String("aXbc".into())
        );
    }
    assert!(block(insert_after_builtin(
        Value::String("abc".into()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Bool(true),
            Value::String("X".into())
        ]),
    ))
    .is_err());
}

#[test]
fn insertion_broadcasts_scalar_or_same_size_positions_and_new_text() {
    let source = Value::StringArray(
        StringArray::new(vec!["abcd".into(), "wxyz".into()], vec![2, 1]).unwrap(),
    );
    let positions =
        Value::Tensor(Tensor::new_integer(IntegerStorage::U64(vec![2, 3]), vec![2, 1]).unwrap());
    let replacements =
        Value::StringArray(StringArray::new(vec!["A".into(), "B".into()], vec![2, 1]).unwrap());
    assert_eq!(
        block(insert_before_builtin(
            source.clone(),
            crate::builtins::common::test_support::builtin_arguments(vec![positions, replacements])
        ))
        .unwrap(),
        Value::StringArray(
            StringArray::new(vec!["aAbcd".into(), "wxByz".into()], vec![2, 1]).unwrap()
        )
    );
    assert_eq!(
        block(insert_after_builtin(
            source,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(1.0),
                Value::String("!".into())
            ])
        ))
        .unwrap(),
        Value::StringArray(
            StringArray::new(vec!["a!bcd".into(), "w!xyz".into()], vec![2, 1]).unwrap()
        )
    );
}

#[test]
fn insertion_applies_text_boundaries_at_every_occurrence() {
    assert_eq!(
        block(insert_before_builtin(
            Value::String("a-b-c".into()),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("-".into()),
                Value::String("X".into())
            ]),
        ))
        .unwrap(),
        Value::String("aX-bX-c".into())
    );
    assert_eq!(
        block(insert_after_builtin(
            Value::String("a-b-c".into()),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("-".into()),
                Value::String("X".into())
            ]),
        ))
        .unwrap(),
        Value::String("a-Xb-Xc".into())
    );
}

#[test]
fn insertion_requires_exact_arity_and_matching_array_shapes() {
    assert!(block(insert_before_builtin(
        Value::String("abc".into()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(1.0),
            Value::String("X".into()),
            Value::String("ignored".into()),
        ]),
    ))
    .is_err());
    let source =
        Value::StringArray(StringArray::new(vec!["a".into(), "b".into()], vec![2, 1]).unwrap());
    let positions = Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap());
    assert!(block(insert_after_builtin(
        source,
        crate::builtins::common::test_support::builtin_arguments(vec![
            positions,
            Value::String("X".into())
        ])
    ))
    .is_err());
}

#[test]
fn insertion_compatibility_gates_precede_resident_gather() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let resident = Value::GpuTensor(
        runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .unwrap()
            .handle,
    );
    let error = block(insert_after_builtin(
        Value::String("abc".into()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            resident,
            Value::String("X".into()),
        ]),
    ))
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        INSERT_AFTER_RESIDENT_POSITION_EXTENSION.error_identifier
    );

    let matrix = Value::CharArray(
        CharArray::from_unicode_scalars(vec!['a', 'b', 'c', 'd'], 2, 2).expect("character matrix"),
    );
    let error = block(insert_before_builtin(
        matrix,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(1.0),
            Value::String("X".into()),
        ]),
    ))
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        INSERT_BEFORE_BROAD_TEXT_EXTENSION.error_identifier
    );
}

#[test]
fn splitlines_reverse_deblank_and_matches_work() {
    assert_eq!(
        block(reverse_builtin(Value::String("abc".into()))).unwrap(),
        Value::String("cba".into())
    );
    assert_eq!(
        block(deblank_builtin(Value::CharArray(CharArray::new_row(
            "abc   "
        ))))
        .unwrap(),
        Value::CharArray(CharArray::new_row("abc"))
    );
    assert_eq!(
        block(matches_builtin(
            Value::String("A12".into()),
            crate::builtins::strings::core::compat::pattern_object(r"A\d+"),
        ))
        .unwrap(),
        Value::Bool(true)
    );
    let cell = make_cell_with_shape(
        ["hit", "miss", "hit", "miss"]
            .into_iter()
            .map(|text| Value::CharArray(CharArray::new_row(text)))
            .collect(),
        vec![2, 2],
    )
    .expect("physical cell order");
    let Value::LogicalArray(result) =
        block(matches_builtin(cell, Value::string("hit"))).expect("cell matches")
    else {
        panic!("logical output");
    };
    assert_eq!(result.shape.as_slice(), &[2, 2]);
    assert_eq!(result.data.as_slice(), &[1, 1, 0, 0]);
    assert_eq!(
        block(erase_urls_builtin(Value::String(
            "see https://example.com now".into()
        )))
        .unwrap(),
        Value::String("see  now".into())
    );
    assert_eq!(
        block(erase_punctuation_builtin(
            Value::String("it's one and/or two.".into()),
            crate::builtins::common::test_support::builtin_arguments(vec![])
        ))
        .unwrap(),
        Value::String("its one andor two".into())
    );
    assert_eq!(
        block(erase_punctuation_builtin(
            Value::CharArray(CharArray::new_row("cost: $5.00!")),
            crate::builtins::common::test_support::builtin_arguments(vec![])
        ))
        .unwrap(),
        Value::CharArray(CharArray::new_row("cost 500"))
    );
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let cell = CellArray::new(
        vec![
            Value::CharArray(CharArray::new_row("alpha,beta!")),
            Value::String("C++ and C#".into()),
        ],
        1,
        2,
    )
    .unwrap();
    match block(erase_punctuation_builtin(
        Value::Cell(cell),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap()
    {
        Value::Cell(out) => {
            assert_eq!(out.shape, vec![1, 2]);
            assert_eq!(
                out.data()[0],
                Value::CharArray(CharArray::new_row("alphabeta"))
            );
            assert_eq!(out.data()[1], Value::String("C and C".into()));
        }
        other => panic!("expected cell array, got {other:?}"),
    }
    assert_eq!(
        block(strjust_builtin(
            None,
            Value::CharArray(CharArray::new_row("  x")),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                "left".into()
            )]),
        ))
        .unwrap(),
        Value::CharArray(CharArray::new_row("x  "))
    );
}

#[test]
fn erase_descriptors_and_integer_audits_are_settled() {
    assert_eq!(ERASE_URLS_DESCRIPTOR.signatures.len(), 2);
    assert_eq!(ERASE_PUNCTUATION_DESCRIPTOR.signatures.len(), 3);
    for name in ["eraseURLs", "erasePunctuation"] {
        let builtin = runmat_builtins::builtin_integer_metadata_by_name(name)
            .expect("integer metadata registered");
        assert_eq!(
            builtin.audit.expect("integer audit").kind,
            BuiltinIntegerAuditKind::NotApplicable
        );
    }
}

#[test]
fn erase_text_builtins_reject_all_integer_classes_and_logical_values() {
    for value in [
        IntValue::I8(-1),
        IntValue::I16(-1),
        IntValue::I32(-1),
        IntValue::I64(-1),
        IntValue::U8(1),
        IntValue::U16(1),
        IntValue::U32(1),
        IntValue::U64(u64::MAX),
    ] {
        assert!(block(erase_urls_builtin(Value::Int(value.clone()))).is_err());
        assert!(block(erase_punctuation_builtin(
            Value::Int(value),
            crate::builtins::common::test_support::builtin_arguments(vec![])
        ))
        .is_err());
    }
    assert!(block(erase_urls_builtin(Value::Bool(true))).is_err());
    assert!(block(erase_punctuation_builtin(
        Value::Bool(true),
        crate::builtins::common::test_support::builtin_arguments(vec![])
    ))
    .is_err());
}

#[test]
fn erase_text_broad_container_extensions_are_independently_gated() {
    let matrix = Value::CharArray(
        CharArray::from_unicode_scalars(vec!['a', '!', 'b', '?'], 2, 2).expect("character matrix"),
    );
    let broad_cell = Value::Cell(
        CellArray::new(vec![Value::String("https://example.com!".into())], 1, 1)
            .expect("broad text cell"),
    );
    let nested_cell = Value::Cell(
        CellArray::new(
            vec![Value::Cell(
                CellArray::new(vec![Value::CharArray(CharArray::new_row("nested!"))], 1, 1)
                    .expect("inner cell"),
            )],
            1,
            1,
        )
        .expect("outer cell"),
    );
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    for (value, identifier) in [
        (
            matrix.clone(),
            "RunMat:compatibility:EraseURLsCharMatrixExtension",
        ),
        (
            broad_cell.clone(),
            "RunMat:compatibility:EraseURLsBroadCellExtension",
        ),
    ] {
        let error = block(erase_urls_builtin(value)).expect_err("strict eraseURLs gate");
        assert_eq!(error.identifier(), Some(identifier));
    }
    for (value, identifier) in [
        (
            matrix,
            "RunMat:compatibility:ErasePunctuationCharMatrixExtension",
        ),
        (
            broad_cell,
            "RunMat:compatibility:ErasePunctuationBroadCellExtension",
        ),
        (
            nested_cell,
            "RunMat:compatibility:ErasePunctuationBroadCellExtension",
        ),
    ] {
        let error = block(erase_punctuation_builtin(
            value,
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .expect_err("strict erasePunctuation gate");
        assert_eq!(error.identifier(), Some(identifier));
    }
}

#[test]
fn erase_text_builtins_reject_resident_numeric_without_provider_access() {
    let resident = Value::GpuTensor(
        runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .unwrap()
            .handle,
    );
    let url_error = block(erase_urls_builtin(resident.clone())).unwrap_err();
    assert_eq!(
        url_error.identifier(),
        ERASE_URLS_ERROR_INVALID_INPUT.identifier
    );
    let nested = Value::Cell(CellArray::new(vec![resident], 1, 1).unwrap());
    let punctuation_error = block(erase_punctuation_builtin(
        nested,
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap_err();
    assert_eq!(
        punctuation_error.identifier(),
        ERASE_PUNCTUATION_ERROR_INVALID_INPUT.identifier
    );
}

#[test]
fn extract_before_and_after_accept_every_integer_scalar_class_exactly() {
    assert_eq!(EXTRACT_BEFORE_DESCRIPTOR.errors.len(), 3);
    assert_eq!(EXTRACT_AFTER_DESCRIPTOR.errors.len(), 3);
    for position in [
        IntValue::I8(2),
        IntValue::I16(2),
        IntValue::I32(2),
        IntValue::I64(2),
        IntValue::U8(2),
        IntValue::U16(2),
        IntValue::U32(2),
        IntValue::U64(2),
    ] {
        assert_eq!(
            block(extract_before_builtin(
                Value::String("abcdef".into()),
                crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(
                    position.clone()
                )]),
            ))
            .expect("extractBefore"),
            Value::String("a".into())
        );
        assert_eq!(
            block(extract_after_builtin(
                Value::String("abcdef".into()),
                crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(
                    position
                )]),
            ))
            .expect("extractAfter"),
            Value::String("cdef".into())
        );
    }
}

#[test]
fn extract_before_and_after_apply_same_size_native_integer_positions() {
    let text = Value::StringArray(
        StringArray::new(vec!["abcd".into(), "wxyz".into()], vec![2, 1]).unwrap(),
    );
    for storage in [
        IntegerStorage::I8(vec![2, 3]),
        IntegerStorage::I16(vec![2, 3]),
        IntegerStorage::I32(vec![2, 3]),
        IntegerStorage::I64(vec![2, 3]),
        IntegerStorage::U8(vec![2, 3]),
        IntegerStorage::U16(vec![2, 3]),
        IntegerStorage::U32(vec![2, 3]),
        IntegerStorage::U64(vec![2, 3]),
    ] {
        let positions = Value::Tensor(Tensor::new_integer(storage, vec![2, 1]).unwrap());
        assert_eq!(
            block(extract_before_builtin(
                text.clone(),
                crate::builtins::common::test_support::builtin_arguments(vec![positions.clone()])
            ))
            .unwrap(),
            Value::StringArray(
                StringArray::new(vec!["a".into(), "wx".into()], vec![2, 1]).unwrap()
            )
        );
        assert_eq!(
            block(extract_after_builtin(
                text.clone(),
                crate::builtins::common::test_support::builtin_arguments(vec![positions])
            ))
            .unwrap(),
            Value::StringArray(
                StringArray::new(vec!["cd".into(), "z".into()], vec![2, 1]).unwrap()
            )
        );
    }
}

#[test]
fn extract_before_and_after_enforce_strict_extensions_before_gather() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let matrix = Value::CharArray(
        CharArray::from_unicode_scalars(vec!['a', 'b', 'c', 'd'], 2, 2).expect("character matrix"),
    );
    let before = block(extract_before_builtin(
        matrix,
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(IntValue::U8(1))]),
    ))
    .expect_err("strict char matrix gate");
    assert_eq!(
        before.identifier(),
        EXTRACT_BEFORE_CHAR_MATRIX_EXTENSION.error_identifier
    );
    let resident = Value::GpuTensor(
        runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .expect("admit unregistered resident fixture")
            .handle,
    );
    let after = block(extract_after_builtin(
        Value::String("abc".into()),
        crate::builtins::common::test_support::builtin_arguments(vec![resident]),
    ))
    .expect_err("strict resident gate");
    assert_eq!(
        after.identifier(),
        EXTRACT_AFTER_RESIDENT_POSITION_EXTENSION.error_identifier
    );
}

#[cfg(target_pointer_width = "64")]
#[test]
fn extract_boundary_preserves_wide_u64_position_without_f64_conversion() {
    let position = (1_u64 << 53) + 1;
    let boundaries =
        BoundaryList::from_value(&Value::Int(IntValue::U64(position)), "extractBefore").unwrap();
    assert!(matches!(
        boundaries.data.as_slice(),
        [Boundary::Position(value)] if *value == position as usize
    ));
}

#[test]
fn extract_before_and_after_reject_extra_arguments() {
    assert!(block(extract_before_builtin(
        Value::String("abc".into()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(2.0),
            Value::Num(3.0)
        ]),
    ))
    .is_err());
    assert!(block(extract_after_builtin(
        Value::String("abc".into()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(2.0),
            Value::Num(3.0)
        ]),
    ))
    .is_err());
}

#[test]
fn extract_before_and_after_reject_zero_floating_positions() {
    for position in [
        Value::Num(0.0),
        Value::Tensor(Tensor::new(vec![0.0], vec![1, 1]).unwrap()),
        Value::Tensor(Tensor::from_f32(vec![0.0], vec![1, 1]).unwrap()),
    ] {
        let before = block(extract_before_builtin(
            Value::String("abc".into()),
            crate::builtins::common::test_support::builtin_arguments(vec![position.clone()]),
        ))
        .expect_err("zero extractBefore position must reject");
        assert_eq!(
            before.identifier(),
            Some("RunMat:extractBefore:InvalidBoundary")
        );
        let after = block(extract_after_builtin(
            Value::String("abc".into()),
            crate::builtins::common::test_support::builtin_arguments(vec![position]),
        ))
        .expect_err("zero extractAfter position must reject");
        assert_eq!(
            after.identifier(),
            Some("RunMat:extractAfter:InvalidBoundary")
        );
    }
}

#[test]
fn extract_before_and_after_accept_empty_character_boundary_without_panicking() {
    let boundary = Value::CharArray(CharArray::from_code_units(Vec::new(), 0, 0).unwrap());
    assert_eq!(
        block(extract_before_builtin(
            Value::String("abc".into()),
            crate::builtins::common::test_support::builtin_arguments(vec![boundary.clone()]),
        ))
        .expect("empty character boundary"),
        Value::String(String::new().into())
    );
    assert_eq!(
        block(extract_after_builtin(
            Value::String("abc".into()),
            crate::builtins::common::test_support::builtin_arguments(vec![boundary]),
        ))
        .expect("empty character boundary"),
        Value::String("abc".into())
    );
}

#[test]
fn extract_before_and_after_classify_unsupported_text_cell_elements_as_invalid_input() {
    let text = Value::Cell(
        CellArray::new(vec![Value::Struct(StructValue::new())], 1, 1).expect("text cell"),
    );
    let before = block(extract_before_builtin(
        text.clone(),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(IntValue::U8(1))]),
    ))
    .expect_err("unsupported extractBefore text cell");
    assert_eq!(
        before.identifier(),
        Some("RunMat:extractBefore:InvalidInput")
    );
    let after = block(extract_after_builtin(
        text,
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(IntValue::U8(1))]),
    ))
    .expect_err("unsupported extractAfter text cell");
    assert_eq!(after.identifier(), Some("RunMat:extractAfter:InvalidInput"));
}

#[test]
fn malformed_text_precedes_resident_position_access() {
    let resident = Value::GpuTensor(
        runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .expect("resident fixture")
            .handle,
    );
    let invalid_text =
        Value::Cell(CellArray::new(vec![Value::Int(IntValue::U8(7))], 1, 1).expect("cell"));
    let before = block(extract_before_builtin(
        invalid_text.clone(),
        crate::builtins::common::test_support::builtin_arguments(vec![resident.clone()]),
    ))
    .expect_err("malformed extraction input");
    assert_eq!(
        before.identifier(),
        Some("RunMat:extractBefore:InvalidInput")
    );
    let after = block(extract_after_builtin(
        invalid_text.clone(),
        crate::builtins::common::test_support::builtin_arguments(vec![resident.clone()]),
    ))
    .expect_err("malformed extraction input");
    assert_eq!(after.identifier(), Some("RunMat:extractAfter:InvalidInput"));
    let insert_source = block(insert_before_builtin(
        invalid_text,
        crate::builtins::common::test_support::builtin_arguments(vec![
            resident.clone(),
            Value::String("x".into()),
        ]),
    ))
    .expect_err("malformed insertion source");
    assert_eq!(
        insert_source.identifier(),
        Some("RunMat:insert:InvalidInput")
    );
    let insert_replacement = block(insert_after_builtin(
        Value::String("abc".into()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            resident,
            Value::Int(IntValue::U8(7)),
        ]),
    ))
    .expect_err("malformed inserted text");
    assert_eq!(
        insert_replacement.identifier(),
        Some("RunMat:insert:InvalidInput")
    );
}

#[test]
fn malformed_resident_boundary_forms_never_reach_provider() {
    let _provider_lock = crate::builtins::common::test_support::accel_test_lock();
    let provider = counted_boundary_provider();
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let resident = resident_position(provider);
    let nested = Value::Cell(CellArray::new(vec![resident.clone()], 1, 1).unwrap());
    let mut malformed_pattern = ObjectInstance::new(runmat_types::standard::PATTERN);
    malformed_pattern.replace_property_unbounded("Regex".into(), Value::String("b".into()));
    malformed_pattern.replace_property_unbounded("Unexpected".into(), resident.clone());

    for extensions_enabled in [false, true] {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(extensions_enabled);
        for boundary in [nested.clone(), Value::Object(malformed_pattern.clone())] {
            let before = block(extract_before_builtin(
                Value::String("abc".into()),
                crate::builtins::common::test_support::builtin_arguments(vec![boundary.clone()]),
            ))
            .expect_err("malformed extractBefore boundary");
            assert_eq!(
                before.identifier(),
                Some("RunMat:extractBefore:InvalidBoundary")
            );
            let after = block(extract_after_builtin(
                Value::String("abc".into()),
                crate::builtins::common::test_support::builtin_arguments(vec![boundary.clone()]),
            ))
            .expect_err("malformed extractAfter boundary");
            assert_eq!(
                after.identifier(),
                Some("RunMat:extractAfter:InvalidBoundary")
            );
            let insert_before = block(insert_before_builtin(
                Value::String("abc".into()),
                crate::builtins::common::test_support::builtin_arguments(vec![
                    boundary.clone(),
                    Value::String("x".into()),
                ]),
            ))
            .expect_err("malformed insertBefore boundary");
            assert_eq!(insert_before.identifier(), INSERT_ERRORS[1].identifier);
            let insert_after = block(insert_after_builtin(
                Value::String("abc".into()),
                crate::builtins::common::test_support::builtin_arguments(vec![
                    boundary,
                    Value::String("x".into()),
                ]),
            ))
            .expect_err("malformed insertAfter boundary");
            assert_eq!(insert_after.identifier(), INSERT_ERRORS[1].identifier);
            assert_eq!(provider.downloads(), 0);
        }
    }
}

#[test]
fn top_level_resident_position_obeys_compatibility_gate_and_gathers_once() {
    let _provider_lock = crate::builtins::common::test_support::accel_test_lock();
    let provider = counted_boundary_provider();
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let resident = resident_position(provider);

    {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let before = block(extract_before_builtin(
            Value::String("abc".into()),
            crate::builtins::common::test_support::builtin_arguments(vec![resident.clone()]),
        ))
        .expect_err("strict extractBefore resident gate");
        assert_eq!(
            before.identifier(),
            EXTRACT_BEFORE_RESIDENT_POSITION_EXTENSION.error_identifier
        );
        let inserted = block(insert_before_builtin(
            Value::String("abc".into()),
            crate::builtins::common::test_support::builtin_arguments(vec![
                resident.clone(),
                Value::String("x".into()),
            ]),
        ))
        .expect_err("strict insertBefore resident gate");
        assert_eq!(
            inserted.identifier(),
            INSERT_BEFORE_RESIDENT_POSITION_EXTENSION.error_identifier
        );
        assert_eq!(provider.downloads(), 0);
    }

    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    assert_eq!(
        block(extract_before_builtin(
            Value::String("abc".into()),
            crate::builtins::common::test_support::builtin_arguments(vec![resident.clone()]),
        ))
        .expect("resident extractBefore"),
        Value::String("a".into())
    );
    assert_eq!(provider.downloads(), 1);
    assert_eq!(
        block(insert_before_builtin(
            Value::String("abc".into()),
            crate::builtins::common::test_support::builtin_arguments(vec![
                resident,
                Value::String("x".into())
            ]),
        ))
        .expect("resident insertBefore"),
        Value::String("axbc".into())
    );
    assert_eq!(provider.downloads(), 2);
}

#[test]
fn deblank_rejects_every_integer_class_without_text_conversion() {
    for value in [
        IntValue::I8(-1),
        IntValue::I16(-1),
        IntValue::I32(-1),
        IntValue::I64(-1),
        IntValue::U8(1),
        IntValue::U16(1),
        IntValue::U32(1),
        IntValue::U64(u64::MAX),
    ] {
        assert!(block(deblank_builtin(Value::Int(value))).is_err());
    }
    assert!(block(deblank_builtin(Value::Tensor(
        runmat_value::Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]).unwrap()
    )))
    .is_err());
    let resident = Value::GpuTensor(
        runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .expect("admit unregistered resident fixture")
            .handle,
    );
    let nested = Value::Cell(CellArray::new(vec![resident], 1, 1).unwrap());
    let err = block(deblank_builtin(nested)).unwrap_err();
    assert!(!err.to_string().contains("provider"));
}
