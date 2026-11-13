use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn repeat_parser_retains_numeric_infinity_and_exact_finite_counts() {
    assert_eq!(repeat_count(&Value::Num(f64::INFINITY)).unwrap(), None);
    assert_eq!(repeat_count(&Value::Num(3.0)).unwrap(), Some(3));
    assert_eq!(
        repeat_count(&Value::Int(runmat_value::IntValue::U64(7))).unwrap(),
        Some(7)
    );
    // Preserve the existing empty-map control; it is not an infinity spelling.
    assert_eq!(repeat_count(&Value::Num(0.0)).unwrap(), Some(0));
    for invalid in [
        Value::Num(f64::NEG_INFINITY),
        Value::Num(f64::NAN),
        Value::Num(-1.0),
        Value::Num(1.5),
        Value::String("Inf".into()),
    ] {
        assert!(repeat_count(&invalid).is_err());
    }
}

#[test]
fn unnamed_cell_format_retains_declared_shape_and_cell_form() {
    let shape = Value::Tensor(Tensor::new(vec![2.0, 1.0], vec![1, 2]).unwrap());
    for cells in [
        vec![char_value("uint16")],
        vec![char_value("uint16"), shape],
    ] {
        let value = Value::Cell(CellArray::new(cells.clone(), 1, cells.len()).unwrap());
        let parsed = MemmapFormat::from_value(&value).unwrap();
        let presented = parsed.to_value().unwrap();
        assert_eq!(presented, value);
        assert_eq!(
            MemmapFormat::from_value(&presented).unwrap().shape,
            parsed.shape
        );
    }
}

#[test]
fn character_format_returns_character_class_and_column_major_shape() {
    let format = MemmapFormat {
        dtype: MemmapDataType::Character,
        shape: vec![2, 2],
        field: None,
        additional_fields: Vec::new(),
        presentation: MemmapFormatPresentation::TypeName,
    };
    let Value::CharArray(chars) = format.decode(b"ABCD", Some(1)).unwrap() else {
        panic!("Format char must return character data");
    };
    assert_eq!(chars.shape, vec![2, 2]);
    assert_eq!(chars.data(), runmat_value::encode_utf16_code_units("ACBD"));
    assert_eq!(
        chars.to_column_major(),
        runmat_value::encode_utf16_code_units("ABCD")
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn numeric_format_encodes_rank_one_exact_storage_and_rejects_other_low_rank_geometry() {
    let format = MemmapFormat {
        dtype: MemmapDataType::Numeric(NumericDType::U16),
        shape: vec![2],
        field: None,
        additional_fields: Vec::new(),
        presentation: MemmapFormatPresentation::TypeName,
    };
    let rank_one = Tensor::new_integer(IntegerStorage::U16(vec![0x0102, 0xff00]), vec![2])
        .expect("rank-one uint16");
    assert_eq!(
        format
            .encode(&Value::Tensor(rank_one), 1, None)
            .expect("encode exact little-endian bytes"),
        [0x02, 0x01, 0x00, 0xff]
    );

    for shape in [Vec::new(), vec![2, 1]] {
        let values = vec![7_u16; shape.iter().product::<usize>().max(1)];
        let mismatched =
            Tensor::new_integer(IntegerStorage::U16(values), shape).expect("low-rank tensor");
        let error = format
            .encode(&Value::Tensor(mismatched), 1, None)
            .expect_err("Format geometry must be exact");
        assert_eq!(error.identifier(), Some("RunMat:memmapfile:InvalidFormat"));
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn named_multirecord_format_retains_declared_field_order() {
    let format = MemmapFormat {
        dtype: MemmapDataType::Numeric(NumericDType::U8),
        shape: vec![1, 1],
        field: Some("zeta".into()),
        additional_fields: vec![MemmapField {
            dtype: MemmapDataType::Numeric(NumericDType::U8),
            shape: vec![1, 1],
            name: "alpha".into(),
        }],
        presentation: MemmapFormatPresentation::Cell { columns: 3 },
    };
    let Value::StructArray(records) = format.decode(&[1, 2, 3, 4], Some(2)).unwrap() else {
        panic!("two format records must retain structure array class");
    };
    assert_eq!(records.field_names().collect::<Vec<_>>(), ["zeta", "alpha"]);
    let elements = records.into_elements();
    assert_eq!(elements.len(), 2);
    for (record, zeta, alpha) in [(&elements[0], 1, 2), (&elements[1], 3, 4)] {
        let Some(Value::Tensor(zeta_value)) = record.fields.get("zeta") else {
            panic!("zeta field");
        };
        let Some(Value::Tensor(alpha_value)) = record.fields.get("alpha") else {
            panic!("alpha field");
        };
        assert_eq!(
            zeta_value.integer_storage(),
            Some(&IntegerStorage::U8(vec![zeta]))
        );
        assert_eq!(
            alpha_value.integer_storage(),
            Some(&IntegerStorage::U8(vec![alpha]))
        );
    }
    let Value::StructArray(empty) = format.decode(&[], Some(0)).unwrap() else {
        panic!("empty named format must retain its structure schema");
    };
    assert_eq!(empty.shape(), &[0, 1]);
    assert_eq!(empty.field_names().collect::<Vec<_>>(), ["zeta", "alpha"]);
}

#[test]
fn format_and_filename_errors_keep_owning_identifiers_and_causes() {
    use std::error::Error;

    let invalid_format = MemmapFormat::from_value(&Value::String("unknown-type".into()))
        .expect_err("unsupported format");
    assert_eq!(
        invalid_format.identifier(),
        Some("RunMat:memmapfile:InvalidFormat")
    );

    let invalid_filename = futures::executor::block_on(super::super::memmapfile_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(7.0)]),
    ))
    .expect_err("filename must be text");
    assert_eq!(
        invalid_filename.identifier(),
        Some("RunMat:memmapfile:InvalidFilename")
    );
    assert!(invalid_filename.source().is_some());

    let invalid_option = MemmapOption::from_value(&Value::Num(1.0)).unwrap_err();
    assert_eq!(
        invalid_option.identifier(),
        Some("RunMat:memmapfile:InvalidOption")
    );
    assert!(invalid_option.source().is_some());

    let invalid_control = numeric_usize(&Value::Num(2.5), "Offset").unwrap_err();
    assert_eq!(
        invalid_control.identifier(),
        Some("RunMat:memmapfile:InvalidControl")
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn selected_file_read_failure_retains_provider_cause() {
    use std::error::Error;
    use std::sync::Arc;

    let _lock = super::super::vfs::provider_override_lock();
    let root = std::path::PathBuf::from("/runmat-memmap-read-cause");
    let provider = Arc::new(super::super::vfs::MemoryFsProvider::with_current_dir(
        root.clone(),
    ));
    let _selected = super::super::vfs::replace_provider(provider);
    super::super::vfs::set_current_dir(&root).unwrap();
    let missing = futures::executor::block_on(super::super::memmapfile_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
            "missing.bin".into(),
        )]),
    ))
    .expect_err("missing selected file");
    assert_eq!(missing.identifier(), Some("RunMat:memmapfile:ReadFailed"));
    assert!(missing.source().is_some());
}

#[cfg(target_pointer_width = "64")]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn wide_repeat_property_is_exact_without_a_large_file() {
    let _compatibility = crate::compatibility::push_runmat_extensions_enabled(true);
    use runmat_value::IntValue;
    use std::sync::Arc;

    let _lock = super::super::vfs::provider_override_lock();
    let root = std::path::PathBuf::from("/runmat-memmap-wide-repeat");
    let provider = Arc::new(super::super::vfs::MemoryFsProvider::with_current_dir(
        root.clone(),
    ));
    provider
        .write_project_path(root.join("one.bin"), b"Z")
        .unwrap();
    let _selected = super::super::vfs::replace_provider(provider);
    super::super::vfs::set_current_dir(&root).unwrap();
    let wide = (1_u64 << 53) + 1;
    let Value::Object(mapped) = futures::executor::block_on(super::super::memmapfile_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("one.bin".into()),
            char_value("Repeat"),
            Value::Int(IntValue::U64(wide)),
        ]),
    ))
    .unwrap() else {
        panic!("memmapfile object");
    };
    let property = |name| {
        let path = crate::object::indexing::ObjectSubscriptPath::single(
            crate::object::indexing::ObjectSubscript::member(name),
        )
        .to_standard_substruct_value()
        .unwrap();
        futures::executor::block_on(super::live::subsref(
            None,
            Value::Object(mapped.clone()),
            path,
        ))
        .unwrap()
    };
    assert_eq!(property("Repeat"), Value::Int(IntValue::U64(wide)));
    let data_owner = property("Data");
    let Value::Tensor(data) = data_owner.value() else {
        panic!("numeric payload");
    };
    assert_eq!(data.shape(), &[1, 1]);
}
