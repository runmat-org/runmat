use super::*;

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
    assert_eq!(chars.data, vec!['A', 'C', 'B', 'D']);
    assert_eq!(chars.to_column_major(), vec!['A', 'B', 'C', 'D']);
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
    assert_eq!(
        records
            .field_names()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["zeta", "alpha"]
    );
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
    assert_eq!(
        empty.field_names().map(String::as_str).collect::<Vec<_>>(),
        ["zeta", "alpha"]
    );
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

    let invalid_filename =
        futures::executor::block_on(super::super::memmapfile_builtin(vec![Value::Num(7.0)]))
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
    #[cfg(not(target_arch = "wasm32"))]
    let root = std::env::temp_dir().join("runmat-memmap-read-cause");
    #[cfg(target_arch = "wasm32")]
    let root = std::path::PathBuf::from("/runmat-memmap-read-cause");
    let provider = Arc::new(super::super::vfs::MemoryFsProvider::with_current_dir(
        root.clone(),
    ));
    let _selected = super::super::vfs::replace_provider(provider);
    super::super::vfs::set_current_dir(&root).unwrap();
    let missing =
        futures::executor::block_on(super::super::memmapfile_builtin(vec![Value::String(
            "missing.bin".into(),
        )]))
        .expect_err("missing selected file");
    assert_eq!(missing.identifier(), Some("RunMat:memmapfile:ReadFailed"));
    assert!(missing.source().is_some());
}

#[cfg(target_pointer_width = "64")]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn wide_repeat_property_is_exact_without_a_large_file() {
    use runmat_value::IntValue;
    use std::sync::Arc;

    let _lock = super::super::vfs::provider_override_lock();
    #[cfg(not(target_arch = "wasm32"))]
    let root = std::env::temp_dir().join("runmat-memmap-wide-repeat");
    #[cfg(target_arch = "wasm32")]
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
    let Value::Object(mapped) =
        futures::executor::block_on(super::super::memmapfile_builtin(vec![
            Value::String("one.bin".into()),
            char_value("Repeat"),
            Value::Int(IntValue::U64(wide)),
        ]))
        .unwrap()
    else {
        panic!("memmapfile object");
    };
    assert_eq!(
        mapped.properties.get("Repeat"),
        Some(&Value::Int(IntValue::U64(wide)))
    );
    let Value::Tensor(data) = mapped.properties.get("Data").unwrap() else {
        panic!("numeric payload");
    };
    assert_eq!(data.shape, vec![1, 1]);
}
