use super::*;

#[test]
fn oversized_or_unbacked_cell_geometry_rejects_before_allocation() {
    assert!(shape_elements(&[usize::MAX, 2]).is_err());
    assert_eq!(
        shape_elements(&[usize::MAX, usize::MAX, 0]).expect("zero extent makes an empty array"),
        0
    );
    let mut payload = Cursor::new(&[][..]);
    let error = parse_cell_array(&mut payload, vec![1_000_000], Endian::Little)
        .expect_err("missing million element headers");
    assert!(error.message().contains("more elements"));
}

#[test]
fn malformed_numeric_geometry_rejects_before_native_storage_decode() {
    let element = TaggedData {
        data_type: MI_DOUBLE,
        data: 7f64.to_le_bytes().to_vec(),
    };
    assert!(ensure_numeric_count(&element, 2).is_err());
    let malformed = TaggedData {
        data_type: MI_DOUBLE,
        data: vec![0; 7],
    };
    assert!(ensure_numeric_count(&malformed, 1).is_err());
    assert!(ensure_numeric_count(&element, 1).is_ok());
}

#[test]
fn sparse_struct_field_table_cannot_imply_unbacked_values() {
    let mut payload = Vec::new();
    payload.extend_from_slice(&MI_INT32.to_le_bytes());
    payload.extend_from_slice(&4u32.to_le_bytes());
    payload.extend_from_slice(&2i32.to_le_bytes());
    payload.extend_from_slice(&[0; 4]);
    payload.extend_from_slice(&MI_INT8.to_le_bytes());
    payload.extend_from_slice(&2u32.to_le_bytes());
    payload.extend_from_slice(b"a\0");
    payload.extend_from_slice(&[0; 6]);
    let mut cursor = Cursor::new(payload.as_slice());
    let error = parse_struct(&mut cursor, vec![2, 1], Endian::Little)
        .expect_err("two values cannot fit absent headers");
    assert!(error.message().contains("more elements"));
}
