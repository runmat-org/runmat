use super::super::encode::gif_repeat_from_bytes;
use super::super::materialize::{map_index, value_to_u16, value_to_u8};
use super::*;
use runmat_builtins::IMWRITE_DESCRIPTOR;

#[test]
fn writes_png_rgb_and_round_trips_layout() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("rgb.png");
    let rgb = tensor(
        vec![255.0, 0.0, 0.0, 0.0, 0.0, 255.0],
        vec![1, 2, 3],
        NumericDType::U8,
    );

    call(vec![
        Value::Tensor(rgb),
        Value::from(path.to_string_lossy().as_ref()),
    ])
    .unwrap();

    let decoded = ImageReader::open(&path)
        .unwrap()
        .decode()
        .unwrap()
        .to_rgb8();
    assert_eq!(decoded.dimensions(), (2, 1));
    assert_eq!(decoded.get_pixel(0, 0).0, [255, 0, 0]);
    assert_eq!(decoded.get_pixel(1, 0).0, [0, 0, 255]);
}

#[test]
fn last_write_mode_overwrite_allows_png_after_earlier_append() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("last_mode.png");
    let image = Value::Tensor(tensor(vec![0.5], vec![1, 1], NumericDType::F64));
    call(vec![
        image,
        Value::String(path.to_string_lossy().into_owned().into()),
        Value::String("WriteMode".into()),
        Value::String("append".into()),
        Value::String("WriteMode".into()),
        Value::String("overwrite".into()),
    ])
    .unwrap();
    assert!(path.exists());
}

#[test]
fn direct_rgba_png_preserves_alpha_channel() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("rgba.png");
    let rgba = typed_tensor(IntegerStorage::U8(vec![10, 20, 30, 40]), vec![1, 1, 4]);
    call(vec![
        Value::Tensor(rgba),
        Value::String(path.to_string_lossy().into_owned().into()),
    ])
    .unwrap();
    let decoded = image::open(&path).unwrap().into_rgba8();
    assert_eq!(decoded.get_pixel(0, 0).0, [10, 20, 30, 40]);
}

#[test]
fn writes_typed_integer_png_rgb_from_exact_storage() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("typed-rgb.png");
    let rgb = typed_tensor(
        IntegerStorage::U8(vec![255, 0, 0, 0, 0, 255]),
        vec![1, 2, 3],
    );

    call(vec![
        Value::Tensor(rgb),
        Value::from(path.to_string_lossy().as_ref()),
    ])
    .unwrap();

    let decoded = ImageReader::open(&path)
        .unwrap()
        .decode()
        .unwrap()
        .to_rgb8();
    assert_eq!(decoded.dimensions(), (2, 1));
    assert_eq!(decoded.get_pixel(0, 0).0, [255, 0, 0]);
    assert_eq!(decoded.get_pixel(1, 0).0, [0, 0, 255]);
}

#[test]
fn writes_png_alpha_option() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("alpha.png");
    let image = tensor(vec![1.0, 0.0, 0.0], vec![1, 1, 3], NumericDType::F64);
    let alpha = tensor(vec![0.5], vec![1, 1], NumericDType::F64);

    call(vec![
        Value::Tensor(image),
        Value::from(path.to_string_lossy().as_ref()),
        Value::from("Alpha"),
        Value::Tensor(alpha),
    ])
    .unwrap();

    let decoded = ImageReader::open(&path)
        .unwrap()
        .decode()
        .unwrap()
        .to_rgba8();
    assert_eq!(decoded.get_pixel(0, 0).0, [255, 0, 0, 128]);
}

#[test]
fn writes_typed_integer_png_alpha_from_exact_storage() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("typed-alpha.png");
    let image = tensor(vec![1.0, 0.0, 0.0], vec![1, 1, 3], NumericDType::F64);
    let alpha = typed_tensor(IntegerStorage::U8(vec![128]), vec![1, 1]);

    call(vec![
        Value::Tensor(image),
        Value::from(path.to_string_lossy().as_ref()),
        Value::from("Alpha"),
        Value::Tensor(alpha),
    ])
    .unwrap();

    let decoded = ImageReader::open(&path)
        .unwrap()
        .decode()
        .unwrap()
        .to_rgba8();
    assert_eq!(decoded.get_pixel(0, 0).0, [255, 0, 0, 128]);
}

#[test]
fn imwrite_numeric_scalar_reads_typed_integer_tensor_exactly() {
    let scalar = Tensor::new_integer(
        runmat_value::IntegerStorage::U64(vec![u64::MAX]),
        vec![1, 1],
    )
    .expect("scalar");
    assert!(numeric_scalar(&Value::Tensor(scalar), "LoopCount").is_err());
    assert!(numeric_scalar(
        &Value::Int(runmat_value::IntValue::I64(i64::MAX)),
        "LoopCount"
    )
    .is_err());
    assert_eq!(
        numeric_scalar(
            &Value::Int(runmat_value::IntValue::I64(i64::MIN)),
            "LoopCount"
        )
        .unwrap(),
        i64::MIN as f64
    );
    let largest_exact_u64_below_max = u64::MAX - 2047;
    assert_eq!(
        numeric_scalar(
            &Value::Int(runmat_value::IntValue::U64(largest_exact_u64_below_max)),
            "LoopCount"
        )
        .unwrap(),
        largest_exact_u64_below_max as f64
    );

    let vector = Tensor::new_integer(runmat_value::IntegerStorage::U16(vec![1, 2]), vec![1, 2])
        .expect("vector");
    assert!(numeric_scalar(&Value::Tensor(vector), "LoopCount").is_err());

    for storage in [
        runmat_value::IntegerStorage::I8(vec![1]),
        runmat_value::IntegerStorage::I16(vec![1]),
        runmat_value::IntegerStorage::I32(vec![1]),
        runmat_value::IntegerStorage::I64(vec![1]),
        runmat_value::IntegerStorage::U8(vec![1]),
        runmat_value::IntegerStorage::U16(vec![1]),
        runmat_value::IntegerStorage::U32(vec![1]),
        runmat_value::IntegerStorage::U64(vec![1]),
    ] {
        let scalar = Tensor::new_integer(storage, vec![1, 1]).expect("scalar");
        assert_eq!(
            numeric_scalar(&Value::Tensor(scalar), "LoopCount").unwrap(),
            1.0
        );
    }
}

#[test]
fn integer_image_scaling_is_exact_for_every_native_class() {
    for (minimum, midpoint, maximum) in [
        (
            NumericScalar::I8(i8::MIN),
            NumericScalar::I8(0),
            NumericScalar::I8(i8::MAX),
        ),
        (
            NumericScalar::I16(i16::MIN),
            NumericScalar::I16(0),
            NumericScalar::I16(i16::MAX),
        ),
        (
            NumericScalar::I32(i32::MIN),
            NumericScalar::I32(0),
            NumericScalar::I32(i32::MAX),
        ),
        (
            NumericScalar::I64(i64::MIN),
            NumericScalar::I64(0),
            NumericScalar::I64(i64::MAX),
        ),
        (
            NumericScalar::U8(0),
            NumericScalar::U8(128),
            NumericScalar::U8(u8::MAX),
        ),
        (
            NumericScalar::U16(0),
            NumericScalar::U16(32768),
            NumericScalar::U16(u16::MAX),
        ),
        (
            NumericScalar::U32(0),
            NumericScalar::U32(1 << 31),
            NumericScalar::U32(u32::MAX),
        ),
        (
            NumericScalar::U64(0),
            NumericScalar::U64((1_u64 << 63) + 1),
            NumericScalar::U64(u64::MAX),
        ),
    ] {
        assert_eq!(value_to_u8(minimum), 0);
        assert_eq!(value_to_u8(midpoint), 128);
        assert_eq!(value_to_u8(maximum), u8::MAX);
        assert_eq!(value_to_u16(minimum), 0);
        assert!((32768..=32896).contains(&value_to_u16(midpoint)));
        assert_eq!(value_to_u16(maximum), u16::MAX);
    }
}

#[test]
fn indexed_integer_values_are_checked_without_floating_conversion() {
    assert_eq!(map_index(NumericScalar::U8(0), 2).unwrap(), 0);
    assert_eq!(map_index(NumericScalar::U16(1), 2).unwrap(), 1);
    assert_eq!(map_index(NumericScalar::I64(1), 2).unwrap(), 0);
    assert_eq!(map_index(NumericScalar::U64(2), 2).unwrap(), 1);

    let error = map_index(NumericScalar::U64(u64::MAX), 2).unwrap_err();
    assert!(error.message.contains(&u64::MAX.to_string()));
    assert!(map_index(NumericScalar::I64(i64::MIN), 2).is_err());
}

#[test]
fn writes_uint16_png_without_downcasting() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("gray16.png");
    let image = tensor(
        vec![0.0, 65535.0, 12345.0, 40000.0],
        vec![2, 2],
        NumericDType::U16,
    );

    call(vec![
        Value::Tensor(image),
        Value::from(path.to_string_lossy().as_ref()),
    ])
    .unwrap();

    let decoded = ImageReader::open(&path).unwrap().decode().unwrap();
    let gray = decoded.as_luma16().expect("expected 16-bit grayscale PNG");
    assert_eq!(gray.dimensions(), (2, 2));
    assert_eq!(gray.get_pixel(0, 0).0, [0]);
    assert_eq!(gray.get_pixel(0, 1).0, [65535]);
    assert_eq!(gray.get_pixel(1, 0).0, [12345]);
    assert_eq!(gray.get_pixel(1, 1).0, [40000]);
}

#[test]
fn resident_image_metadata_is_validated_and_preserved_before_file_effect() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        runmat_accelerate::ensure_residency_hooks();
        let dir = tempdir().unwrap();
        let path = dir.path().join("resident.png");
        let image = typed_tensor(IntegerStorage::U8(vec![255, 0, 0]), vec![1, 1, 3]);
        let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &image)
            .expect("upload image");
        let handle = handle
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .expect("publish provenance view");
        runmat_accelerate_api::mark_residency(&handle);
        call(vec![
            Value::GpuTensor(handle.clone()),
            Value::from(path.to_string_lossy().as_ref()),
        ])
        .expect("resident write");
        assert!(path.exists());
        assert!(runmat_accelerate::fusion_residency::is_resident(&handle));
        assert!(runmat_accelerate_api::handle_is_explicit(&handle));
        let bad_path = dir.path().join("bad.png");
        let mut wire = handle.wire_descriptor();
        wire.physical = runmat_accelerate_api::allocation::GpuPhysicalDescriptor::new(
            runmat_accelerate_api::NumericElementType::U16,
            runmat_accelerate_api::GpuTensorStorage::Real,
            3,
            6,
            6,
        )
        .unwrap();
        assert!(
            provider.acquire_live(&wire).is_err(),
            "contradictory image class cannot become an operational input"
        );
        assert!(!bad_path.exists());
    });
}

#[test]
fn single_gif_extension_rejects_before_file_effect_in_matlab_mode() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let dir = tempdir().unwrap();
    let path = dir.path().join("single.gif");
    let image = tensor(vec![0.5], vec![1, 1], NumericDType::F32);
    let err = call(vec![
        Value::Tensor(image),
        Value::from(path.to_string_lossy().as_ref()),
    ])
    .expect_err("single GIF extension must be gated");
    assert_eq!(
        err.identifier(),
        Some("RunMat:compatibility:ImwriteSingleGifTiffExtension")
    );
    assert!(!path.exists());
}

#[test]
fn writes_indexed_gif_with_zero_based_uint8_indices() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("indexed.gif");
    let x = tensor(vec![0.0, 1.0], vec![1, 2], NumericDType::U8);
    let map = tensor(
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        vec![2, 3],
        NumericDType::F64,
    );

    call(vec![
        Value::Tensor(x),
        Value::Tensor(map),
        Value::from(path.to_string_lossy().as_ref()),
    ])
    .unwrap();

    let decoded = ImageReader::open(&path)
        .unwrap()
        .decode()
        .unwrap()
        .to_rgb8();
    assert_eq!(decoded.dimensions(), (2, 1));
    assert_eq!(decoded.get_pixel(0, 0).0, [255, 0, 0]);
    assert_eq!(decoded.get_pixel(1, 0).0, [0, 0, 255]);
}

#[test]
fn writes_typed_integer_indexed_gif_from_exact_storage() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("typed-indexed.gif");
    let x = typed_tensor(IntegerStorage::U8(vec![0, 1]), vec![1, 2]);
    let map = tensor(
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        vec![2, 3],
        NumericDType::F64,
    );

    call(vec![
        Value::Tensor(x),
        Value::Tensor(map),
        Value::from(path.to_string_lossy().as_ref()),
    ])
    .unwrap();

    let decoded = ImageReader::open(&path)
        .unwrap()
        .decode()
        .unwrap()
        .to_rgb8();
    assert_eq!(decoded.dimensions(), (2, 1));
    assert_eq!(decoded.get_pixel(0, 0).0, [255, 0, 0]);
    assert_eq!(decoded.get_pixel(1, 0).0, [0, 0, 255]);
}

#[test]
fn appends_gif_frame() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("animated.gif");
    let first = typed_tensor(IntegerStorage::U8(vec![0]), vec![1, 1]);
    let second = typed_tensor(IntegerStorage::U8(vec![1]), vec![1, 1]);
    let map = tensor(
        vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        vec![2, 3],
        NumericDType::F64,
    );

    call(vec![
        Value::Tensor(first),
        Value::Tensor(map.clone()),
        Value::from(path.to_string_lossy().as_ref()),
        Value::from("LoopCount"),
        Value::Num(f64::INFINITY),
        Value::from("DelayTime"),
        Value::Num(0.25),
    ])
    .unwrap();
    call(vec![
        Value::Tensor(second),
        Value::Tensor(map),
        Value::from(path.to_string_lossy().as_ref()),
        Value::from("WriteMode"),
        Value::from("append"),
        Value::from("DelayTime"),
        Value::Num(0.25),
    ])
    .unwrap();

    let bytes = fs::read(&path).unwrap();
    assert!(matches!(
        gif_repeat_from_bytes(&bytes),
        Some(Repeat::Infinite)
    ));
    let decoder = GifDecoder::new(Cursor::new(bytes)).unwrap();
    let frames = decoder.into_frames().collect_frames().unwrap();
    assert_eq!(frames.len(), 2);
}

#[test]
fn rejects_alpha_for_jpeg() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("bad.jpg");
    let image = tensor(vec![1.0, 0.0, 0.0], vec![1, 1, 3], NumericDType::F64);
    let alpha = tensor(vec![1.0], vec![1, 1], NumericDType::F64);

    let err = call(vec![
        Value::Tensor(image),
        Value::from(path.to_string_lossy().as_ref()),
        Value::from("Alpha"),
        Value::Tensor(alpha),
    ])
    .unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:imwrite:InvalidOption"));
}

#[test]
fn descriptor_has_stable_errors() {
    let codes: Vec<&str> = IMWRITE_DESCRIPTOR
        .errors
        .iter()
        .map(|error| error.code)
        .collect();
    assert!(codes.contains(&"RM.IMWRITE.INVALID_IMAGE"));
    assert!(codes.contains(&"RM.IMWRITE.ENCODE"));
}
