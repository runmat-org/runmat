use super::*;

#[test]
fn logical_host_upload_preserves_borrowed_bytes_and_shape_without_numeric_relabeling() {
    let input = MxArray::logical(vec![1, 0, 0, 1], vec![2, 1, 2]).unwrap();
    let MxArrayData::Logical(data) = input.data() else {
        unreachable!()
    };
    let (class, UploadSource::Logical(view)) = host_upload(&input).unwrap() else {
        panic!("logical input must use the logical producer boundary")
    };
    assert_eq!(class, MxClassId::Logical);
    assert_eq!(view.data.as_ptr(), data.as_slice().as_ptr());
    assert_eq!(view.shape, [2, 1, 2]);
    view.validate().unwrap();
}

#[test]
fn logical_transfer_decodes_all_supported_physical_encodings_before_cuda_upload() {
    for data in [
        HostNumericDataOwned::U8(vec![1, 0]),
        HostNumericDataOwned::F32(vec![1.0, 0.0]),
        HostNumericDataOwned::F64(vec![1.0, 0.0]),
    ] {
        let host = host_array_from_download(
            MxClassId::Logical,
            HostNumericTensorOwned {
                data,
                shape: vec![1, 1, 2],
                storage: GpuTensorStorage::Real,
            },
        )
        .unwrap();
        let (_, UploadSource::Logical(view)) = host_upload(&host).unwrap() else {
            unreachable!()
        };
        assert_eq!(view.data, [1, 0]);
        assert_eq!(view.shape, [1, 1, 2]);
    }
    let empty = host_array_from_download(
        MxClassId::Logical,
        HostNumericTensorOwned {
            data: HostNumericDataOwned::F32(vec![]),
            shape: vec![2, 0, 3],
            storage: GpuTensorStorage::Real,
        },
    )
    .unwrap();
    assert_eq!(empty.shape(), [2, 0, 3]);
}

#[test]
fn native_logical_class_rejects_complex_allocation_before_provider_selection() {
    let mut api = MxApi::new(crate::MxApiMode::InterleavedComplex);
    let error = api
        .gpu_create(MxClassId::Logical, vec![2, 1], true, false)
        .unwrap_err();
    assert!(error.contains("logical"), "{error}");
    assert!(
        !error.contains("registered CUDA"),
        "class admission must precede provider access: {error}"
    );
    assert_eq!(
        native_class(MxClassId::Logical).unwrap(),
        NativeDeviceClass::Logical
    );
    assert_eq!(
        native_class(MxClassId::Uint8).unwrap(),
        NativeDeviceClass::Numeric(NumericElementType::U8)
    );
}
