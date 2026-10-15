use super::ProviderWorkload;
use crate::ProviderPrecision;

#[test]
fn exact_numeric_download_binds_lazy_view_class_roles_and_one_host_output() {
    use crate::{
        test_support::TestAllocationOwner, GpuTensorLayout, GpuTensorStorage, NumericElementType,
        ProviderElementType, ProviderFeasibilityQuery, ProviderLayout, ProviderOperationFamily,
        ProviderOperationIdentity, ProviderRepresentation, ProviderResidency, ProviderStorage,
        TransposeInfo,
    };
    let source = TestAllocationOwner::new()
        .allocate_zeroed(vec![2, 3], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap()
        .handle;
    let lazy = source
        .with_view(source.view().clone().with_shape(vec![3, 2]).with_layout(
            GpuTensorLayout::Transpose(TransposeInfo {
                base_rows: 2,
                base_cols: 3,
            }),
        ))
        .unwrap();
    let input = ProviderRepresentation {
        element_type: ProviderElementType::F64,
        storage: ProviderStorage::DenseReal,
        layout: ProviderLayout::Strided,
        shape: vec![3, 2],
        residency: ProviderResidency::Device,
    };
    let output = ProviderRepresentation {
        layout: ProviderLayout::ColumnMajorContiguous,
        residency: ProviderResidency::Host,
        ..input.clone()
    };
    let exact = ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new("transfer.download"),
        family: ProviderOperationFamily::Download,
        inputs: vec![input],
        outputs: vec![output],
        workload: ProviderWorkload {
            requested_outputs: Some(1),
            ..ProviderWorkload::default()
        },
    };
    assert!(exact.matches_numeric_download(&lazy));
    let mut wrong = exact.clone();
    wrong.inputs[0].layout = ProviderLayout::ColumnMajorContiguous;
    assert!(!wrong.matches_numeric_download(&lazy));
    wrong = exact.clone();
    wrong.outputs[0].element_type = ProviderElementType::F32;
    assert!(!wrong.matches_numeric_download(&lazy));
    wrong = exact.clone();
    wrong.inputs[0].element_type = ProviderElementType::F32;
    assert!(!wrong.matches_numeric_download(&lazy));
    wrong = exact.clone();
    wrong.inputs[0].storage = ProviderStorage::DenseComplexInterleaved;
    assert!(!wrong.matches_numeric_download(&lazy));
    wrong = exact.clone();
    wrong.outputs[0].shape = vec![2, 3];
    assert!(!wrong.matches_numeric_download(&lazy));
    wrong = exact.clone();
    wrong.outputs[0].residency = ProviderResidency::Device;
    assert!(!wrong.matches_numeric_download(&lazy));
    wrong = exact.clone();
    wrong.inputs.push(wrong.inputs[0].clone());
    assert!(!wrong.matches_numeric_download(&lazy));
    wrong = exact.clone();
    wrong.outputs.push(wrong.outputs[0].clone());
    assert!(!wrong.matches_numeric_download(&lazy));
    wrong = exact.clone();
    wrong.workload.requested_outputs = Some(2);
    assert!(!wrong.matches_numeric_download(&lazy));
    wrong = exact.clone();
    wrong.workload.scalar_work_units = Some(7);
    assert!(!wrong.matches_numeric_download(&lazy));
    wrong = exact.clone();
    wrong.operation = ProviderOperationIdentity::new("transfer.upload");
    assert!(!wrong.matches_numeric_download(&lazy));
}

#[test]
fn bounded_upload_keeps_exact_shape_unclaimed_and_checks_late_shape() {
    use crate::{
        ProviderElementType, ProviderFeasibilityQuery, ProviderLayout, ProviderOperationFamily,
        ProviderOperationIdentity, ProviderPrecision, ProviderRepresentation, ProviderResidency,
        ProviderStorage, ProviderTransferOutputBound,
    };
    let bound = ProviderTransferOutputBound {
        element_type: ProviderElementType::F64,
        storage: ProviderStorage::DenseReal,
        layout: ProviderLayout::ColumnMajorContiguous,
        residency: ProviderResidency::Device,
        max_elements: 8,
    };
    let query = ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new("transfer.upload"),
        family: ProviderOperationFamily::Upload,
        inputs: vec![],
        outputs: vec![],
        workload: ProviderWorkload {
            transfer_output_bounds: Some(vec![bound.clone()]),
            requested_outputs: Some(1),
            ..ProviderWorkload::default()
        },
    };
    let crate::ProviderFeasibility::Rejected { rejection } =
        query.conservative_transfer_feasibility(ProviderPrecision::F64)
    else {
        panic!("generic provider cannot assert physical bounded-upload feasibility");
    };
    assert_eq!(
        rejection.code,
        crate::ProviderRejectionCode::UnknownResourceEstimate
    );
    let mut actual = ProviderRepresentation {
        element_type: ProviderElementType::F64,
        storage: ProviderStorage::DenseReal,
        layout: ProviderLayout::ColumnMajorContiguous,
        shape: vec![1, 7],
        residency: ProviderResidency::Device,
    };
    assert!(bound.admits(&actual));
    actual.shape = vec![1, 9];
    assert!(!bound.admits(&actual));
    let logical = ProviderTransferOutputBound {
        element_type: ProviderElementType::Logical,
        ..bound.clone()
    };
    assert_eq!(logical.max_bytes(), None);
    let decoded: ProviderFeasibilityQuery =
        serde_json::from_slice(&serde_json::to_vec(&query).unwrap()).unwrap();
    assert_eq!(decoded, query);
}

#[test]
fn scalar_work_is_optional_in_older_queries_and_round_trips_exactly() {
    let mut workload: ProviderWorkload =
        serde_json::from_str(r#"{"elements":3,"flops":null,"batch":null}"#).unwrap();
    assert_eq!(workload.scalar_work_units, None);
    assert_eq!(workload.preparation_route, None);
    assert_eq!(workload.fused_elementwise, None);
    for units in [0, 1, u64::MAX] {
        workload.scalar_work_units = Some(units);
        let bytes = serde_json::to_vec(&workload).unwrap();
        assert_eq!(
            serde_json::from_slice::<ProviderWorkload>(&bytes).unwrap(),
            workload
        );
    }
    workload.fused_elementwise = Some(crate::ProviderFusedElementwiseWorkload {
        abi: crate::ProviderFusedElementwiseAbi::BroadcastReal {
            precision: ProviderPrecision::F32,
        },
        host_materialization_elements: 8,
        host_workspace_bytes: 32,
    });
    let encoded = serde_json::to_value(&workload).unwrap();
    assert_eq!(
        serde_json::from_value::<ProviderWorkload>(encoded.clone()).unwrap(),
        workload
    );
    for abi_nested in [false, true] {
        let mut unknown = encoded.clone();
        let target = if abi_nested {
            &mut unknown["fused_elementwise"]["abi"]["BroadcastReal"]
        } else {
            &mut unknown["fused_elementwise"]
        };
        target
            .as_object_mut()
            .unwrap()
            .insert("unrecognized".into(), serde_json::json!(true));
        assert!(serde_json::from_value::<ProviderWorkload>(unknown).is_err());
    }
}

#[test]
fn preparation_route_is_explicit_and_round_trips_without_shader_payload() {
    use crate::{ProviderPrecision, ProviderPreparationProgram, ProviderPreparationRoute};
    assert_eq!(
        ProviderPreparationProgram::Elementwise {
            shader: "private source"
        }
        .route(),
        None
    );
    assert_eq!(ProviderPreparationProgram::Sum.route(), None);
    for target in [ProviderPrecision::F32, ProviderPrecision::F64] {
        let workload = ProviderWorkload {
            preparation_route: ProviderPreparationProgram::FloatingCast { target }.route(),
            ..Default::default()
        };
        assert_eq!(
            workload.preparation_route,
            Some(ProviderPreparationRoute::FloatingCast { target })
        );
        let encoded = serde_json::to_vec(&workload).unwrap();
        assert_eq!(
            serde_json::from_slice::<ProviderWorkload>(&encoded).unwrap(),
            workload
        );
    }
}

#[test]
fn convolution_route_round_trips_without_changing_legacy_workloads() {
    use crate::{ProviderConvOrientation, ProviderConvolutionForm, ProviderConvolutionWorkload};
    let legacy: ProviderWorkload =
        serde_json::from_str(r#"{"elements":3,"flops":null,"batch":null}"#).unwrap();
    assert_eq!(legacy.convolution, None);
    let workload = ProviderWorkload {
        convolution: Some(ProviderConvolutionWorkload {
            form: ProviderConvolutionForm::Vector {
                orientation: ProviderConvOrientation::Column,
            },
            mode: runmat_types::ConvolutionMode::Valid,
            input_backings: None,
        }),
        ..Default::default()
    };
    let encoded = serde_json::to_vec(&workload).unwrap();
    assert_eq!(
        serde_json::from_slice::<ProviderWorkload>(&encoded).unwrap(),
        workload
    );
}

#[test]
fn extrema_route_separates_public_demand_from_physical_two_result_hook() {
    use crate::{
        test_support::TestAllocationOwner, GpuTensorStorage, NumericElementType,
        ProviderExtremaDemand, ProviderExtremaInputBacking, ProviderExtremaKind,
        ProviderExtremaWorkload,
    };

    let legacy: ProviderWorkload =
        serde_json::from_str(r#"{"elements":3,"flops":null,"batch":null}"#).unwrap();
    assert_eq!(legacy.extrema, None);
    let one = ProviderWorkload {
        extrema: Some(ProviderExtremaWorkload {
            kind: ProviderExtremaKind::Minimum,
            axis: 1,
            demand: ProviderExtremaDemand::ValuesOnly,
            input_backing: None,
        }),
        requested_outputs: Some(1),
        ..ProviderWorkload::default()
    };
    let two = ProviderWorkload {
        extrema: Some(ProviderExtremaWorkload {
            kind: ProviderExtremaKind::Maximum,
            axis: 0,
            demand: ProviderExtremaDemand::ValuesAndIndices,
            input_backing: None,
        }),
        requested_outputs: Some(2),
        ..ProviderWorkload::default()
    };
    assert_ne!(one, two);
    for workload in [one, two] {
        let bytes = serde_json::to_vec(&workload).unwrap();
        assert_eq!(
            serde_json::from_slice::<ProviderWorkload>(&bytes).unwrap(),
            workload
        );
    }

    let input = TestAllocationOwner::new()
        .allocate_zeroed(vec![2, 3], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap()
        .handle;
    let live = ProviderExtremaInputBacking::from_full_live_handle(&input)
        .unwrap()
        .expect("full contiguous source");
    assert!(live.matches_live_handle(&input).unwrap());
    assert_eq!(
        live.live_requested_bytes(),
        Some(input.physical().requested_bytes())
    );
    let decoded: ProviderExtremaInputBacking =
        serde_json::from_slice(&serde_json::to_vec(&live).unwrap()).unwrap();
    assert_eq!(decoded.live_requested_bytes(), None);
    assert!(!decoded.matches_live_handle(&input).unwrap());
}

#[test]
fn fspecial_route_keeps_exact_scalar_bits_in_serialized_identity() {
    use crate::{FspecialFilter, FspecialRequest, ProviderFspecialFilter};
    let first = ProviderFspecialFilter::from(&FspecialRequest {
        filter: FspecialFilter::Gaussian {
            rows: 3,
            cols: 5,
            sigma: 1.0,
        },
    });
    let next = ProviderFspecialFilter::from(&FspecialRequest {
        filter: FspecialFilter::Gaussian {
            rows: 3,
            cols: 5,
            sigma: f64::from_bits(1.0_f64.to_bits() + 1),
        },
    });
    assert_ne!(first, next);
    assert_eq!(first.output_shape(), Some([3, 5]));
    let workload = ProviderWorkload {
        fspecial: Some(first),
        ..Default::default()
    };
    let bytes = serde_json::to_vec(&workload).unwrap();
    assert_eq!(
        serde_json::from_slice::<ProviderWorkload>(&bytes).unwrap(),
        workload
    );
}

#[test]
fn imfilter_route_keeps_exact_fill_bits_and_typed_options() {
    use crate::{
        ImfilterMode, ImfilterOptions, ImfilterPadding, ImfilterShape, ProviderImfilterWorkload,
    };
    let options = ImfilterOptions {
        mode: ImfilterMode::Convolution,
        shape: ImfilterShape::Full,
        padding: ImfilterPadding::Circular,
        constant_value: -0.0,
    };
    let workload = ProviderImfilterWorkload::from_options(&options, true, None);
    assert!(workload.matches_options(&options, true));
    assert!(!workload.matches_options(&options, false));
    let mut other = options.clone();
    other.constant_value = 0.0;
    assert!(!workload.matches_options(&other, true));
    let wrapped = ProviderWorkload {
        imfilter: Some(workload),
        ..Default::default()
    };
    let encoded = serde_json::to_vec(&wrapped).unwrap();
    assert_eq!(
        serde_json::from_slice::<ProviderWorkload>(&encoded).unwrap(),
        wrapped
    );
}
