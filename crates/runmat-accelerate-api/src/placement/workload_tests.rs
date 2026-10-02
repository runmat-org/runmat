use super::ProviderWorkload;

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
    for units in [0, 1, u64::MAX] {
        workload.scalar_work_units = Some(units);
        let bytes = serde_json::to_vec(&workload).unwrap();
        assert_eq!(
            serde_json::from_slice::<ProviderWorkload>(&bytes).unwrap(),
            workload
        );
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
