//! Actual compiled source oracle for the held Add -> direct SQRT requests.
use super::*;

pub(super) fn require_selected_operations(
    invocation: &ImageSourceInvocation,
    session: &ImageNormalizeStageSession,
    plan: &crate::fusion::FusionGroupPlan,
    values: &[Value],
    source: &crate::fusion::image_reduction::ImageSourceTranscript,
) {
    let moments = invocation
        .requested_staged_workload()
        .unwrap()
        .moment_program
        .as_ref()
        .unwrap();
    let sigma_operations: crate::fusion::image_reduction::ImageSelectedSigmaOperations = invocation
        .selected_sigma_operations()
        .expect("sealed original Add and SQRT with typed epsilon");
    let actual_sqrt = source
        .steps
        .iter()
        .find(|step| u64::from(step.output) == moments.sigma.raw_output)
        .unwrap();
    let actual_add = source
        .steps
        .iter()
        .find(|step| step.output == actual_sqrt.inputs[0])
        .unwrap();
    assert_eq!(
        sigma_operations.source_ids(),
        (
            moments.variance.last().unwrap().public_output,
            u64::from(actual_add.inputs[1]),
            u64::from(actual_add.output),
            u64::from(actual_sqrt.output),
        )
    );
    let exact_control = plan
        .inputs
        .iter()
        .position(|id| *id == actual_add.inputs[1])
        .and_then(|position| values.get(position))
        .or_else(|| plan.const_values.get(&actual_add.inputs[1]))
        .unwrap();
    let represented_control = match exact_control {
        Value::Num(value) => runmat_value::NumericScalar::F64(*value),
        Value::Tensor(tensor) if tensor.len() == 1 => tensor.numeric_value_at(0).unwrap(),
        _ => panic!("compiled epsilon must be numeric scalar"),
    };
    let expected_control = match represented_control {
        runmat_value::NumericScalar::F32(value) => {
            runmat_accelerate_api::scalar_program::ScalarLiteral::F32Bits(value.to_bits())
        }
        runmat_value::NumericScalar::F64(value) => {
            runmat_accelerate_api::scalar_program::ScalarLiteral::F64Bits(value.to_bits())
        }
        _ => panic!("compiled epsilon must be floating scalar"),
    };
    assert_eq!(sigma_operations.control(), expected_control);
    assert_eq!(
        sigma_operations.control_digest(),
        crate::fusion::semantics::exact_constant_digest(exact_control).unwrap()
    );
    assert_eq!(
        sigma_operations.mode(),
        runmat_types::RealFloatingAddMode::Single
    );
    assert_eq!(
        sigma_operations.identity().unwrap(),
        invocation
            .selected_sigma_operations()
            .unwrap()
            .identity()
            .unwrap()
    );
    let final_variance = session
        .selected_mean_plans
        .as_ref()
        .unwrap()
        .variance
        .last()
        .unwrap();
    let (carrier, carrier_bytes) = if let Some(cast) = final_variance.casts.last() {
        (cast.output.clone(), cast.output_bytes)
    } else {
        let step = final_variance.recipe.steps().last().unwrap();
        (
            runmat_accelerate_api::ProviderRepresentation {
                element_type: match step.output_precision() {
                    ProviderPrecision::F32 => runmat_accelerate_api::ProviderElementType::F32,
                    ProviderPrecision::F64 => runmat_accelerate_api::ProviderElementType::F64,
                },
                storage: runmat_accelerate_api::ProviderStorage::DenseReal,
                layout: runmat_accelerate_api::ProviderLayout::ColumnMajorContiguous,
                residency: runmat_accelerate_api::ProviderResidency::Device,
                shape: step
                    .output_shape()
                    .iter()
                    .map(|&extent| extent as u64)
                    .collect(),
            },
            step.output_bytes(),
        )
    };
    let add_request = sigma_operations
        .add_request(&carrier, carrier_bytes)
        .unwrap();
    assert_eq!(
        add_request.operation,
        runmat_accelerate_api::ProviderRealElementwiseOperation::ScalarAdd {
            control: expected_control,
            mode: runmat_types::RealFloatingAddMode::Single,
        }
    );
    assert_eq!(
        sigma_operations
            .sqrt_request(&carrier, carrier_bytes)
            .unwrap()
            .operation,
        runmat_accelerate_api::ProviderRealElementwiseOperation::Sqrt
    );
    let mut wrong_carrier = carrier.clone();
    wrong_carrier.shape[0] += 1;
    assert!(sigma_operations
        .add_request(&wrong_carrier, carrier_bytes)
        .is_none());
    wrong_carrier = carrier.clone();
    wrong_carrier.element_type = runmat_accelerate_api::ProviderElementType::F64;
    assert!(sigma_operations
        .sqrt_request(&wrong_carrier, carrier_bytes)
        .is_none());
    wrong_carrier = carrier.clone();
    wrong_carrier.layout = runmat_accelerate_api::ProviderLayout::Strided;
    assert!(sigma_operations
        .add_request(&wrong_carrier, carrier_bytes)
        .is_none());
    assert!(sigma_operations.add_request(&carrier, 0).is_none());
}
