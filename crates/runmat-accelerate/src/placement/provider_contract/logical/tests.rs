use super::*;
use crate::simple_provider::InProcessProvider;
use futures::executor::block_on;
use runmat_accelerate_api::{
    GpuTensorStorage, HostNumericDataView, HostNumericTensorView, ProviderWorkload,
};

fn input(shape: &[u64], element_type: ProviderElementType) -> ProviderRepresentation {
    ProviderRepresentation {
        element_type,
        storage: ProviderStorage::DenseReal,
        layout: ProviderLayout::ColumnMajorContiguous,
        shape: shape.to_vec(),
        residency: ProviderResidency::Host,
    }
}

fn query(operation: Operation) -> ProviderFeasibilityQuery {
    ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new(operation.identity()),
        family: ProviderOperationFamily::Elementwise,
        inputs: vec![input(&[1, 4], ProviderElementType::F64); operation.arity()],
        outputs: Vec::new(),
        workload: ProviderWorkload::default(),
    }
}

fn upload(
    provider: &InProcessProvider,
    data: &[f64],
    shape: &[usize],
) -> runmat_accelerate_api::GpuTensorHandle {
    provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F64(data),
            shape,
            storage: GpuTensorStorage::Real,
        })
        .unwrap()
}

#[test]
fn exact_canonical_queries_match_actual_in_process_logical_hooks() {
    let provider = InProcessProvider::new();
    let left_values = [-2.0, 0.0, 3.0, 4.0];
    let right_values = [0.0, 0.0, 2.0, 5.0];
    let left = upload(&provider, &left_values, &[1, 4]);
    let right = upload(&provider, &right_values, &[1, 4]);
    for &operation in OPERATIONS {
        let query = query(operation);
        assert_eq!(
            runmat_builtins::builtin_id_by_name(operation.identity())
                .unwrap()
                .0,
            query.operation.0
        );
        assert!(provider
            .capability_snapshot()
            .supports_operation(&query.operation, query.family));
        assert!(matches!(
            provider.query_feasibility(&query),
            ProviderFeasibility::Supported { .. }
        ));
        let output = match operation {
            Operation::Cast => {
                let zeros = provider.zeros_like(&left, left.shape()).unwrap();
                let result = block_on(provider.elem_ne(&left, &zeros)).unwrap();
                zeros.release().unwrap();
                result
            }
            Operation::Binary(LogicalBinaryOperator::And) => {
                provider.logical_and(&left, &right).unwrap()
            }
            Operation::Binary(LogicalBinaryOperator::Or) => {
                provider.logical_or(&left, &right).unwrap()
            }
            Operation::Binary(LogicalBinaryOperator::Xor) => {
                provider.logical_xor(&left, &right).unwrap()
            }
            Operation::Compare(RelationalOperator::Equal) => {
                block_on(provider.elem_eq(&left, &right)).unwrap()
            }
            Operation::Compare(RelationalOperator::NotEqual) => {
                block_on(provider.elem_ne(&left, &right)).unwrap()
            }
            Operation::Compare(RelationalOperator::LessThan) => {
                block_on(provider.elem_lt(&left, &right)).unwrap()
            }
            Operation::Compare(RelationalOperator::LessThanOrEqual) => {
                block_on(provider.elem_le(&left, &right)).unwrap()
            }
            Operation::Compare(RelationalOperator::GreaterThan) => {
                block_on(provider.elem_gt(&left, &right)).unwrap()
            }
            Operation::Compare(RelationalOperator::GreaterThanOrEqual) => {
                block_on(provider.elem_ge(&left, &right)).unwrap()
            }
        };
        let expected: Vec<_> = left_values
            .iter()
            .zip(right_values)
            .map(|(&a, b)| {
                let value = match operation {
                    Operation::Cast => a != 0.0,
                    Operation::Binary(LogicalBinaryOperator::And) => a != 0.0 && b != 0.0,
                    Operation::Binary(LogicalBinaryOperator::Or) => a != 0.0 || b != 0.0,
                    Operation::Binary(LogicalBinaryOperator::Xor) => (a != 0.0) != (b != 0.0),
                    Operation::Compare(RelationalOperator::Equal) => a == b,
                    Operation::Compare(RelationalOperator::NotEqual) => a != b,
                    Operation::Compare(RelationalOperator::LessThan) => a < b,
                    Operation::Compare(RelationalOperator::LessThanOrEqual) => a <= b,
                    Operation::Compare(RelationalOperator::GreaterThan) => a > b,
                    Operation::Compare(RelationalOperator::GreaterThanOrEqual) => a >= b,
                };
                if value {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        let downloaded = block_on(provider.download(&output)).unwrap();
        assert_eq!(downloaded.shape, vec![1, 4]);
        assert_eq!(downloaded.data, expected, "{}", operation.identity());
        output.release().unwrap();
    }
    left.release().unwrap();
    right.release().unwrap();
}

#[test]
fn in_process_nd_broadcast_is_preserved() {
    let provider = InProcessProvider::new();
    let mut query = query(Operation::Binary(LogicalBinaryOperator::And));
    query.inputs = vec![
        input(&[1, 2, 2], ProviderElementType::F64),
        input(&[1, 1, 2], ProviderElementType::F64),
    ];
    assert!(matches!(
        provider.query_feasibility(&query),
        ProviderFeasibility::Supported { .. }
    ));
    let left = upload(&provider, &[1.0, 0.0, 2.0, 0.0], &[1, 2, 2]);
    let right = upload(&provider, &[0.0, 1.0], &[1, 1, 2]);
    let output = provider.logical_and(&left, &right).unwrap();
    let downloaded = block_on(provider.download(&output)).unwrap();
    assert_eq!(downloaded.shape, vec![1, 2, 2]);
    assert_eq!(downloaded.data, vec![0.0, 0.0, 1.0, 0.0]);
    for handle in [left, right, output] {
        handle.release().unwrap();
    }
}

#[test]
fn unknown_id_wrong_family_type_and_shape_are_not_admitted() {
    let provider = InProcessProvider::new();
    let baseline = query(Operation::Binary(LogicalBinaryOperator::And));
    let reject = |query: &ProviderFeasibilityQuery| {
        assert!(matches!(
            provider.query_feasibility(query),
            ProviderFeasibility::Rejected { .. }
        ))
    };
    for name in ["not_a_logical_hook", "all", "any"] {
        let mut query = baseline.clone();
        query.operation = ProviderOperationIdentity::new(name);
        reject(&query);
    }
    let mut query = baseline.clone();
    query.family = ProviderOperationFamily::Reduction;
    reject(&query);
    for element_type in [
        ProviderElementType::F32,
        ProviderElementType::I64,
        ProviderElementType::ComplexF64,
        ProviderElementType::Logical,
    ] {
        let mut query = baseline.clone();
        query.inputs[0].element_type = element_type;
        reject(&query);
    }
    let mut query = baseline.clone();
    query.inputs.pop();
    reject(&query);
    let mut query = baseline.clone();
    query.inputs[1].shape = vec![1, 3];
    reject(&query);
    let mut query = baseline.clone();
    query.inputs[0].layout = ProviderLayout::Strided;
    reject(&query);
    let mut query = baseline;
    query.inputs[0].storage = ProviderStorage::Sparse;
    reject(&query);
}

#[cfg(feature = "wgpu")]
#[test]
fn wgpu_profile_keeps_integer_broadcast_and_matching_shape_nd_float_support() {
    let mut query = query(Operation::Compare(RelationalOperator::Equal));
    query.inputs = vec![input(&[1, 2, 2], ProviderElementType::F32); 2];
    assert!(validate(HookProfile::Wgpu, ProviderPrecision::F32, &query).is_none());
    query.inputs[1].shape = vec![1, 1, 2];
    assert!(validate(HookProfile::Wgpu, ProviderPrecision::F32, &query).is_some());
    for input in &mut query.inputs {
        input.element_type = ProviderElementType::I64;
    }
    assert!(validate(HookProfile::Wgpu, ProviderPrecision::F32, &query).is_none());
    query.inputs[1].element_type = ProviderElementType::U64;
    assert!(validate(HookProfile::Wgpu, ProviderPrecision::F32, &query).is_some());
    query.inputs[1].element_type = ProviderElementType::I64;
    query.inputs[0].shape = vec![1; crate::backend::wgpu::params::BCAST_MAX_RANK + 1];
    assert!(validate(HookProfile::Wgpu, ProviderPrecision::F32, &query).is_some());
}
