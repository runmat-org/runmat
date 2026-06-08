//! MATLAB-compatible `argsort` builtin returning permutation indices.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;

use super::sort;
use super::type_resolvers::index_output_type;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::sorting_sets::argsort")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "argsort",
    op_kind: GpuOpKind::Custom("sort"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("sort_dim")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: true,
    notes: "Shares provider hooks with `sort`; when unavailable tensors are gathered to host memory before computing indices.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::array::sorting_sets::argsort"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "argsort",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "`argsort` breaks fusion chains and acts as a residency sink; upstream tensors are gathered when no GPU sort kernel is provided.",
};

const ARGSORT_OUTPUT_I: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "I",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "One-based permutation indices that sort each slice.",
}];

const ARGSORT_INPUTS_A: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input array.",
}];

const ARGSORT_INPUTS_A_ARG1: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array.",
    },
    BuiltinParamDescriptor {
        name: "arg1",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Dimension selector or direction token.",
    },
];

const ARGSORT_INPUTS_A_ARG1_ARG2: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array.",
    },
    BuiltinParamDescriptor {
        name: "arg1",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Dimension selector, placeholder, or direction token.",
    },
    BuiltinParamDescriptor {
        name: "arg2",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Dimension selector or direction token.",
    },
];

const ARGSORT_INPUTS_COMPARISON_METHOD: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array.",
    },
    BuiltinParamDescriptor {
        name: "arg",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Optional dimension/direction arguments.",
    },
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"ComparisonMethod\""),
        description: "Name-value option key.",
    },
    BuiltinParamDescriptor {
        name: "method",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"auto\""),
        description: "Comparison method: 'auto', 'real', or 'abs'.",
    },
];

const ARGSORT_INPUTS_MISSING_PLACEMENT: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array.",
    },
    BuiltinParamDescriptor {
        name: "arg",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Optional dimension/direction arguments.",
    },
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"MissingPlacement\""),
        description: "Name-value option key.",
    },
    BuiltinParamDescriptor {
        name: "placement",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"auto\""),
        description: "Requested NaN placement option (currently unsupported).",
    },
];

const ARGSORT_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "I = argsort(A)",
        inputs: &ARGSORT_INPUTS_A,
        outputs: &ARGSORT_OUTPUT_I,
    },
    BuiltinSignatureDescriptor {
        label: "I = argsort(A, arg1)",
        inputs: &ARGSORT_INPUTS_A_ARG1,
        outputs: &ARGSORT_OUTPUT_I,
    },
    BuiltinSignatureDescriptor {
        label: "I = argsort(A, arg1, arg2)",
        inputs: &ARGSORT_INPUTS_A_ARG1_ARG2,
        outputs: &ARGSORT_OUTPUT_I,
    },
    BuiltinSignatureDescriptor {
        label: "I = argsort(A, ..., \"ComparisonMethod\", method)",
        inputs: &ARGSORT_INPUTS_COMPARISON_METHOD,
        outputs: &ARGSORT_OUTPUT_I,
    },
    BuiltinSignatureDescriptor {
        label: "I = argsort(A, ..., \"MissingPlacement\", placement)",
        inputs: &ARGSORT_INPUTS_MISSING_PLACEMENT,
        outputs: &ARGSORT_OUTPUT_I,
    },
];

const ARGSORT_ERROR_INVALID_DIMENSION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ARGSORT.INVALID_DIMENSION",
    identifier: Some("RunMat:sort:InvalidDimension"),
    when: "Dimension argument is non-positive, non-integer, or otherwise invalid.",
    message: "sort: invalid dimension argument",
};

const ARGSORT_ERROR_COMPARISON_METHOD_REQUIRES_STRING: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.ARGSORT.COMPARISON_METHOD_REQUIRES_STRING",
        identifier: Some("RunMat:sort:ComparisonMethodRequiresString"),
        when: "ComparisonMethod option value is not string-like.",
        message: "sort: 'ComparisonMethod' requires a string value",
    };

const ARGSORT_ERROR_COMPARISON_METHOD_UNKNOWN: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ARGSORT.COMPARISON_METHOD_UNKNOWN",
    identifier: Some("RunMat:sort:ComparisonMethodUnknown"),
    when: "ComparisonMethod option value is not one of 'auto'/'real'/'abs'.",
    message: "sort: unsupported ComparisonMethod",
};

const ARGSORT_ERROR_MISSINGPLACEMENT_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ARGSORT.MISSINGPLACEMENT_UNSUPPORTED",
    identifier: Some("RunMat:sort:MissingPlacementUnsupported"),
    when: "MissingPlacement option is provided but unsupported.",
    message: "sort: the 'MissingPlacement' option is not supported yet",
};

const ARGSORT_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ARGSORT.INVALID_ARGUMENT",
    identifier: Some("RunMat:sort:InvalidArgument"),
    when: "Parser encounters invalid or unrecognized option/value arguments.",
    message: "sort: invalid argument sequence",
};

const ARGSORT_ERRORS: [BuiltinErrorDescriptor; 5] = [
    ARGSORT_ERROR_INVALID_DIMENSION,
    ARGSORT_ERROR_COMPARISON_METHOD_REQUIRES_STRING,
    ARGSORT_ERROR_COMPARISON_METHOD_UNKNOWN,
    ARGSORT_ERROR_MISSINGPLACEMENT_UNSUPPORTED,
    ARGSORT_ERROR_INVALID_ARGUMENT,
];

pub const ARGSORT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ARGSORT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ARGSORT_ERRORS,
};

#[runtime_builtin(
    name = "argsort",
    category = "array/sorting_sets",
    summary = "Return permutation indices that sort arrays along a dimension.",
    keywords = "argsort,sort,indices,permutation,gpu",
    accel = "sink",
    sink = true,
    type_resolver(index_output_type),
    descriptor(crate::builtins::array::sorting_sets::argsort::ARGSORT_DESCRIPTOR),
    builtin_path = "crate::builtins::array::sorting_sets::argsort"
)]
async fn argsort_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let evaluation = sort::evaluate(value, &rest).await?;
    Ok(evaluation.indices_value())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::index_output_type;
    use super::sort;
    use super::ARGSORT_ERROR_COMPARISON_METHOD_REQUIRES_STRING;
    use super::ARGSORT_ERROR_COMPARISON_METHOD_UNKNOWN;
    use super::ARGSORT_ERROR_INVALID_DIMENSION;
    use super::ARGSORT_ERROR_MISSINGPLACEMENT_UNSUPPORTED;
    use futures::executor::block_on;

    fn argsort_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
        block_on(super::argsort_builtin(value, rest))
    }
    use crate::builtins::common::test_support;
    use runmat_builtins::{ComplexTensor, IntValue, ResolveContext, Tensor, Type, Value};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn argsort_vector_default() {
        let tensor = Tensor::new(vec![4.0, 1.0, 3.0], vec![3, 1]).unwrap();
        let indices = argsort_builtin(Value::Tensor(tensor), Vec::new()).expect("argsort");
        match indices {
            Value::Tensor(t) => {
                assert_eq!(t.data, vec![2.0, 3.0, 1.0]);
                assert_eq!(t.shape, vec![3, 1]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn argsort_type_resolver_indices() {
        assert_eq!(
            index_output_type(&[Type::tensor()], &ResolveContext::new(Vec::new())),
            Type::tensor()
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn argsort_descend_direction() {
        let tensor = Tensor::new(vec![10.0, 4.0, 7.0, 9.0], vec![4, 1]).unwrap();
        let indices =
            argsort_builtin(Value::Tensor(tensor), vec![Value::from("descend")]).expect("argsort");
        match indices {
            Value::Tensor(t) => assert_eq!(t.data, vec![1.0, 4.0, 3.0, 2.0]),
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn argsort_dimension_two() {
        let tensor = Tensor::new(vec![1.0, 6.0, 4.0, 2.0, 3.0, 5.0], vec![2, 3]).unwrap();
        let args = vec![Value::Int(IntValue::I32(2))];
        let indices =
            argsort_builtin(Value::Tensor(tensor.clone()), args.clone()).expect("argsort");
        let expected = futures::executor::block_on(sort::evaluate(Value::Tensor(tensor), &args))
            .expect("sort evaluate")
            .indices_value();
        assert_eq!(indices, expected);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn argsort_absolute_comparison() {
        let tensor = Tensor::new(vec![-8.0, -1.0, 3.0, -2.0], vec![4, 1]).unwrap();
        let indices = argsort_builtin(
            Value::Tensor(tensor),
            vec![Value::from("ComparisonMethod"), Value::from("abs")],
        )
        .expect("argsort");
        match indices {
            Value::Tensor(t) => assert_eq!(t.data, vec![2.0, 4.0, 3.0, 1.0]),
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn argsort_handles_nan_like_sort() {
        let tensor = Tensor::new(vec![f64::NAN, 4.0, 1.0, 2.0], vec![4, 1]).unwrap();
        let indices = argsort_builtin(Value::Tensor(tensor), Vec::new()).expect("argsort");
        match indices {
            Value::Tensor(t) => assert_eq!(t.data, vec![3.0, 4.0, 2.0, 1.0]),
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn argsort_dimension_placeholder_then_dim() {
        let tensor = Tensor::new(vec![1.0, 3.0, 4.0, 2.0], vec![2, 2]).unwrap();
        let placeholder = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
        let args = vec![
            Value::Tensor(placeholder),
            Value::Int(IntValue::I32(2)),
            Value::from("descend"),
        ];
        let indices =
            argsort_builtin(Value::Tensor(tensor.clone()), args.clone()).expect("argsort");
        let expected = futures::executor::block_on(sort::evaluate(Value::Tensor(tensor), &args))
            .expect("sort evaluate")
            .indices_value();
        assert_eq!(indices, expected);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn argsort_dimension_greater_than_ndims_returns_ones() {
        let tensor = Tensor::new(vec![1.0, 3.0, 2.0], vec![3, 1]).unwrap();
        let indices = argsort_builtin(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(5))])
            .expect("argsort");
        match indices {
            Value::Tensor(t) => assert!(t.data.iter().all(|v| (*v - 1.0).abs() < f64::EPSILON)),
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn argsort_dimension_zero_errors() {
        let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let err =
            argsort_builtin(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(0))]).unwrap_err();
        assert_eq!(err.identifier(), ARGSORT_ERROR_INVALID_DIMENSION.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn argsort_invalid_argument_errors() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let err = argsort_builtin(
            Value::Tensor(tensor),
            vec![Value::from("MissingPlacement"), Value::from("auto")],
        )
        .unwrap_err();
        assert_eq!(
            err.identifier(),
            ARGSORT_ERROR_MISSINGPLACEMENT_UNSUPPORTED.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn argsort_invalid_comparison_method_errors() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let err = argsort_builtin(
            Value::Tensor(tensor),
            vec![Value::from("ComparisonMethod"), Value::from("unknown")],
        )
        .unwrap_err();
        assert_eq!(
            err.identifier(),
            ARGSORT_ERROR_COMPARISON_METHOD_UNKNOWN.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn argsort_invalid_comparison_method_value_errors() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let err = argsort_builtin(
            Value::Tensor(tensor),
            vec![
                Value::from("ComparisonMethod"),
                Value::Int(IntValue::I32(1)),
            ],
        )
        .unwrap_err();
        assert_eq!(
            err.identifier(),
            ARGSORT_ERROR_COMPARISON_METHOD_REQUIRES_STRING.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn argsort_stable_with_duplicates() {
        let tensor = Tensor::new(vec![2.0, 2.0, 1.0, 2.0], vec![4, 1]).unwrap();
        let indices = argsort_builtin(Value::Tensor(tensor), Vec::new()).expect("argsort");
        match indices {
            Value::Tensor(t) => assert_eq!(t.data, vec![3.0, 1.0, 2.0, 4.0]),
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn argsort_complex_real_method() {
        let tensor =
            ComplexTensor::new(vec![(1.0, 2.0), (-3.0, 0.5), (1.0, -1.0)], vec![3, 1]).unwrap();
        let indices = argsort_builtin(
            Value::ComplexTensor(tensor),
            vec![Value::from("ComparisonMethod"), Value::from("real")],
        )
        .expect("argsort");
        match indices {
            Value::Tensor(t) => assert_eq!(t.data, vec![2.0, 3.0, 1.0]),
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn argsort_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![3.0, 1.0, 2.0], vec![3, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let indices = argsort_builtin(Value::GpuTensor(handle), Vec::new()).expect("argsort");
            match indices {
                Value::Tensor(t) => assert_eq!(t.data, vec![2.0, 3.0, 1.0]),
                other => panic!("expected tensor result, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn argsort_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![0.0, 5.0, -1.0, 2.0], vec![4, 1]).unwrap();
        let cpu_indices = argsort_builtin(Value::Tensor(tensor.clone()), Vec::new()).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let gpu_handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .expect("upload");
        let gpu_indices = argsort_builtin(Value::GpuTensor(gpu_handle), Vec::new()).unwrap();

        let cpu_tensor = match cpu_indices {
            Value::Tensor(t) => t,
            other => panic!("expected tensor, got {other:?}"),
        };
        let gpu_tensor = match gpu_indices {
            Value::Tensor(t) => t,
            other => panic!("expected tensor, got {other:?}"),
        };
        assert_eq!(gpu_tensor.shape, cpu_tensor.shape);
        assert_eq!(gpu_tensor.data, cpu_tensor.data);
    }
}
