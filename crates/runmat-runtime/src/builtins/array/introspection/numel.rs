//! MATLAB-compatible `numel` builtin with GPU-aware semantics for RunMat.

use crate::builtins::common::shape::{value_dimensions, value_numel};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::{build_runtime_error, RuntimeError};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ResolveContext, Type,
};
use runmat_macros::runtime_builtin;
use runmat_value::{Tensor, Value};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::introspection::numel")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "numel",
    op_kind: GpuOpKind::Custom("metadata"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Counts elements using tensor metadata; gathers once only if provider metadata is missing.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::array::introspection::numel"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "numel",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Metadata query; fusion planner treats this builtin as a host scalar.",
};

fn numel_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin("numel").build()
}

fn numel_type(args: &[Type], _context: &ResolveContext) -> Type {
    if args.is_empty() {
        Type::Unknown
    } else {
        Type::Num
    }
}

const NUMEL_DIMENSIONS_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "numel-dimension-selectors",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "numel(A, dim, ...) dimension-selector syntax is a RunMat extension; MATLAB's public numel syntax accepts only A",
    error_identifier: Some("RunMat:compatibility:NumelDimensionSelectorsExtension"),
};

pub const NUMEL_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [NUMEL_DIMENSIONS_EXTENSION];

const NUMEL_INTEGER_DATA_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Every integer scalar and array class is counted from shape metadata; element values are never converted or inspected.",
    }];

const NUMEL_INTEGER_DIMENSION_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "dim",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "RunMat's additional scalar, variadic, and vector dimension selectors parse all eight integer classes exactly as positive structural indices.",
    }];

pub const NUMEL_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "n = numel(integer_A)",
        inputs: &NUMEL_INTEGER_DATA_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "The documented double count is produced only when the exact usize count is representable as binary64; coherent resident shape metadata avoids payload gather.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "n = numel(A, integer_dim, ...)",
        inputs: &NUMEL_INTEGER_DIMENSION_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "This compatibility-gated RunMat extension multiplies selected extents with checked usize arithmetic and then requires an exact double result.",
    },
];

const NUMEL_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "n",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Number of elements selected from input.",
}];

const NUMEL_SIG_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input value to inspect.",
}];

const NUMEL_SIG_DIM_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input value to inspect.",
    },
    BuiltinParamDescriptor {
        name: "dim",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Dimension selectors (scalar or vector forms).",
    },
];

const NUMEL_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "n = numel(A)",
        inputs: &NUMEL_SIG_INPUTS,
        outputs: &NUMEL_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "n = numel(A, dim, ...)",
        inputs: &NUMEL_SIG_DIM_INPUTS,
        outputs: &NUMEL_OUTPUT,
    },
];

const NUMEL_ERRORS: [BuiltinErrorDescriptor; 8] = [
    BuiltinErrorDescriptor {
        code: "RM.NUMEL.DIM_ARG_TYPE",
        identifier: None,
        when: "Dimension arguments are not numeric scalars/vectors.",
        message: "numel: dimension arguments must be numeric scalars or vectors",
    },
    BuiltinErrorDescriptor {
        code: "RM.NUMEL.DIM_VECTOR_EMPTY",
        identifier: None,
        when: "Dimension vector argument has zero elements.",
        message: "numel: dimension vector must contain at least one element",
    },
    BuiltinErrorDescriptor {
        code: "RM.NUMEL.DIM_VECTOR_SHAPE",
        identifier: None,
        when: "Dimension vector argument is not vector-shaped.",
        message: "numel: dimension vector must be a vector of positive integers",
    },
    BuiltinErrorDescriptor {
        code: "RM.NUMEL.DIM_NON_FINITE",
        identifier: None,
        when: "A dimension selector is non-finite.",
        message: "numel: dimension must be finite",
    },
    BuiltinErrorDescriptor {
        code: "RM.NUMEL.DIM_NON_INTEGER",
        identifier: None,
        when: "A dimension selector is non-integer.",
        message: "numel: dimension must be an integer",
    },
    BuiltinErrorDescriptor {
        code: "RM.NUMEL.DIM_LT_ONE",
        identifier: None,
        when: "A dimension selector is less than one.",
        message: "numel: dimension must be >= 1",
    },
    BuiltinErrorDescriptor {
        code: "RM.NUMEL.DIM_LIST_EMPTY",
        identifier: None,
        when: "No dimensions are provided after parsing.",
        message: "numel: dimension list must contain at least one element",
    },
    NUMEL_ERROR_COUNT_NOT_EXACT,
];

const NUMEL_ERROR_COUNT_NOT_EXACT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NUMEL.COUNT_NOT_EXACT_DOUBLE",
    identifier: Some("RunMat:numel:CountNotExactDouble"),
    when: "The element count cannot be represented exactly by the documented double scalar output.",
    message: "numel: result exceeds exact double range",
};

pub const NUMEL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &NUMEL_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &NUMEL_ERRORS,
};

#[runtime_builtin(
    name = "numel",
    category = "array/introspection",
    summary = "Count the number of elements in scalars, vectors, matrices, and N-D arrays.",
    keywords = "numel,number of elements,array length,gpu metadata,dimensions",
    accel = "metadata",
    type_resolver(numel_type),
    descriptor(crate::builtins::array::introspection::numel::NUMEL_DESCRIPTOR),
    extensions(crate::builtins::array::introspection::numel::NUMEL_EXTENSIONS),
    integer_capabilities(crate::builtins::array::introspection::numel::NUMEL_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::array::introspection::numel"
)]
async fn numel_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if rest.is_empty() {
        return Ok(Value::Num(exact_numel_count_as_f64(
            value_numel(&value).await?,
        )?));
    }

    crate::compatibility::ensure_builtin_extension_enabled(&NUMEL_DIMENSIONS_EXTENSION, "numel")?;

    let dims = parse_dimension_args(&rest)?;
    let shape = value_dimensions(&value).await?;

    let mut product = 1usize;
    for dim in dims {
        let extent = dimension_extent(&shape, dim);
        product = product.checked_mul(extent).ok_or_else(|| {
            crate::runtime_descriptor_error_with_detail(
                "numel",
                &NUMEL_ERROR_COUNT_NOT_EXACT,
                "selected dimension product exceeds the host structural range",
            )
        })?;
    }

    Ok(Value::Num(exact_numel_count_as_f64(product)?))
}

fn exact_numel_count_as_f64(count: usize) -> crate::BuiltinResult<f64> {
    if count != 0 {
        let significant_bits = usize::BITS - count.leading_zeros();
        let discarded_bits = significant_bits.saturating_sub(f64::MANTISSA_DIGITS);
        if count.trailing_zeros() < discarded_bits {
            return Err(crate::runtime_descriptor_error_with_detail(
                "numel",
                &NUMEL_ERROR_COUNT_NOT_EXACT,
                format!("element count {count} cannot be represented exactly as double"),
            ));
        }
    }
    Ok(count as f64)
}

fn parse_dimension_args(args: &[Value]) -> crate::BuiltinResult<Vec<usize>> {
    let mut dims = Vec::new();
    for arg in args {
        match arg {
            Value::Int(_) | Value::Num(_) => {
                dims.push(tensor::parse_dimension(arg, "numel").map_err(|e| numel_error(e))?);
            }
            Value::Tensor(t) => {
                ensure_dim_vector(t)?;
                if t.is_empty() {
                    return Err(numel_error(
                        "numel: dimension vector must contain at least one element",
                    ));
                }
                let parsed = match tensor::integer_tensor_dimension_vector(t, "numel", false) {
                    Some(parsed) => parsed.map_err(numel_error)?,
                    None => (0..t.len())
                        .map(|index| parse_dim_scalar(tensor::tensor_value_f64(t, index)))
                        .collect::<crate::BuiltinResult<Vec<_>>>()?,
                };
                dims.extend(parsed);
            }
            _ => {
                return Err(numel_error(
                    "numel: dimension arguments must be numeric scalars or vectors",
                ));
            }
        }
    }
    if dims.is_empty() {
        return Err(numel_error(
            "numel: dimension list must contain at least one element",
        ));
    }
    Ok(dims)
}

fn ensure_dim_vector(t: &Tensor) -> crate::BuiltinResult<()> {
    let non_unit = t.shape.iter().filter(|&&dim| dim > 1).count();
    if non_unit <= 1 {
        Ok(())
    } else {
        Err(numel_error(
            "numel: dimension vector must be a vector of positive integers",
        ))
    }
}

fn parse_dim_scalar(raw: f64) -> crate::BuiltinResult<usize> {
    if !raw.is_finite() {
        return Err(numel_error("numel: dimension must be finite"));
    }
    let rounded = raw.round();
    if (rounded - raw).abs() > f64::EPSILON {
        return Err(numel_error("numel: dimension must be an integer"));
    }
    if rounded < 1.0 {
        return Err(numel_error("numel: dimension must be >= 1"));
    }
    Ok(rounded as usize)
}

fn dimension_extent(dimensions: &[usize], dim: usize) -> usize {
    dimensions.get(dim.saturating_sub(1)).copied().unwrap_or(1)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;

    fn numel_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
        block_on(super::numel_builtin(value, rest))
    }
    use runmat_value::{CellArray, CharArray, IntegerStorage, Tensor};

    #[test]
    fn numel_type_returns_double() {
        assert_eq!(
            numel_type(
                &[Type::Tensor { shape: None }],
                &ResolveContext::new(Vec::new())
            ),
            Type::Num
        );
    }

    #[test]
    fn numel_integer_metadata_and_extension_gate_match_public_syntax() {
        assert_eq!(NUMEL_INTEGER_CAPABILITIES.len(), 2);
        assert_eq!(NUMEL_EXTENSIONS.len(), 1);
        let strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let err = numel_builtin(
            Value::Int(runmat_value::IntValue::U8(1)),
            vec![Value::Num(1.0)],
        )
        .expect_err("dimension syntax must be gated");
        assert_eq!(
            err.identifier(),
            NUMEL_DIMENSIONS_EXTENSION.error_identifier
        );
        drop(strict);
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn numel_double_boundary_rejects_unrepresentable_structural_count() {
        assert_eq!(
            exact_numel_count_as_f64(9_007_199_254_740_992).expect("exact count"),
            9_007_199_254_740_992.0
        );
        let err = exact_numel_count_as_f64(9_007_199_254_740_993)
            .expect_err("count is not exact binary64");
        assert_eq!(err.identifier(), NUMEL_ERROR_COUNT_NOT_EXACT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn numel_scalar_is_one() {
        let result = numel_builtin(Value::Num(42.0), Vec::new()).expect("numel");
        assert_eq!(result, Value::Num(1.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn numel_matrix_counts_elements() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result = numel_builtin(Value::Tensor(tensor), Vec::new()).expect("numel");
        assert_eq!(result, Value::Num(4.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn numel_cell_array_counts_cells() {
        let cells = vec![
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(3.0),
            Value::Num(4.0),
        ];
        let cell_array = CellArray::new(cells, 2, 2).unwrap();
        let result = numel_builtin(Value::Cell(cell_array), Vec::new()).expect("numel");
        assert_eq!(result, Value::Num(4.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn numel_char_array_counts_characters() {
        let chars = CharArray::new("RunMat".chars().collect(), 1, 6).unwrap();
        let result = numel_builtin(Value::CharArray(chars), Vec::new()).expect("numel");
        assert_eq!(result, Value::Num(6.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn numel_selected_dimensions_multiplies_extents() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor = Tensor::new(vec![0.0; 24], vec![2, 3, 4]).unwrap();
        let args = vec![Value::from(1.0), Value::from(2.0)];
        let result = numel_builtin(Value::Tensor(tensor), args).expect("numel");
        assert_eq!(result, Value::Num(6.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn numel_dimension_vector_argument_supported() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor = Tensor::new(vec![0.0; 24], vec![2, 3, 4]).unwrap();
        let dims = Tensor::new(vec![1.0, 3.0], vec![1, 2]).unwrap();
        let result =
            numel_builtin(Value::Tensor(tensor), vec![Value::Tensor(dims)]).expect("numel");
        assert_eq!(result, Value::Num(8.0));
    }

    #[test]
    fn numel_dimension_vector_reads_native_single_storage() {
        let dims = Tensor::from_f32(vec![1.0, 3.0], vec![1, 2]).unwrap();
        let parsed = parse_dimension_args(&[Value::Tensor(dims)]).expect("parse dims");
        assert_eq!(parsed, vec![1, 3]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn numel_dimension_vector_reads_integer_tensor_exactly() {
        let large = 9_007_199_254_740_993_u64;
        let dims =
            Tensor::new_integer(IntegerStorage::U64(vec![1, large]), vec![1, 2]).expect("dims");
        let parsed = parse_dimension_args(&[Value::Tensor(dims)]).expect("parse dims");
        assert_eq!(parsed, vec![1, large as usize]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn numel_gpu_tensor_uses_shape() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0; 12], vec![3, 4]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = numel_builtin(Value::GpuTensor(handle), Vec::new()).expect("numel");
            assert_eq!(result, Value::Num(12.0));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn numel_dimension_must_be_positive_integer() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor = Tensor::new(vec![0.0; 4], vec![2, 2]).unwrap();
        let err = numel_builtin(Value::Tensor(tensor), vec![Value::from(0.0)])
            .expect_err("expected dimension error");
        assert!(
            err.to_string().contains("dimension must be >= 1"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn numel_dimension_vector_requires_vector_shape() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor = Tensor::new(vec![0.0; 8], vec![2, 2, 2]).unwrap();
        let dims = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let err = numel_builtin(Value::Tensor(tensor), vec![Value::Tensor(dims)])
            .expect_err("expected vector shape error");
        assert!(
            err.to_string()
                .contains("dimension vector must be a vector"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn numel_dimension_arguments_must_be_numeric() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor = Tensor::new(vec![0.0; 4], vec![2, 2]).unwrap();
        let err = numel_builtin(Value::Tensor(tensor), vec![Value::from("omitnan")])
            .expect_err("expected numeric argument error");
        assert!(
            err.to_string()
                .contains("dimension arguments must be numeric"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn numel_wgpu_counts_elements() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![0.0; 18], vec![3, 3, 2]).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .expect("wgpu provider")
            .upload(&view)
            .expect("upload");
        let result = numel_builtin(Value::GpuTensor(handle), Vec::new()).expect("numel");
        assert_eq!(result, Value::Num(18.0));
    }
}
