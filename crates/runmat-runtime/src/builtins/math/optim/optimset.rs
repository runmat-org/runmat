//! Minimal MATLAB-compatible `optimset` options struct builder.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_builtins::{StructValue, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::math::optim::common::{canonical_option_name, field_name};
use crate::builtins::math::optim::type_resolvers::optim_options_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "optimset";

const INTEGER_OPTION_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "optimset-integer-option",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "optimset with native-class integer option payloads is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:OptimsetIntegerOptionExtension"),
};
const RESIDENT_OPTION_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "optimset-resident-option",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "optimset preserving explicit gpuArray option payloads is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:OptimsetResidentOptionExtension"),
};
pub const EXTENSIONS: [BuiltinExtensionDescriptor; 2] =
    [INTEGER_OPTION_EXTENSION, RESIDENT_OPTION_EXTENSION];

const INTEGER_OPTION_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "option value",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Documented numeric optimset options use single or double; RunMat can preserve exact native integer payloads in its struct extension.",
}];
pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor { form: "options = optimset(___, name, integer_value, ___)", inputs: &INTEGER_OPTION_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::PreserveInput, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "Typed integer payloads are gated and retained exactly in the RunMat options struct; automatic resident payloads gather while explicit resident preservation is separately gated." }];

const OPTIMSET_OUTPUT_OPTIONS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "options",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Options struct for optimization solvers.",
}];

const OPTIMSET_INPUTS_PAIRS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Option field name.",
    },
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Option value.",
    },
];

const OPTIMSET_INPUTS_EXISTING_AND_PAIRS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "oldopts",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Existing options struct to update.",
    },
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Option field name.",
    },
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Option value(s) and additional name/value pairs.",
    },
];

const OPTIMSET_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "options = optimset()",
        inputs: &[],
        outputs: &OPTIMSET_OUTPUT_OPTIONS,
    },
    BuiltinSignatureDescriptor {
        label: "options = optimset(name, value, ...)",
        inputs: &OPTIMSET_INPUTS_PAIRS,
        outputs: &OPTIMSET_OUTPUT_OPTIONS,
    },
    BuiltinSignatureDescriptor {
        label: "options = optimset(oldopts, name, value, ...)",
        inputs: &OPTIMSET_INPUTS_EXISTING_AND_PAIRS,
        outputs: &OPTIMSET_OUTPUT_OPTIONS,
    },
];

const OPTIMSET_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OPTIMSET.INVALID_ARGUMENT",
    identifier: Some("RunMat:optimset:InvalidArgument"),
    when: "Name/value argument grammar is invalid.",
    message: "optimset: invalid argument",
};

const OPTIMSET_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OPTIMSET.INVALID_INPUT",
    identifier: Some("RunMat:optimset:InvalidInput"),
    when: "Option field names are not valid string scalars.",
    message: "optimset: invalid input",
};

const OPTIMSET_ERRORS: [BuiltinErrorDescriptor; 2] = [
    OPTIMSET_ERROR_INVALID_ARGUMENT,
    OPTIMSET_ERROR_INVALID_INPUT,
];

pub const OPTIMSET_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OPTIMSET_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &OPTIMSET_ERRORS,
};

fn optimset_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let detail = detail.as_ref();
    let message = if detail.starts_with("optimset:") {
        detail.to_string()
    } else {
        format!("{}: {detail}", error.message)
    };
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::optim::optimset")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "optimset",
    op_kind: GpuOpKind::Custom("options"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::InheritInputs,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host metadata construction. Automatic resident option payloads gather transparently; explicitly resident payload preservation is a gated RunMat extension.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::optim::optimset")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "optimset",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Option struct construction is host metadata work and does not fuse.",
};

#[runtime_builtin(
    name = "optimset",
    category = "math/optim",
    summary = "Create or update optimization options structures.",
    keywords = "optimset,options,TolX,TolFun,MaxIter,Display",
    type_resolver(optim_options_type),
    descriptor(crate::builtins::math::optim::optimset::OPTIMSET_DESCRIPTOR),
    extensions(crate::builtins::math::optim::optimset::EXTENSIONS),
    integer_capabilities(crate::builtins::math::optim::optimset::INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::math::optim::optimset"
)]
async fn optimset_builtin(rest: Vec<Value>) -> BuiltinResult<Value> {
    ensure_optimset_extensions(&rest)?;
    let mut fields = StructValue::new();
    let mut args = rest.into_iter();

    if let Some(first) = args.next() {
        match first {
            Value::Struct(existing) => {
                for (name, value) in existing.fields {
                    fields.insert(name, prepare_optimset_payload(value).await?);
                }
            }
            other => {
                let second = args.next().ok_or_else(|| {
                    optimset_error_with_detail(
                        &OPTIMSET_ERROR_INVALID_ARGUMENT,
                        "expected option name/value pairs",
                    )
                })?;
                let name = field_name(&other).map_err(|err| {
                    optimset_error_with_detail(&OPTIMSET_ERROR_INVALID_INPUT, err.message())
                })?;
                fields.insert(
                    canonical_option_name(&name),
                    prepare_optimset_payload(second).await?,
                );
            }
        }
    }

    let remaining = args.collect::<Vec<_>>();
    if remaining.len() % 2 != 0 {
        return Err(optimset_error_with_detail(
            &OPTIMSET_ERROR_INVALID_ARGUMENT,
            "expected option name/value pairs",
        ));
    }
    for pair in remaining.chunks(2) {
        let name = field_name(&pair[0]).map_err(|err| {
            optimset_error_with_detail(&OPTIMSET_ERROR_INVALID_INPUT, err.message())
        })?;
        fields.insert(
            canonical_option_name(&name),
            prepare_optimset_payload(pair[1].clone()).await?,
        );
    }

    Ok(Value::Struct(fields))
}

fn ensure_optimset_extensions(args: &[Value]) -> BuiltinResult<()> {
    let mut payloads = Vec::new();
    let mut pair_start = 0usize;
    if let Some(Value::Struct(existing)) = args.first() {
        payloads.extend(existing.fields.values());
        pair_start = 1;
    }
    let mut index = pair_start + 1;
    while index < args.len() {
        payloads.push(&args[index]);
        index += 2;
    }
    let integer = payloads.iter().any(|value| {
        crate::builtins::common::validation::value_contains_native_integer_class(value)
    });
    if integer {
        crate::compatibility::ensure_builtin_extension_enabled(&INTEGER_OPTION_EXTENSION, NAME)?;
    }
    let explicit = payloads
        .iter()
        .any(|value| crate::builtins::common::validation::value_contains_explicit_gpu(value));
    if explicit {
        crate::compatibility::ensure_builtin_extension_enabled(&RESIDENT_OPTION_EXTENSION, NAME)?;
    }
    Ok(())
}

async fn prepare_optimset_payload(value: Value) -> BuiltinResult<Value> {
    if crate::builtins::common::validation::value_contains_explicit_gpu(&value) {
        return Ok(value);
    }
    crate::dispatcher::gather_if_needed_async(&value)
        .await
        .map_err(|error| {
            optimset_error_with_detail(
                &OPTIMSET_ERROR_INVALID_INPUT,
                format!("failed to gather option value: {error}"),
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{IntegerStorage, Tensor};

    #[test]
    fn optimset_builds_struct_from_pairs() {
        let value = block_on(optimset_builtin(vec![
            Value::from("TolX"),
            Value::Num(1.0e-8),
            Value::from("Display"),
            Value::from("off"),
        ]))
        .unwrap();
        match value {
            Value::Struct(options) => {
                assert!(matches!(options.fields.get("TolX"), Some(Value::Num(_))));
                assert!(matches!(
                    options.fields.get("Display"),
                    Some(Value::String(_))
                ));
            }
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn optimset_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = OPTIMSET_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert_eq!(
            labels,
            vec![
                "options = optimset()",
                "options = optimset(name, value, ...)",
                "options = optimset(oldopts, name, value, ...)",
            ]
        );

        let codes: Vec<&str> = OPTIMSET_DESCRIPTOR
            .errors
            .iter()
            .map(|error| error.code)
            .collect();
        assert_eq!(
            codes,
            vec!["RM.OPTIMSET.INVALID_ARGUMENT", "RM.OPTIMSET.INVALID_INPUT"]
        );
    }

    #[test]
    fn optimset_odd_name_value_pairs_use_stable_identifier() {
        let err = block_on(optimset_builtin(vec![Value::from("TolX")])).unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:optimset:InvalidArgument"));
    }

    #[test]
    fn optimset_strict_mode_rejects_integer_option_payload() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let value = Tensor::new_integer(IntegerStorage::I64(vec![17]), vec![1, 1]).unwrap();

        let error = block_on(optimset_builtin(vec![
            Value::from("MaxIter"),
            Value::Tensor(value),
        ]))
        .expect_err("integer payload is a RunMat-only extension");

        assert_eq!(
            error.identifier(),
            INTEGER_OPTION_EXTENSION.error_identifier
        );
    }

    #[test]
    fn optimset_runmat_mode_preserves_wide_integer_payload_exactly() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let expected = u64::MAX;
        let value = Tensor::new_integer(IntegerStorage::U64(vec![expected]), vec![1, 1]).unwrap();

        let result = block_on(optimset_builtin(vec![
            Value::from("MaxIter"),
            Value::Tensor(value),
        ]))
        .expect("RunMat integer payload");

        let Value::Struct(options) = result else {
            panic!("expected options struct")
        };
        let Some(Value::Tensor(stored)) = options.fields.get("MaxIter") else {
            panic!("expected exact MaxIter tensor")
        };
        assert!(matches!(
            stored.numeric_value_at(0),
            Some(runmat_builtins::NumericScalar::U64(value)) if value == expected
        ));
    }

    #[test]
    fn optimset_automatic_resident_payload_gathers_but_explicit_payload_is_gated() {
        test_support::with_test_provider(|provider| {
            let values = [3.5];
            let shape = [1, 1];
            let automatic = provider
                .upload(&HostTensorView {
                    data: &values,
                    shape: &shape,
                })
                .expect("automatic upload");
            let automatic =
                automatic.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic);

            let automatic_result = block_on(optimset_builtin(vec![
                Value::from("TolX"),
                Value::GpuTensor(automatic),
            ]))
            .expect("automatic residency gathers");
            assert!(matches!(
                automatic_result,
                Value::Struct(ref options)
                    if matches!(options.fields.get("TolX"), Some(Value::Tensor(tensor)) if tensor.materialize_f64() == &[3.5])
            ));

            let explicit = provider
                .upload(&HostTensorView {
                    data: &values,
                    shape: &shape,
                })
                .expect("explicit upload");
            let explicit =
                explicit.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);
            let strict = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = block_on(optimset_builtin(vec![
                Value::from("TolX"),
                Value::GpuTensor(explicit.clone()),
            ]))
            .expect_err("explicit payload must be gated");
            assert_eq!(
                error.identifier(),
                RESIDENT_OPTION_EXTENSION.error_identifier
            );
            drop(strict);

            let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
            let mixed_automatic = provider
                .upload(&HostTensorView {
                    data: &values,
                    shape: &shape,
                })
                .expect("mixed automatic upload");
            let mixed_automatic = mixed_automatic
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic);
            let explicit_result = block_on(optimset_builtin(vec![
                Value::from("TolX"),
                Value::GpuTensor(explicit),
                Value::from("MaxFunEvals"),
                Value::GpuTensor(mixed_automatic),
            ]))
            .expect("RunMat preserves explicit and gathers automatic payloads independently");
            assert!(matches!(
                explicit_result,
                Value::Struct(ref options)
                    if matches!(options.fields.get("TolX"), Some(Value::GpuTensor(_)))
                        && matches!(options.fields.get("MaxFunEvals"), Some(Value::Tensor(tensor)) if tensor.materialize_f64() == &[3.5])
            ));
        });
    }
}
