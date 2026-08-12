use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

const BUILTIN_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "varargout",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Output value(s) from the selected builtin implementation.",
}];

const BUILTIN_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Builtin function name.",
    },
    BuiltinParamDescriptor {
        name: "varargin",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Builtin function arguments.",
    },
];

const BUILTIN_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "[varargout] = builtin(name, varargin)",
    inputs: &BUILTIN_INPUTS,
    outputs: &BUILTIN_OUTPUT,
}];

const BUILTIN_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "varargin",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight integer classes are passed unchanged when they are valid inputs for the builtin selected by name; mixed-class and scalar-double rules belong to that target.",
    }];

pub const BUILTIN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "[varargout] = builtin(name,varargin) with integer target arguments or results",
        inputs: &BUILTIN_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::FunctionSpecific,
        backend: BuiltinIntegerBackendRule::FunctionSpecific,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "builtin performs no numeric conversion or arithmetic of its own; input admission, output count and class, overflow, shape, and host or provider behavior are inherited from the selected registered builtin.",
    }];

pub const BUILTIN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &BUILTIN_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};

fn parse_substruct_type_and_subs(arg: &Value) -> crate::BuiltinResult<(String, Value)> {
    let Value::Struct(st) = arg else {
        return Err(crate::build_runtime_error(
            "builtin: substruct argument must be a scalar struct",
        )
        .with_builtin("builtin")
        .build());
    };
    let kind_value = st.fields.get("type").ok_or_else(|| {
        crate::build_runtime_error("builtin: substruct argument missing field 'type'")
            .with_builtin("builtin")
            .build()
    })?;
    let kind = String::try_from(kind_value).map_err(|err| {
        crate::build_runtime_error(format!("builtin subsref/subsasgn: {err}"))
            .with_builtin("builtin")
            .build()
    })?;
    let subs = st.fields.get("subs").cloned().ok_or_else(|| {
        crate::build_runtime_error("builtin: substruct argument missing field 'subs'")
            .with_builtin("builtin")
            .build()
    })?;
    Ok((kind, subs))
}

fn builtin_member_field_name(subs: &Value, op: &str) -> crate::BuiltinResult<String> {
    String::try_from(subs).map_err(|err| {
        crate::build_runtime_error(format!("builtin {op}: {err}"))
            .with_builtin("builtin")
            .build()
    })
}

async fn dispatch_builtin_subsref(
    rest: &[Value],
    requested_outputs: usize,
) -> crate::BuiltinResult<Value> {
    if rest.len() != 2 {
        return crate::call_builtin_async_with_outputs("subsref", rest, requested_outputs).await;
    }
    let (kind, subs) = parse_substruct_type_and_subs(&rest[1])?;
    if kind != crate::OBJECT_INDEX_MEMBER {
        return crate::call_builtin_async_with_outputs("subsref", rest, requested_outputs).await;
    }
    let field = builtin_member_field_name(&subs, "subsref")?;
    match crate::call_builtin_async_with_outputs(
        "getfield",
        &[rest[0].clone(), Value::String(field.clone())],
        requested_outputs,
    )
    .await
    {
        Ok(value) => Ok(value),
        Err(err) if err.identifier() == Some("RunMat:getfield:ObjectProperty") => {
            crate::call_builtin_async_with_outputs(
                "call_method",
                &[rest[0].clone(), Value::String(field)],
                requested_outputs,
            )
            .await
        }
        Err(err) => Err(err),
    }
}

async fn dispatch_builtin_subsasgn(
    rest: &[Value],
    requested_outputs: usize,
) -> crate::BuiltinResult<Value> {
    if rest.len() != 3 {
        return crate::call_builtin_async_with_outputs("subsasgn", rest, requested_outputs).await;
    }
    let (kind, subs) = parse_substruct_type_and_subs(&rest[1])?;
    if kind != crate::OBJECT_INDEX_MEMBER {
        return crate::call_builtin_async_with_outputs("subsasgn", rest, requested_outputs).await;
    }
    let field = builtin_member_field_name(&subs, "subsasgn")?;
    crate::call_builtin_async_with_outputs(
        "setfield",
        &[rest[0].clone(), Value::String(field), rest[2].clone()],
        requested_outputs,
    )
    .await
}

#[runtime_builtin(
    name = "builtin",
    category = "introspection",
    summary = "Invoke a builtin implementation by name.",
    keywords = "builtin,dispatch,subsref,subsasgn",
    descriptor(crate::builtins::introspection::builtin::BUILTIN_DESCRIPTOR),
    integer_capabilities(crate::builtins::introspection::builtin::BUILTIN_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::introspection::builtin"
)]
pub async fn builtin_builtin(name: String, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let requested_outputs = crate::output_count::current_output_count().unwrap_or(1);

    if name.eq_ignore_ascii_case("subsref") {
        return dispatch_builtin_subsref(&rest, requested_outputs).await;
    }

    if name.eq_ignore_ascii_case("subsasgn") {
        return dispatch_builtin_subsasgn(&rest, requested_outputs).await;
    }

    crate::call_builtin_async_with_outputs(&name, &rest, requested_outputs).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_value::{IntValue, IntegerStorage, NumericStorage, Tensor};

    fn call(name: &str, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
        block_on(builtin_builtin(name.to_string(), rest))
    }

    #[cfg(feature = "wgpu")]
    fn register_wgpu_provider_available() -> bool {
        runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .is_ok()
            && runmat_accelerate_api::provider().is_some()
    }

    #[test]
    fn builtin_descriptor_declares_dynamic_outputs_and_integer_delegation() {
        assert_eq!(
            BUILTIN_DESCRIPTOR.output_mode,
            BuiltinOutputMode::ByRequestedOutputCount
        );
        assert_eq!(
            BUILTIN_DESCRIPTOR.signatures[0].outputs[0].arity,
            BuiltinParamArity::Variadic
        );
        assert_eq!(BUILTIN_INTEGER_CAPABILITIES.len(), 1);
        assert_eq!(
            BUILTIN_INTEGER_CAPABILITIES[0].backend,
            BuiltinIntegerBackendRule::FunctionSpecific
        );
    }

    #[test]
    fn builtin_plus_preserves_all_exact_integer_classes_and_target_saturation() {
        let cases = [
            (
                IntegerStorage::I8(vec![i8::MAX, -5]),
                IntValue::I8(1),
                IntegerStorage::I8(vec![i8::MAX, -4]),
            ),
            (
                IntegerStorage::I16(vec![i16::MAX, -5]),
                IntValue::I16(1),
                IntegerStorage::I16(vec![i16::MAX, -4]),
            ),
            (
                IntegerStorage::I32(vec![i32::MAX, -5]),
                IntValue::I32(1),
                IntegerStorage::I32(vec![i32::MAX, -4]),
            ),
            (
                IntegerStorage::I64(vec![i64::MAX, -5]),
                IntValue::I64(1),
                IntegerStorage::I64(vec![i64::MAX, -4]),
            ),
            (
                IntegerStorage::U8(vec![u8::MAX, 5]),
                IntValue::U8(1),
                IntegerStorage::U8(vec![u8::MAX, 6]),
            ),
            (
                IntegerStorage::U16(vec![u16::MAX, 5]),
                IntValue::U16(1),
                IntegerStorage::U16(vec![u16::MAX, 6]),
            ),
            (
                IntegerStorage::U32(vec![u32::MAX, 5]),
                IntValue::U32(1),
                IntegerStorage::U32(vec![u32::MAX, 6]),
            ),
            (
                IntegerStorage::U64(vec![u64::MAX, 9_007_199_254_740_993]),
                IntValue::U64(1),
                IntegerStorage::U64(vec![u64::MAX, 9_007_199_254_740_994]),
            ),
        ];
        for (input, scalar, expected) in cases {
            let input = Tensor::new_integer(input, vec![2, 1]).expect("input");
            let result = call("plus", vec![Value::Tensor(input), Value::Int(scalar)])
                .expect("delegated integer plus");
            let Value::Tensor(result) = result else {
                panic!("expected exact integer tensor, got {result:?}");
            };
            assert_eq!(result.integer_storage(), Some(&expected));
        }
    }

    #[test]
    fn builtin_preserves_target_native_single_and_requested_outputs() {
        let left = Tensor::from_f32(vec![1.25, 2.5], vec![2, 1]).expect("left");
        let right = Tensor::from_f32(vec![1.0], vec![1, 1]).expect("right");
        let result = call("plus", vec![Value::Tensor(left), Value::Tensor(right)])
            .expect("delegated single plus");
        let Value::Tensor(result) = result else {
            panic!("expected native single tensor");
        };
        assert_eq!(
            result.into_numeric_storage().expect("single storage"),
            NumericStorage::F32(vec![2.25, 3.5])
        );

        let _outputs = crate::output_count::push_output_count(Some(2));
        let source = Tensor::new_integer(IntegerStorage::I16(vec![0, 7, 0, -3]), vec![2, 2])
            .expect("source");
        let result = call("find", vec![Value::Tensor(source)]).expect("delegated find");
        let Value::OutputList(outputs) = result else {
            panic!("expected two requested outputs, got {result:?}");
        };
        assert_eq!(outputs.len(), 2);
        assert!(outputs
            .iter()
            .all(|value| matches!(value, Value::Tensor(tensor) if tensor.numeric_dtype() == runmat_value::NumericDType::F64)));
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn builtin_inherits_exact_resident_integer_target_behavior() {
        let _guard = crate::builtins::common::test_support::accel_test_lock();
        if !register_wgpu_provider_available() {
            return;
        }
        let provider = runmat_accelerate_api::provider().expect("provider");
        let source = Tensor::new_integer(
            IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
            vec![2, 1],
        )
        .expect("source");
        let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &source)
            .expect("integer upload");
        let result = call(
            "plus",
            vec![Value::GpuTensor(handle), Value::Int(IntValue::U64(1))],
        )
        .expect("delegated resident plus");
        let Value::GpuTensor(result) = result else {
            panic!("plus residency must survive builtin dispatch, got {result:?}");
        };
        let gathered = crate::builtins::common::test_support::gather(Value::GpuTensor(result))
            .expect("gather");
        assert_eq!(
            gathered.integer_storage(),
            Some(&IntegerStorage::U64(vec![9_007_199_254_740_994, u64::MAX]))
        );
    }
}
