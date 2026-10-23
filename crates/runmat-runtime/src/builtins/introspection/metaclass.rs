//! MATLAB-compatible `metaclass` builtin backed by RunMat class metadata.
pub use runmat_builtins::{METACLASS_DESCRIPTOR, METACLASS_ERROR_TOO_MANY_OUTPUTS};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::introspection::class_identity::class_identity_for_value;
use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::introspection::metaclass")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "metaclass",
    op_kind: GpuOpKind::Custom("metadata"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::InheritInputs,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Metadata-only predicate. RunMat reports explicit gpuArray wrapper metadata and the underlying class for internal automatic residency without gathering buffers.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::introspection::metaclass")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "metaclass",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes:
        "Not eligible for fusion planning; metaclass executes on the host and produces metadata.",
};

#[runtime_builtin(
    name = "metaclass",
    binding_variant = "default",
    builtin_path = "crate::builtins::introspection::metaclass"
)]
fn metaclass_builtin(value: Value) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_bounded_outputs(
        &METACLASS_DESCRIPTOR,
        &METACLASS_ERROR_TOO_MANY_OUTPUTS,
        "metaclass",
    )?;
    let identity = class_identity_for_value(&value);
    Ok(Value::ClassRef(identity.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::METACLASS_INTEGER_CAPABILITIES;
    use runmat_value::{
        CellArray, CharArray, HandleRef, IntValue, Listener, MException, ObjectInstance,
        StringArray, Tensor,
    };

    fn handle_target() -> runmat_gc::GcHandle {
        runmat_gc::gc_allocate(Value::Num(0.0)).expect("gc allocation")
    }

    fn call(value: Value) -> String {
        let Value::ClassRef(class_name) = metaclass_builtin(value).expect("metaclass") else {
            panic!("metaclass should return ClassRef");
        };
        class_name.display_name().to_owned()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn metaclass_reports_object_and_handle_runtime_classes() {
        assert_eq!(
            call(Value::Object(ObjectInstance::new("pkg.Point".to_string()))),
            "pkg.Point"
        );
        assert_eq!(
            call(Value::HandleObject(HandleRef {
                class_name: "pkg.HandlePoint".into(),
                target: handle_target(),
                valid: true,
            })),
            "pkg.HandlePoint"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn metaclass_reports_runtime_classes_for_text_values() {
        assert_eq!(call(Value::String("Point".to_string().into())), "string");
        assert_eq!(call(Value::CharArray(CharArray::new_row("Point"))), "char");
        assert_eq!(
            call(Value::StringArray(
                StringArray::new(vec!["Point".to_string()], vec![1, 1]).expect("string array")
            )),
            "string"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn metaclass_reports_core_runtime_classes() {
        assert_eq!(call(Value::Num(1.0)), "double");
        assert_eq!(call(Value::Bool(true)), "logical");
        assert_eq!(
            call(Value::Cell(
                CellArray::new(vec![Value::Num(1.0)], 1, 1).expect("cell")
            )),
            "cell"
        );
        assert_eq!(call(Value::ClassRef("Point".into())), "meta.class");
        assert_eq!(
            call(Value::MException(MException::new(
                "RunMat:Test".to_string(),
                "failure".to_string()
            ))),
            "MException"
        );
    }

    #[test]
    fn metaclass_reports_every_integer_class_without_payload_conversion() {
        for (value, expected) in [
            (IntValue::I8(i8::MIN), "int8"),
            (IntValue::I16(i16::MIN), "int16"),
            (IntValue::I32(i32::MIN), "int32"),
            (IntValue::I64(i64::MIN), "int64"),
            (IntValue::U8(u8::MAX), "uint8"),
            (IntValue::U16(u16::MAX), "uint16"),
            (IntValue::U32(u32::MAX), "uint32"),
            (IntValue::U64(u64::MAX), "uint64"),
        ] {
            assert_eq!(call(Value::Int(value)), expected);
        }
        assert_eq!(METACLASS_INTEGER_CAPABILITIES.len(), 1);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn metaclass_reports_listener_and_residency_provenance_without_gather() {
        assert_eq!(
            call(Value::Listener(Listener {
                id: 7,
                target: handle_target(),
                target_class_name: "pkg.Target".into(),
                event_name: "Changed".into(),
                callback: handle_target(),
                enabled: true,
                valid: true,
            })),
            "event.listener"
        );

        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).expect("tensor");
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: tensor.shape(),
            };
            let handle = provider.upload(&view).expect("upload");
            let caller_alias = handle.clone();
            let original = handle.wire_descriptor();
            let lease = handle.allocation_lease().downgrade();
            provider.reset_telemetry();
            assert_eq!(call(Value::GpuTensor(handle.clone())), "double");
            let explicit = handle
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
                .expect("explicit immutable view");
            drop(handle);
            let handle = explicit;
            assert_eq!(call(Value::GpuTensor(handle)), "gpuArray");
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
            assert_eq!(caller_alias.wire_descriptor(), original);
            assert!(lease.acquire().is_some());
            drop(caller_alias);
            assert!(lease.acquire().is_none());
        });
    }
}
