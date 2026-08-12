//! MATLAB-compatible `display` builtin for Command Window variable output.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::console::{record_console_line, ConsoleStream};
use crate::gather_if_needed_async;

use super::disp::{empty_return_value, format_for_display};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::display")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "display",
    op_kind: GpuOpKind::Custom("sink"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Always formats on the CPU; GPU tensors are gathered via the active provider before display.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::display")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "display",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Side-effecting sink; excluded from fusion planning.",
};

const DISPLAY_INPUTS_VALUE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Value to display with a variable-name header.",
}];

const DISPLAY_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "display(X)",
    inputs: &DISPLAY_INPUTS_VALUE,
    outputs: &[],
}];

const DISPLAY_ERROR_ARG_CONFIG: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DISPLAY.ARG_CONFIG",
    identifier: Some("RunMat:display:TooManyInputs"),
    when: "Too many input arguments are passed to display.",
    message: "display: too many input arguments",
};

const DISPLAY_ERROR_GATHER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DISPLAY.GATHER",
    identifier: Some("RunMat:display:GatherFailed"),
    when: "Input value cannot be gathered onto the host for rendering.",
    message: "display: failed to gather value for display",
};
const DISPLAY_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DISPLAY.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:display:TooManyOutputs"),
    when: "An output is requested from display.",
    message: "display does not return output arguments",
};

const DISPLAY_ERRORS: [BuiltinErrorDescriptor; 3] = [
    DISPLAY_ERROR_ARG_CONFIG,
    DISPLAY_ERROR_GATHER,
    DISPLAY_ERROR_TOO_MANY_OUTPUTS,
];

pub const DISPLAY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DISPLAY_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DISPLAY_ERRORS,
};

const DISPLAY_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Every integer class is displayed exactly through the variable-aware display path unless class-specific display or disp dispatch handles the object.",
    }];

pub const DISPLAY_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "display(X)",
        inputs: &DISPLAY_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "display is a no-output host rendering sink with class-method dispatch. Resident integer arrays gather through exact typed provider download before the shared formatter runs.",
    }];

fn display_error(error: &'static BuiltinErrorDescriptor) -> crate::RuntimeError {
    display_error_with(error, error.message)
}

fn display_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> crate::RuntimeError {
    let mut builder = crate::build_runtime_error(message).with_builtin("display");
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "display",
    category = "io",
    summary = "Display values with a variable-name header.",
    keywords = "display,disp,print,object,gpu",
    sink = true,
    accel = "sink",
    suppress_auto_output = true,
    type_resolver(crate::builtins::io::type_resolvers::disp_type),
    descriptor(crate::builtins::io::display::DISPLAY_DESCRIPTOR),
    integer_capabilities(crate::builtins::io::display::DISPLAY_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::io::display"
)]
pub async fn display_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if matches!(crate::output_count::current_output_count(), Some(n) if n > 0) {
        return Err(display_error(&DISPLAY_ERROR_TOO_MANY_OUTPUTS));
    }
    if !rest.is_empty() {
        return Err(display_error(&DISPLAY_ERROR_ARG_CONFIG));
    }

    if dispatch_custom_display_method(&value).await? {
        return Ok(empty_return_value());
    }
    if dispatch_custom_disp_method(&value).await? {
        return Ok(empty_return_value());
    }

    let host_value = gather_if_needed_async(&value)
        .await
        .map_err(|e| display_error_with(&DISPLAY_ERROR_GATHER, format!("display: {e}")))?;
    let lines = format_for_display(&host_value);
    if !lines.is_empty() {
        let body = match display_label() {
            Some(label) => format_display_body(&label, &lines),
            None => lines.join("\n"),
        };
        record_console_line(ConsoleStream::Stdout, body);
    }
    Ok(empty_return_value())
}

async fn dispatch_custom_display_method(value: &Value) -> crate::BuiltinResult<bool> {
    match value {
        Value::Object(_) | Value::HandleObject(_) => {}
        _ => return Ok(false),
    }
    let args = vec![value.clone()];
    crate::dispatcher::try_call_registered_instance_method("display", &args, Some(0))
        .await
        .map(|result| result.is_some())
}

async fn dispatch_custom_disp_method(value: &Value) -> crate::BuiltinResult<bool> {
    match value {
        Value::Object(_) | Value::HandleObject(_) => {}
        _ => return Ok(false),
    }
    let args = vec![value.clone()];
    crate::dispatcher::try_call_registered_instance_method("disp", &args, Some(0))
        .await
        .map(|result| result.is_some())
}

fn display_label() -> Option<String> {
    match crate::callsite::arg_text(0) {
        Some(text) => is_simple_identifier(text.trim()).then(|| text.trim().to_string()),
        None => Some("ans".to_string()),
    }
}

fn is_simple_identifier(text: &str) -> bool {
    let mut chars = text.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    (first == '_' || first.is_ascii_alphabetic())
        && chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
}

fn format_display_body(label: &str, lines: &[String]) -> String {
    match lines {
        [] => format!("{label} ="),
        [single] if !single.contains('\n') => format!("{label} = {single}"),
        _ => format!("{label} =\n{}", lines.join("\n")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::console::{reset_thread_buffer, take_thread_buffer};
    use futures::executor::block_on;
    use runmat_builtins::{ClassDef, MethodDef};
    use runmat_hir::{SourceId, Span};
    use runmat_value::{Access, IntValue, ObjectInstance, Tensor};
    use std::collections::HashMap;
    use std::sync::Arc;

    fn span_of(source: &str, needle: &str) -> Span {
        let start = source.find(needle).expect("needle present");
        Span {
            start,
            end: start + needle.len(),
        }
    }

    fn stdout_text() -> String {
        take_thread_buffer()
            .into_iter()
            .filter(|entry| entry.stream == ConsoleStream::Stdout)
            .map(|entry| entry.text)
            .collect::<String>()
    }

    #[test]
    fn display_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = DISPLAY_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"display(X)"));
        assert!(DISPLAY_DESCRIPTOR.signatures[0].outputs.is_empty());
    }

    #[test]
    fn display_rejects_requested_output() {
        let _outputs = crate::output_count::push_output_count(Some(1));
        let err = block_on(display_builtin(Value::Num(1.0), Vec::new())).unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:display:TooManyOutputs"));
    }

    #[test]
    fn display_retains_typed_empty_representation() {
        reset_thread_buffer();
        block_on(display_builtin(
            Value::Tensor(Tensor::zeros(vec![0, 0])),
            Vec::new(),
        ))
        .unwrap();
        assert_eq!(stdout_text(), "ans = []\n");
    }

    #[test]
    fn display_formats_wide_integer_exactly_under_ans_header() {
        reset_thread_buffer();
        block_on(display_builtin(
            Value::Int(IntValue::U64(u64::MAX)),
            Vec::new(),
        ))
        .expect("display");
        let text = stdout_text();
        assert!(text.contains("ans ="));
        assert!(text.contains(&u64::MAX.to_string()));
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn display_gathers_wgpu_integer_tensor_exactly() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .expect("register wgpu provider");
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let values = [(1_u64 << 53) + 1, u64::MAX];
        let handle = provider
            .upload_integer(&runmat_accelerate_api::HostIntegerTensorView {
                data: runmat_accelerate_api::HostIntegerDataView::U64(&values),
                shape: &[1, 2],
            })
            .expect("upload integer");
        reset_thread_buffer();
        block_on(display_builtin(Value::GpuTensor(handle), Vec::new())).expect("display");
        let text = stdout_text();
        assert!(text.contains("9007199254740993"));
        assert!(text.contains("18446744073709551615"));
    }

    #[test]
    fn display_uses_simple_callsite_variable_name() {
        let source = "display(alpha);";
        let _catalog_guard = crate::source_context::replace_source_catalog(vec![(
            SourceId(81),
            "/tmp/display_name.m".to_string(),
            source.to_string(),
        )]);
        let _callsite_guard = crate::callsite::push_callsite(
            Some(SourceId(81)),
            Some(vec![span_of(source, "alpha")]),
        );

        reset_thread_buffer();
        let result = block_on(display_builtin(Value::Num(42.0), Vec::new())).expect("display");

        assert_eq!(result, Value::Tensor(Tensor::zeros(vec![0, 0])));
        assert_eq!(stdout_text(), "alpha = 42\n");
    }

    #[test]
    fn display_expression_has_no_ans_header() {
        let source = "display(alpha + 1);";
        let _catalog_guard = crate::source_context::replace_source_catalog(vec![(
            SourceId(82),
            "/tmp/display_expr.m".to_string(),
            source.to_string(),
        )]);
        let _callsite_guard = crate::callsite::push_callsite(
            Some(SourceId(82)),
            Some(vec![span_of(source, "alpha + 1")]),
        );

        reset_thread_buffer();
        block_on(display_builtin(Value::Num(43.0), Vec::new())).expect("display");

        assert_eq!(stdout_text(), "43\n");
    }

    #[test]
    fn display_formats_multiline_values_under_header() {
        reset_thread_buffer();
        let tensor = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).expect("tensor");
        block_on(display_builtin(Value::Tensor(tensor), Vec::new())).expect("display");

        assert_eq!(stdout_text(), "ans =\n     1       2\n     3       4\n");
    }

    #[test]
    fn display_dispatches_custom_object_display_method() {
        let class_name = "DisplayHookObject".to_string();
        runmat_builtins::register_class(ClassDef {
            name: class_name.clone(),
            parent: None,
            properties: HashMap::new(),
            methods: HashMap::from([(
                "display".to_string(),
                MethodDef {
                    name: "display".to_string(),
                    is_static: false,
                    is_abstract: false,
                    is_sealed: false,
                    access: Access::Public,
                    function_name: "DisplayHookObject.customDisplayImpl".to_string(),
                    implicit_class_argument: None,
                },
            )]),
        });
        let _resolver =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "DisplayHookObject.customDisplayImpl").then_some(771)
            })));
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |function, args, requested_outputs| {
                let args = args.to_vec();
                Box::pin(async move {
                    assert_eq!(function, 771);
                    assert_eq!(requested_outputs, 0);
                    assert!(
                        matches!(args.first(), Some(Value::Object(object)) if object.class_name == "DisplayHookObject")
                    );
                    record_console_line(ConsoleStream::Stdout, "custom display");
                    Ok(empty_return_value())
                })
            },
        )));

        reset_thread_buffer();
        block_on(display_builtin(
            Value::Object(ObjectInstance::new(class_name)),
            Vec::new(),
        ))
        .expect("display");

        assert_eq!(stdout_text(), "custom display\n");
    }

    #[test]
    fn display_falls_back_to_overloaded_disp_method() {
        let class_name = "DispFallbackObject".to_string();
        runmat_builtins::register_class(ClassDef {
            name: class_name.clone(),
            parent: None,
            properties: HashMap::new(),
            methods: HashMap::from([(
                "disp".to_string(),
                MethodDef {
                    name: "disp".to_string(),
                    is_static: false,
                    is_abstract: false,
                    is_sealed: false,
                    access: Access::Public,
                    function_name: "DispFallbackObject.renderDisp".to_string(),
                    implicit_class_argument: None,
                },
            )]),
        });
        let _resolver =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "DispFallbackObject.renderDisp").then_some(773)
            })));
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |function, args, requested_outputs| {
                let args = args.to_vec();
                Box::pin(async move {
                    assert_eq!(function, 773);
                    assert_eq!(requested_outputs, 0);
                    assert!(
                        matches!(args.first(), Some(Value::Object(object)) if object.class_name == "DispFallbackObject")
                    );
                    record_console_line(ConsoleStream::Stdout, "custom disp");
                    Ok(empty_return_value())
                })
            },
        )));
        reset_thread_buffer();
        block_on(display_builtin(
            Value::Object(ObjectInstance::new(class_name)),
            Vec::new(),
        ))
        .expect("display through disp");
        assert_eq!(stdout_text(), "custom disp\n");
    }

    #[test]
    fn display_dispatches_inherited_custom_object_display_method() {
        let parent_class_name = "DisplayHookParent".to_string();
        let child_class_name = "DisplayHookChild".to_string();
        runmat_builtins::register_class(ClassDef {
            name: parent_class_name.clone(),
            parent: None,
            properties: HashMap::new(),
            methods: HashMap::from([(
                "display".to_string(),
                MethodDef {
                    name: "display".to_string(),
                    is_static: false,
                    is_abstract: false,
                    is_sealed: false,
                    access: Access::Public,
                    function_name: "DisplayHookParent.renderDisplay".to_string(),
                    implicit_class_argument: None,
                },
            )]),
        });
        runmat_builtins::register_class(ClassDef {
            name: child_class_name.clone(),
            parent: Some(parent_class_name),
            properties: HashMap::new(),
            methods: HashMap::new(),
        });
        let _resolver =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "DisplayHookParent.renderDisplay").then_some(772)
            })));
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |function, args, requested_outputs| {
                let args = args.to_vec();
                Box::pin(async move {
                    assert_eq!(function, 772);
                    assert_eq!(requested_outputs, 0);
                    assert!(
                        matches!(args.first(), Some(Value::Object(object)) if object.class_name == "DisplayHookChild")
                    );
                    record_console_line(ConsoleStream::Stdout, "inherited display");
                    Ok(empty_return_value())
                })
            },
        )));

        reset_thread_buffer();
        block_on(display_builtin(
            Value::Object(ObjectInstance::new(child_class_name)),
            Vec::new(),
        ))
        .expect("display");

        assert_eq!(stdout_text(), "inherited display\n");
    }

    #[test]
    fn display_rejects_extra_arguments() {
        let err = block_on(display_builtin(Value::Num(1.0), vec![Value::Num(2.0)]))
            .expect_err("extra argument should fail");
        assert_eq!(err.identifier(), DISPLAY_ERROR_ARG_CONFIG.identifier);
    }
}
