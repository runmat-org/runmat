//! MATLAB-compatible `clf` builtin.

use std::collections::BTreeSet;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use super::op_common::figure_actions::{parse_clf_action, FigureAction};
use super::state::{clear_figure, clear_figure_with_builtin, figure_handles, FigureHandle};
use crate::builtins::plotting::type_resolvers::handle_scalar_type;

const CLF_OUTPUT_RESULT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "result",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Cleared handle for single-target calls or count for multi/all clears.",
}];

const CLF_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const CLF_INPUTS_TARGETS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "targets",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Figure handle(s) or mode tokens ('all', 'reset').",
}];

const CLF_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "result = clf()",
        inputs: &CLF_INPUTS_NONE,
        outputs: &CLF_OUTPUT_RESULT,
    },
    BuiltinSignatureDescriptor {
        label: "result = clf(targets...)",
        inputs: &CLF_INPUTS_TARGETS,
        outputs: &CLF_OUTPUT_RESULT,
    },
];

const CLF_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CLF.INVALID_ARGUMENT",
    identifier: Some("RunMat:clf:InvalidArgument"),
    when: "One or more clear targets are invalid.",
    message: "clf: invalid argument",
};

const CLF_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CLF.INTERNAL",
    identifier: Some("RunMat:clf:Internal"),
    when: "Internal figure clear operation fails.",
    message: "clf: internal operation failed",
};

const CLF_ERRORS: [BuiltinErrorDescriptor; 2] = [CLF_ERROR_INVALID_ARGUMENT, CLF_ERROR_INTERNAL];

pub const CLF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CLF_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CLF_ERRORS,
};

pub(crate) const CLF_INTEGER_FIGURE_NUMBER_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "clf-integer-figure-number",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "clf with a typed integer figure number is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ClfIntegerFigureNumberExtension"),
    };

pub(crate) const CLF_ALL_SELECTOR_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "clf-all-selector",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "clf('all') is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ClfAllSelectorExtension"),
    };

pub(crate) const CLF_VARIADIC_TARGETS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "clf-variadic-targets",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "clf with separate variadic figure targets is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ClfVariadicTargetsExtension"),
    };

pub const CLF_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    CLF_INTEGER_FIGURE_NUMBER_EXTENSION,
    CLF_ALL_SELECTOR_EXTENSION,
    CLF_VARIADIC_TARGETS_EXTENSION,
];

const CLF_INTEGER_TARGET_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "fig",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "RunMat mode interprets a scalar or array of any built-in integer class as figure numbers, not as numeric figure data. The public compatibility contract documents figure numbers but does not advertise typed integer classes.",
    }];

pub const CLF_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "f = clf(integer_fig)",
        inputs: &CLF_INTEGER_TARGET_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Figure numbers are read from authoritative host integer storage and must be positive and representable as RunMat's u32 figure identifier. RunMat currently encodes a returned figure opaquely as f64 and returns a scalar count for multiple targets; those are explicit graphics-representation gaps, not integer numeric output. No numeric kernel or provider dispatch occurs.",
    }];

#[runtime_builtin(
    name = "clf",
    category = "plotting",
    summary = "Clear figure contents.",
    keywords = "clf,clear figure,plotting",
    sink = true,
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    descriptor(crate::builtins::plotting::clf::CLF_DESCRIPTOR),
    extensions(crate::builtins::plotting::clf::CLF_EXTENSIONS),
    integer_capabilities(crate::builtins::plotting::clf::CLF_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::plotting::clf"
)]
pub fn clf_builtin(rest: Vec<Value>) -> crate::BuiltinResult<f64> {
    if requests_variadic_targets(&rest) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &CLF_VARIADIC_TARGETS_EXTENSION,
            "clf",
        )?;
    }
    if rest.iter().any(is_all_selector) {
        crate::compatibility::ensure_builtin_extension_enabled(&CLF_ALL_SELECTOR_EXTENSION, "clf")?;
    }
    if rest.iter().any(is_typed_integer_value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &CLF_INTEGER_FIGURE_NUMBER_EXTENSION,
            "clf",
        )?;
    }
    let (action, _reset) = parse_clf_action(&rest)?;
    match action {
        FigureAction::Current => {
            let cleared = clear_figure_with_builtin("clf", None)?;
            Ok(cleared.as_u32() as f64)
        }
        FigureAction::Handles(handles) => {
            let ordered: BTreeSet<u32> = handles.into_iter().map(|h| h.as_u32()).collect();
            if ordered.is_empty() {
                let cleared = clear_figure_with_builtin("clf", None)?;
                return Ok(cleared.as_u32() as f64);
            }
            for id in &ordered {
                clear_figure_with_builtin("clf", Some(FigureHandle::from(*id)))?;
            }
            if ordered.len() == 1 {
                Ok(*ordered.iter().next().unwrap() as f64)
            } else {
                Ok(ordered.len() as f64)
            }
        }
        FigureAction::All => {
            let handles = figure_handles();
            if handles.is_empty() {
                return Ok(0.0);
            }
            let count = handles.len();
            for handle in handles {
                let _ = clear_figure(Some(handle));
            }
            Ok(count as f64)
        }
    }
}

fn is_typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
}

fn is_all_selector(value: &Value) -> bool {
    match value {
        Value::String(text) => text.trim().eq_ignore_ascii_case("all"),
        Value::CharArray(chars) if chars.rows == 1 => chars
            .data
            .iter()
            .collect::<String>()
            .trim()
            .eq_ignore_ascii_case("all"),
        Value::StringArray(strings) if strings.data.len() == 1 => {
            strings.data[0].trim().eq_ignore_ascii_case("all")
        }
        _ => false,
    }
}

fn requests_variadic_targets(values: &[Value]) -> bool {
    values
        .iter()
        .filter(|value| !is_reset_selector(value) && !is_all_selector(value))
        .count()
        > 1
}

fn is_reset_selector(value: &Value) -> bool {
    match value {
        Value::String(text) => text.trim().eq_ignore_ascii_case("reset"),
        Value::CharArray(chars) if chars.rows == 1 => chars
            .data
            .iter()
            .collect::<String>()
            .trim()
            .eq_ignore_ascii_case("reset"),
        Value::StringArray(strings) if strings.data.len() == 1 => {
            strings.data[0].trim().eq_ignore_ascii_case("reset")
        }
        _ => false,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::plotting::tests::ensure_plot_test_env;
    use runmat_builtins::{ResolveContext, Type};
    use runmat_value::{IntegerStorage, Tensor};

    fn setup_plot_tests() {
        ensure_plot_test_env();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn clf_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = CLF_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"result = clf()"));
        assert!(labels.contains(&"result = clf(targets...)"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn defaults_to_current() {
        setup_plot_tests();
        assert!(matches!(
            parse_clf_action(&[]).unwrap(),
            (FigureAction::Current, false)
        ));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn parses_all_flag() {
        setup_plot_tests();
        let values = vec![Value::String("all".to_string())];
        assert!(matches!(
            parse_clf_action(&values).unwrap(),
            (FigureAction::All, false)
        ));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn parses_handles() {
        setup_plot_tests();
        let values = vec![Value::Num(2.0)];
        match parse_clf_action(&values).unwrap() {
            (FigureAction::Handles(handles), _) => {
                assert_eq!(handles.len(), 1);
                assert_eq!(handles[0].as_u32(), 2);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn parses_all_integer_figure_number_classes_exactly() {
        let storages = [
            IntegerStorage::I8(vec![2]),
            IntegerStorage::I16(vec![2]),
            IntegerStorage::I32(vec![2]),
            IntegerStorage::I64(vec![2]),
            IntegerStorage::U8(vec![2]),
            IntegerStorage::U16(vec![2]),
            IntegerStorage::U32(vec![2]),
            IntegerStorage::U64(vec![2]),
        ];

        for storage in storages {
            let value = Value::Tensor(Tensor::new_integer(storage, vec![1, 1]).expect("figure"));
            let (FigureAction::Handles(handles), false) = parse_clf_action(&[value]).unwrap()
            else {
                panic!("expected integer figure target");
            };
            assert_eq!(handles, vec![FigureHandle::from(2)]);
        }
    }

    #[test]
    fn integer_figure_numbers_enforce_positive_u32_bounds() {
        for storage in [
            IntegerStorage::I64(vec![-1]),
            IntegerStorage::U64(vec![u32::MAX as u64 + 1]),
        ] {
            let value = Value::Tensor(Tensor::new_integer(storage, vec![1, 1]).expect("figure"));
            assert!(parse_clf_action(&[value]).is_err());
        }
    }

    #[test]
    fn compatibility_mode_rejects_all_integer_figure_number_classes() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let storages = [
            IntegerStorage::I8(vec![2]),
            IntegerStorage::I16(vec![2]),
            IntegerStorage::I32(vec![2]),
            IntegerStorage::I64(vec![2]),
            IntegerStorage::U8(vec![2]),
            IntegerStorage::U16(vec![2]),
            IntegerStorage::U32(vec![2]),
            IntegerStorage::U64(vec![2]),
        ];

        for storage in storages {
            let value = Value::Tensor(Tensor::new_integer(storage, vec![1, 1]).expect("figure"));
            let err = clf_builtin(vec![value]).expect_err("typed integer extension must be gated");
            assert_eq!(
                err.identifier(),
                Some("RunMat:compatibility:ClfIntegerFigureNumberExtension")
            );
        }
    }

    #[test]
    fn compatibility_mode_rejects_runmat_all_selector_before_plot_state_access() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let err =
            clf_builtin(vec![Value::String("all".into())]).expect_err("RunMat-only all selector");
        assert_eq!(
            err.identifier(),
            CLF_ALL_SELECTOR_EXTENSION.error_identifier
        );
    }

    #[test]
    fn compatibility_mode_rejects_separate_variadic_targets() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let err = clf_builtin(vec![Value::Num(1.0), Value::Num(2.0)])
            .expect_err("RunMat-only variadic targets");
        assert_eq!(
            err.identifier(),
            CLF_VARIADIC_TARGETS_EXTENSION.error_identifier
        );
    }

    #[test]
    fn resident_numeric_targets_are_rejected_without_provider_dispatch() {
        let resident = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
            shape: vec![1, 1],
            device_id: u32::MAX,
            buffer_id: u64::MAX,
            descriptor: Default::default(),
        });
        assert!(parse_clf_action(&[resident]).is_err());
    }

    #[test]
    fn clf_integer_capability_is_structural_and_host_only() {
        let capability = &CLF_INTEGER_CAPABILITIES[0];
        assert_eq!(capability.inputs[0].classes.len(), 8);
        assert_eq!(
            capability.computation_domain,
            BuiltinIntegerComputationDomain::Structural
        );
        assert_eq!(capability.backend, BuiltinIntegerBackendRule::HostOnly);
        assert_eq!(
            capability.output_class,
            BuiltinIntegerOutputClassRule::NotApplicable
        );
    }

    #[test]
    fn clf_type_is_numeric_handle() {
        assert_eq!(
            handle_scalar_type(&[Type::tensor()], &ResolveContext::new(Vec::new())),
            Type::Num
        );
    }
}
