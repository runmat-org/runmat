//! MATLAB-compatible `daspect` builtin.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    Tensor, Value,
};
use runmat_macros::runtime_builtin;

use super::op_common::axes_target::AxesTarget;
use super::properties::{data_aspect_ratio_mode_from_value, resolve_plot_handle, PlotHandle};
use super::state::{
    data_aspect_ratio_snapshot, data_aspect_ratio_snapshot_for_axes, set_data_aspect_ratio,
    set_data_aspect_ratio_for_axes,
};
use super::{plotting_error, plotting_error_with_source};
use crate::builtins::plotting::type_resolvers::daspect_type;
use crate::BuiltinResult;

const BUILTIN_NAME: &str = "daspect";

const DASPECT_OUTPUT_RATIO: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ratio",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Current data aspect ratio [dx dy dz].",
}];

const DASPECT_OUTPUT_MODE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Current data aspect ratio mode, 'auto' or 'manual'.",
}];

const DASPECT_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const DASPECT_INPUTS_RATIO: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ratio",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Three-element positive numeric data aspect ratio.",
}];

const DASPECT_INPUTS_MODE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Mode string: 'auto', 'manual', or 'mode'.",
}];

const DASPECT_INPUTS_AX_RATIO: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes handle.",
    },
    BuiltinParamDescriptor {
        name: "ratio_or_mode",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Ratio vector or mode string.",
    },
];

const DASPECT_INPUTS_AX: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ax",
    ty: BuiltinParamType::AxesHandle,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Target axes handle.",
}];

const DASPECT_SIGNATURES: [BuiltinSignatureDescriptor; 7] = [
    BuiltinSignatureDescriptor {
        label: "ratio = daspect()",
        inputs: &DASPECT_INPUTS_NONE,
        outputs: &DASPECT_OUTPUT_RATIO,
    },
    BuiltinSignatureDescriptor {
        label: "daspect(ratio)",
        inputs: &DASPECT_INPUTS_RATIO,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "mode = daspect('mode')",
        inputs: &DASPECT_INPUTS_MODE,
        outputs: &DASPECT_OUTPUT_MODE,
    },
    BuiltinSignatureDescriptor {
        label: "daspect(mode)",
        inputs: &DASPECT_INPUTS_MODE,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "ratio = daspect(ax)",
        inputs: &DASPECT_INPUTS_AX,
        outputs: &DASPECT_OUTPUT_RATIO,
    },
    BuiltinSignatureDescriptor {
        label: "mode = daspect(ax, 'mode')",
        inputs: &DASPECT_INPUTS_AX_RATIO,
        outputs: &DASPECT_OUTPUT_MODE,
    },
    BuiltinSignatureDescriptor {
        label: "daspect(ax, ratio_or_mode)",
        inputs: &DASPECT_INPUTS_AX_RATIO,
        outputs: &[],
    },
];

const DASPECT_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DASPECT.INVALID_ARGUMENT",
    identifier: Some("RunMat:daspect:InvalidArgument"),
    when: "Argument count, axes handle, ratio vector, or mode string is invalid.",
    message: "daspect: invalid argument",
};

const DASPECT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DASPECT.INTERNAL",
    identifier: Some("RunMat:daspect:Internal"),
    when: "Internal plotting state update fails.",
    message: "daspect: internal operation failed",
};

const DASPECT_ERRORS: [BuiltinErrorDescriptor; 2] =
    [DASPECT_ERROR_INVALID_ARGUMENT, DASPECT_ERROR_INTERNAL];

pub const DASPECT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DASPECT_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DASPECT_ERRORS,
};

pub const DASPECT_NUMERIC_AXES_ALIAS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "daspect-numeric-axes-handle-alias",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "daspect with RunMat's numeric encoded axes handle is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:DaspectNumericAxesAliasExtension"),
    };
pub const DASPECT_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [DASPECT_NUMERIC_AXES_ALIAS_EXTENSION];

const DASPECT_INTEGER_RATIO: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "ratio",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight positive integer classes are documented. Shape and positivity are checked from authoritative integer storage before conversion to the host double graphics-property boundary.",
    }];
const DASPECT_INTEGER_AXES_ALIAS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "ax",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed integers are accepted only as exact scalar aliases for RunMat's encoded axes handles and are separately compatibility gated.",
    }];
const DASPECT_RESIDENT_INTEGER: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "resident ratio or ax",
        classes: &[],
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Resident numeric arrays are rejected without provider lookup, download, or upload.",
    }];

pub const DASPECT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor { form: "daspect(integer_ratio)", inputs: &DASPECT_INTEGER_RATIO, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "The documented integer vector is validated exactly, then converted once into the double-valued DataAspectRatio property; the setter returns no value." },
    BuiltinIntegerCapabilityDescriptor { form: "daspect(integer_ax, ratio_or_mode)", inputs: &DASPECT_INTEGER_AXES_ALIAS, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::ScalarOnly, notes: "RunMat-only exact numeric handle alias; the public contract accepts an Axes object." },
    BuiltinIntegerCapabilityDescriptor { form: "daspect(resident_integer, ...)", inputs: &DASPECT_RESIDENT_INTEGER, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GpuRestricted, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Graphics metadata control never gathers a resident numeric array to reinterpret it as a ratio or handle." },
];

#[runtime_builtin(
    name = "daspect",
    category = "plotting",
    summary = "Query or set axes data aspect ratio.",
    keywords = "daspect,data aspect ratio,plotting,axes",
    suppress_auto_output = true,
    type_resolver(daspect_type),
    descriptor(crate::builtins::plotting::daspect::DASPECT_DESCRIPTOR),
    extensions(crate::builtins::plotting::daspect::DASPECT_EXTENSIONS),
    integer_capabilities(crate::builtins::plotting::daspect::DASPECT_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::plotting::daspect"
)]
pub fn daspect_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let (target, args) = split_optional_axes_target(args)?;
    match args.as_slice() {
        [] => query_ratio(target),
        [value] => {
            if let Some(text) = value_as_string(value) {
                match text.trim().to_ascii_lowercase().as_str() {
                    "mode" => return query_mode(target),
                    "auto" | "manual" => {
                        ensure_setter_has_no_output()?;
                        let mode = data_aspect_ratio_mode_from_value(value, BUILTIN_NAME)?;
                        let ratio = current_ratio(target)?;
                        set_ratio_and_mode(target, ratio, mode)?;
                        return Ok(Value::OutputList(Vec::new()));
                    }
                    _ => {}
                }
            }
            ensure_setter_has_no_output()?;
            let ratio = ratio_from_value(value)?;
            set_ratio_and_mode(target, ratio, "manual")?;
            Ok(Value::OutputList(Vec::new()))
        }
        _ => Err(daspect_err("expected zero or one data-aspect argument")),
    }
}

fn requested_outputs() -> usize {
    crate::output_count::current_output_count()
        .or_else(crate::output_context::requested_output_count)
        .unwrap_or(1)
}

fn ensure_setter_has_no_output() -> BuiltinResult<()> {
    if requested_outputs() == 0 {
        Ok(())
    } else {
        Err(daspect_err("setter forms do not return a value"))
    }
}

fn split_optional_axes_target(args: Vec<Value>) -> BuiltinResult<(AxesTarget, Vec<Value>)> {
    let mut iter = args.into_iter();
    let Some(first) = iter.next() else {
        return Ok((None, Vec::new()));
    };
    if let Some(alias) = exact_numeric_axes_alias(&first)? {
        if let Ok(PlotHandle::Axes(handle, axes_index)) =
            resolve_plot_handle(&Value::Num(alias), BUILTIN_NAME)
        {
            crate::compatibility::ensure_builtin_extension_enabled(
                &DASPECT_NUMERIC_AXES_ALIAS_EXTENSION,
                BUILTIN_NAME,
            )?;
            return Ok((Some((handle, axes_index)), iter.collect()));
        }
    }
    let mut rest = Vec::with_capacity(iter.size_hint().0 + 1);
    rest.push(first);
    rest.extend(iter);
    Ok((None, rest))
}

fn exact_numeric_axes_alias(value: &Value) -> BuiltinResult<Option<f64>> {
    let candidate = match value {
        Value::Num(value) => Some(*value),
        Value::Int(value) => super::op_common::handles::numeric_handle_from_integer(value),
        Value::Tensor(tensor) if crate::builtins::common::tensor::is_scalar_tensor(tensor) => {
            if let Some(storage) = tensor.integer_storage() {
                super::op_common::handles::numeric_handle_from_integer(
                    &storage.value_at(0).expect("scalar integer storage"),
                )
            } else {
                Some(crate::builtins::common::tensor::tensor_value_f64(tensor, 0))
            }
        }
        _ => return Ok(None),
    };
    let value =
        candidate.ok_or_else(|| daspect_err("axes handle must be exactly representable"))?;
    if !value.is_finite() || value <= 0.0 || value.fract() != 0.0 {
        return Err(daspect_err(
            "axes handle alias must be a positive integer scalar",
        ));
    }
    Ok(Some(value))
}

fn ratio_from_value(value: &Value) -> BuiltinResult<[f64; 3]> {
    let Value::Tensor(tensor) = value else {
        return Err(daspect_err(
            "DataAspectRatio must be a host row or column numeric vector",
        ));
    };
    let is_vector = matches!(tensor.shape.as_slice(), [3] | [1, 3] | [3, 1]);
    if !is_vector {
        return Err(daspect_err(
            "DataAspectRatio must be a 3-element row or column vector",
        ));
    }
    if let Some(storage) = tensor.integer_storage() {
        let mut out = [0.0; 3];
        for (index, value) in storage.exact_values().iter().enumerate() {
            if value.try_to_u64().is_none_or(|value| value == 0) {
                return Err(daspect_err("DataAspectRatio values must be positive"));
            }
            out[index] = value.to_f64();
        }
        return Ok(out);
    }
    let values = crate::builtins::common::tensor::tensor_values_f64(tensor);
    if values
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(daspect_err(
            "DataAspectRatio values must be positive and finite",
        ));
    }
    Ok([values[0], values[1], values[2]])
}

fn query_ratio(target: AxesTarget) -> BuiltinResult<Value> {
    Ok(ratio_value(current_ratio(target)?))
}

fn query_mode(target: AxesTarget) -> BuiltinResult<Value> {
    let (_, mode) = match target {
        Some((handle, axes_index)) => data_aspect_ratio_snapshot_for_axes(handle, axes_index)
            .map_err(|err| {
                plotting_error_with_source(BUILTIN_NAME, format!("{BUILTIN_NAME}: {err}"), err)
            })?,
        None => data_aspect_ratio_snapshot(),
    };
    Ok(Value::String(mode))
}

fn current_ratio(target: AxesTarget) -> BuiltinResult<[f64; 3]> {
    let (ratio, _) = match target {
        Some((handle, axes_index)) => data_aspect_ratio_snapshot_for_axes(handle, axes_index)
            .map_err(|err| {
                plotting_error_with_source(BUILTIN_NAME, format!("{BUILTIN_NAME}: {err}"), err)
            })?,
        None => data_aspect_ratio_snapshot(),
    };
    Ok(ratio)
}

fn set_ratio_and_mode(target: AxesTarget, ratio: [f64; 3], mode: &str) -> BuiltinResult<()> {
    match target {
        Some((handle, axes_index)) => {
            set_data_aspect_ratio_for_axes(handle, axes_index, ratio, mode).map_err(|err| {
                plotting_error_with_source(BUILTIN_NAME, format!("{BUILTIN_NAME}: {err}"), err)
            })
        }
        None => {
            set_data_aspect_ratio(ratio, mode);
            Ok(())
        }
    }
}

fn ratio_value(ratio: [f64; 3]) -> Value {
    Value::Tensor(Tensor::new(ratio.to_vec(), vec![1, 3]).expect("aspect-ratio row vector"))
}

fn value_as_string(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::CharArray(chars) => chars.row_string(),
        _ => None,
    }
}

fn daspect_err(detail: impl AsRef<str>) -> crate::RuntimeError {
    plotting_error(BUILTIN_NAME, detail.as_ref().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::set::set_builtin;
    use crate::builtins::plotting::state::axis_display_bounds_snapshot_for_axes;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, reset_hold_state_for_run};
    use futures::executor::block_on;
    use runmat_builtins::{IntegerStorage, LogicalArray};

    fn setup() -> crate::builtins::plotting::state::PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    fn set_daspect(args: Vec<Value>) -> BuiltinResult<Value> {
        let _output = crate::output_count::push_output_count(Some(0));
        daspect_builtin(args)
    }

    #[test]
    fn daspect_descriptor_and_runtime_keep_setters_outputless() {
        for signature in DASPECT_DESCRIPTOR.signatures {
            if signature.label.starts_with("daspect(") {
                assert!(signature.outputs.is_empty(), "{}", signature.label);
            }
        }
        assert_eq!(
            DASPECT_DESCRIPTOR.output_mode,
            BuiltinOutputMode::ByRequestedOutputCount
        );

        let _guard = setup();
        let error =
            daspect_builtin(vec![ratio(&[1.0, 2.0, 1.0])]).expect_err("setter output must reject");
        assert!(error.message().contains("do not return"));
        assert_eq!(
            set_daspect(vec![ratio(&[1.0, 2.0, 1.0])]).unwrap(),
            Value::OutputList(Vec::new())
        );
    }

    #[test]
    fn daspect_queries_sets_ratio_and_mode() {
        let _guard = setup();
        let ax = crate::builtins::plotting::gca::gca_builtin(Vec::new()).unwrap();
        assert_eq!(
            set_daspect(vec![ratio(&[1.0, 2.0, 1.0])]).unwrap(),
            Value::OutputList(Vec::new())
        );
        assert_eq!(
            daspect_builtin(vec![Value::String("mode".into())]).unwrap(),
            Value::String("manual".into())
        );
        assert_eq!(
            get_builtin(vec![
                ax.clone(),
                Value::String("DataAspectRatioMode".into())
            ])
            .unwrap(),
            Value::String("manual".into())
        );

        let prop = get_builtin(vec![ax, Value::String("DataAspectRatio".into())]).unwrap();
        assert_eq!(tensor_data(prop), vec![1.0, 2.0, 1.0]);

        assert_eq!(
            set_daspect(vec![Value::String("auto".into())]).unwrap(),
            Value::OutputList(Vec::new())
        );
    }

    #[test]
    fn daspect_explicit_axes_does_not_change_current_axes() {
        let _guard = setup();
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let ax1 = crate::builtins::plotting::subplot::subplot_builtin(
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(1.0),
        )
        .unwrap();
        let ax2 = crate::builtins::plotting::subplot::subplot_builtin(
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(2.0),
        )
        .unwrap();
        set_daspect(vec![Value::Num(ax1), ratio(&[3.0, 1.0, 1.0])]).unwrap();
        let current = daspect_builtin(Vec::new()).unwrap();
        assert_eq!(tensor_data(current), vec![1.0, 1.0, 1.0]);
        let left = daspect_builtin(vec![Value::Num(ax1)]).unwrap();
        assert_eq!(tensor_data(left), vec![3.0, 1.0, 1.0]);
        let right = daspect_builtin(vec![Value::Num(ax2)]).unwrap();
        assert_eq!(tensor_data(right), vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn daspect_round_trips_through_axes_properties_and_bounds() {
        let _guard = setup();
        let ax = crate::builtins::plotting::gca::gca_builtin(Vec::new()).unwrap();
        block_on(crate::builtins::plotting::plot::plot_builtin(vec![
            ratio(&[0.0, 10.0]),
            ratio(&[0.0, 1.0]),
        ]))
        .unwrap();
        set_builtin(vec![
            ax.clone(),
            Value::String("DataAspectRatio".into()),
            ratio(&[2.0, 1.0, 1.0]),
        ])
        .unwrap();
        let bounds = axis_display_bounds_snapshot_for_axes(
            crate::builtins::plotting::current_figure_handle(),
            0,
        )
        .unwrap()
        .unwrap();
        assert_eq!(bounds, (0.0, 10.0, -2.0, 3.0));
    }

    #[test]
    fn daspect_rejects_invalid_ratios() {
        let _guard = setup();
        let err = set_daspect(vec![ratio(&[1.0, 0.0, 1.0])]).unwrap_err();
        assert!(err.message.contains("positive"));
    }

    #[test]
    fn daspect_accepts_all_documented_integer_ratio_classes() {
        let _guard = setup();
        for storage in [
            IntegerStorage::I8(vec![1, 2, 3]),
            IntegerStorage::I16(vec![1, 2, 3]),
            IntegerStorage::I32(vec![1, 2, 3]),
            IntegerStorage::I64(vec![1, 2, 3]),
            IntegerStorage::U8(vec![1, 2, 3]),
            IntegerStorage::U16(vec![1, 2, 3]),
            IntegerStorage::U32(vec![1, 2, 3]),
            IntegerStorage::U64(vec![1, 2, 3]),
        ] {
            let value = Value::Tensor(Tensor::new_integer(storage, vec![1, 3]).unwrap());
            assert_eq!(
                set_daspect(vec![value]).unwrap(),
                Value::OutputList(Vec::new())
            );
            assert_eq!(
                tensor_data(daspect_builtin(Vec::new()).unwrap()),
                vec![1.0, 2.0, 3.0]
            );
        }
        assert_eq!(DASPECT_INTEGER_CAPABILITIES[0].inputs[0].classes.len(), 8);
    }

    #[test]
    fn daspect_rejects_nonvector_logical_complex_and_resident_ratio_inputs() {
        let _guard = setup();
        let nonvector = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 1, 3]).unwrap());
        assert!(set_daspect(vec![nonvector]).is_err());
        let logical = Value::LogicalArray(LogicalArray::new(vec![1, 1, 1], vec![1, 3]).unwrap());
        assert!(set_daspect(vec![logical]).is_err());
        let complex = Value::ComplexTensor(
            runmat_builtins::ComplexTensor::new(
                vec![(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)],
                vec![1, 3],
            )
            .unwrap(),
        );
        assert!(set_daspect(vec![complex]).is_err());
        let resident = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
            shape: vec![1, 3],
            device_id: u32::MAX,
            buffer_id: u64::MAX - 396,
        });
        assert!(set_daspect(vec![resident]).is_err());
    }

    #[test]
    fn daspect_rejects_registered_resident_ratio_without_provider_access() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let handle = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &[1.0, 2.0, 3.0],
                    shape: &[1, 3],
                })
                .unwrap();
            provider.reset_telemetry();
            let error = set_daspect(vec![Value::GpuTensor(handle.clone())])
                .expect_err("resident ratio must reject");
            assert!(error.message().contains("host row or column"));
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
            let _ = provider.free(&handle);
        });
    }

    #[test]
    fn daspect_parses_all_eight_integer_handle_classes_exactly() {
        for value in [
            runmat_builtins::IntValue::I8(1),
            runmat_builtins::IntValue::I16(1),
            runmat_builtins::IntValue::I32(1),
            runmat_builtins::IntValue::I64(1),
            runmat_builtins::IntValue::U8(1),
            runmat_builtins::IntValue::U16(1),
            runmat_builtins::IntValue::U32(1),
            runmat_builtins::IntValue::U64(1),
        ] {
            assert_eq!(
                exact_numeric_axes_alias(&Value::Int(value)).unwrap(),
                Some(1.0)
            );
        }
        assert!(
            exact_numeric_axes_alias(&Value::Int(runmat_builtins::IntValue::U64(
                9_007_199_254_740_993,
            )))
            .is_err()
        );
    }

    #[test]
    fn daspect_numeric_axes_alias_is_compatibility_gated() {
        let _guard = setup();
        let ax_value = crate::builtins::plotting::gca::gca_builtin(Vec::new()).unwrap();
        let Value::Num(ax) = ax_value else {
            panic!("expected numeric axes handle");
        };
        let alias = Value::Int(runmat_builtins::IntValue::U64(ax as u64));
        let strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = daspect_builtin(vec![alias.clone()]).expect_err("numeric axes alias gate");
        assert_eq!(
            error.identifier(),
            DASPECT_NUMERIC_AXES_ALIAS_EXTENSION.error_identifier
        );
        drop(strict);
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        assert!(daspect_builtin(vec![alias]).is_ok());
        assert!(daspect_builtin(vec![Value::Num(ax + 0.5)]).is_err());
    }

    fn ratio(values: &[f64]) -> Value {
        Value::Tensor(
            Tensor::new(values.to_vec(), vec![1, values.len()]).expect("aspect-ratio test vector"),
        )
    }

    fn tensor_data(value: Value) -> Vec<f64> {
        match value {
            Value::Tensor(tensor) => tensor.materialize_f64(),
            other => panic!("expected tensor, got {other:?}"),
        }
    }
}
