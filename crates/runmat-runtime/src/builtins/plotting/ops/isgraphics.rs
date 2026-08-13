use crate::builtins::plotting::type_resolvers::handle_logical_type;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerAuditDescriptor, BuiltinIntegerAuditKind,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor,
};
use runmat_builtins::{LogicalArray, Value};
use runmat_macros::runtime_builtin;

const ISGRAPHICS_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "True when input resolves to a valid plotting graphics handle.",
}];

const ISGRAPHICS_INPUTS_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Candidate plotting handle value.",
}];

const ISGRAPHICS_INPUTS_HANDLE_TYPE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "h",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Candidate plotting handle value.",
    },
    BuiltinParamDescriptor {
        name: "type",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Required graphics object type.",
    },
];

const ISGRAPHICS_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "tf = isgraphics(h)",
        inputs: &ISGRAPHICS_INPUTS_HANDLE,
        outputs: &ISGRAPHICS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "tf = isgraphics(h, type)",
        inputs: &ISGRAPHICS_INPUTS_HANDLE_TYPE,
        outputs: &ISGRAPHICS_OUTPUT,
    },
];

const ISGRAPHICS_ERRORS: [BuiltinErrorDescriptor; 0] = [];

pub const ISGRAPHICS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ISGRAPHICS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ISGRAPHICS_ERRORS,
};

const ISGRAPHICS_INTEGER_HANDLE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "isgraphics-integer-handle-alias",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "isgraphics numeric integer handle aliases are a RunMat extension",
        error_identifier: Some("RunMat:compatibility:IsgraphicsIntegerHandleExtension"),
    };
const ISGRAPHICS_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [ISGRAPHICS_INTEGER_HANDLE_EXTENSION];
pub const ISGRAPHICS_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "MATLAB graphics predicates operate on graphics objects; numeric integer handle aliases are a separately declared RunMat-only extension and compatibility mode returns same-shaped false.",
    };

#[runtime_builtin(
    name = "isgraphics",
    category = "plotting",
    summary = "Return true if the input is a valid plotting graphics handle.",
    keywords = "isgraphics,plotting,handle",
    suppress_auto_output = true,
    type_resolver(handle_logical_type),
    descriptor(crate::builtins::plotting::isgraphics::ISGRAPHICS_DESCRIPTOR),
    extensions(ISGRAPHICS_EXTENSIONS),
    integer_audit(crate::builtins::plotting::isgraphics::ISGRAPHICS_INTEGER_AUDIT),
    builtin_path = "crate::builtins::plotting::isgraphics"
)]
pub fn isgraphics_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    if !(1..=2).contains(&args.len()) {
        return Err(crate::builtins::plotting::plotting_error(
            "isgraphics",
            "isgraphics: expected one or two input arguments",
        ));
    }
    let mut result = handle_predicate_value(&args[0], "isgraphics");
    if let Some(kind) = args.get(1) {
        let kind = graphics_type_name(kind)?;
        result = handle_predicate_value_of_type(&args[0], &kind, result);
    }
    Ok(result)
}

fn graphics_type_name(value: &Value) -> crate::BuiltinResult<String> {
    match value {
        Value::String(value) => Ok(value.trim().to_ascii_lowercase()),
        Value::StringArray(value) if value.data.len() == 1 => {
            Ok(value.data[0].trim().to_ascii_lowercase())
        }
        Value::CharArray(value) if value.rows == 1 => Ok(value
            .data
            .iter()
            .collect::<String>()
            .trim()
            .to_ascii_lowercase()),
        _ => Err(crate::builtins::plotting::plotting_error(
            "isgraphics",
            "isgraphics: type must be a text scalar",
        )),
    }
}

fn handle_predicate_value_of_type(value: &Value, kind: &str, base: Value) -> Value {
    let scalar_matches = |value: &Value| graphics_handle_matches_type(value, kind);
    match (value, base) {
        (Value::Tensor(tensor), Value::LogicalArray(mut logical)) => {
            for index in 0..tensor.len() {
                let candidate = tensor
                    .numeric_value_at(index)
                    .and_then(numeric_graphics_handle)
                    .map(Value::Num)
                    .unwrap_or(Value::Num(f64::NAN));
                logical.data[index] &= u8::from(scalar_matches(&candidate));
            }
            Value::LogicalArray(logical)
        }
        (_, Value::Bool(base)) => Value::Bool(base && scalar_matches(value)),
        (_, other) => other,
    }
}

fn numeric_graphics_handle(value: runmat_builtins::NumericScalar) -> Option<f64> {
    use runmat_builtins::NumericScalar;
    match value {
        NumericScalar::F64(value) => Some(value),
        NumericScalar::F32(value) => Some(f64::from(value)),
        integer => integer.into_int_value().and_then(|integer| {
            crate::builtins::plotting::op_common::handles::numeric_handle_from_integer(&integer)
        }),
    }
}

fn graphics_handle_matches_type(value: &Value, kind: &str) -> bool {
    use crate::builtins::plotting::properties::PlotHandle;
    use crate::builtins::plotting::state::{PlotChildHandleState, PlotObjectKind};

    let normalized;
    let value = if let Value::Int(integer) = value {
        let Some(handle) =
            crate::builtins::plotting::op_common::handles::numeric_handle_from_integer(integer)
        else {
            return false;
        };
        normalized = Value::Num(handle);
        &normalized
    } else {
        value
    };

    match crate::builtins::plotting::properties::resolve_plot_handle(value, "isgraphics") {
        Ok(PlotHandle::Root) => kind == "root",
        Ok(PlotHandle::Figure(_)) => kind == "figure",
        Ok(PlotHandle::Axes(_, _)) => kind == "axes",
        Ok(PlotHandle::Legend(_, _)) => kind == "legend",
        Ok(PlotHandle::Ruler(_, _, ruler)) => {
            kind == "ruler"
                || matches!(
                    (ruler, kind),
                    (PlotObjectKind::XAxis, "xaxis") | (PlotObjectKind::YAxis, "yaxis")
                )
        }
        Ok(PlotHandle::Text(_, _, _)) => kind == "text",
        Ok(PlotHandle::PlotChild(_, child)) => match child.as_ref() {
            PlotChildHandleState::Histogram(_) => kind == "histogram",
            PlotChildHandleState::Histogram2(_) => kind == "histogram2",
            PlotChildHandleState::Line(_) => kind == "line",
            PlotChildHandleState::AnimatedLine(_) => kind == "animatedline",
            PlotChildHandleState::Scatter(_) => kind == "scatter",
            PlotChildHandleState::Bar(_) => kind == "bar",
            PlotChildHandleState::Stem(_) => kind == "stem",
            PlotChildHandleState::ErrorBar(_) => kind == "errorbar",
            PlotChildHandleState::Stairs(_) => kind == "stairs",
            PlotChildHandleState::Quiver(_) => kind == "quiver",
            PlotChildHandleState::Image(_) => kind == "image",
            PlotChildHandleState::Heatmap(_) => kind == "heatmap",
            PlotChildHandleState::Binscatter(_) => kind == "binscatter",
            PlotChildHandleState::FunctionSurface(_) => kind == "functionsurface",
            PlotChildHandleState::FunctionContour(_) => kind == "functioncontour",
            PlotChildHandleState::Area(_) => kind == "area",
            PlotChildHandleState::Surface(_) => kind == "surface",
            PlotChildHandleState::Patch(_) => kind == "patch",
            PlotChildHandleState::Line3(_) => kind == "line",
            PlotChildHandleState::Scatter3(_) => kind == "scatter",
            PlotChildHandleState::Contour(_) | PlotChildHandleState::ContourFill(_) => {
                kind == "contour"
            }
            PlotChildHandleState::ReferenceLine(_) => kind == "constantline",
            PlotChildHandleState::Pie(_) => kind == "patch",
            PlotChildHandleState::Text(_) => kind == "text",
            PlotChildHandleState::TextScatter(_) => kind == "textscatter",
            PlotChildHandleState::WordCloud(_) => kind == "wordcloud",
            PlotChildHandleState::StackedPlot(_) => kind == "stackedplot",
        },
        Err(_) => false,
    }
}

fn handle_predicate_value(value: &Value, builtin: &'static str) -> Value {
    match value {
        Value::GpuTensor(handle) => false_value(handle.shape.clone()),
        Value::Int(integer) => {
            if !crate::compatibility::runmat_extensions_enabled() {
                return Value::Bool(false);
            }
            Value::Bool(
                crate::builtins::plotting::op_common::handles::numeric_handle_from_integer(integer)
                    .is_some_and(|handle| handle_is_graphics(handle, builtin)),
            )
        }
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => {
            if crate::compatibility::ensure_builtin_extension_enabled(
                &ISGRAPHICS_INTEGER_HANDLE_EXTENSION,
                builtin,
            )
            .is_err()
            {
                return Value::LogicalArray(
                    LogicalArray::new(vec![0; tensor.len()], tensor.shape.clone())
                        .expect("logical shape from tensor"),
                );
            }
            integer_handle_predicate(tensor, builtin, handle_is_graphics)
        }
        Value::Tensor(tensor) => numeric_handle_predicate(tensor, builtin, handle_is_graphics),
        _ => Value::Bool(
            crate::builtins::plotting::properties::resolve_plot_handle(value, builtin).is_ok(),
        ),
    }
}

fn false_value(shape: Vec<usize>) -> Value {
    if shape.iter().product::<usize>() == 1 {
        Value::Bool(false)
    } else {
        Value::LogicalArray(
            LogicalArray::new(vec![0; shape.iter().product()], shape)
                .expect("logical shape from input metadata"),
        )
    }
}

fn integer_handle_predicate(
    tensor: &runmat_builtins::Tensor,
    builtin: &'static str,
    predicate: fn(f64, &'static str) -> bool,
) -> Value {
    let storage = tensor
        .integer_storage()
        .expect("integer handle predicate requires integer storage");
    let data = storage
        .exact_values()
        .iter()
        .map(|value| {
            u8::from(
                crate::builtins::plotting::op_common::handles::numeric_handle_from_integer(value)
                    .is_some_and(|handle| predicate(handle, builtin)),
            )
        })
        .collect();
    Value::LogicalArray(
        LogicalArray::new(data, tensor.shape.clone()).expect("logical shape from tensor"),
    )
}

fn numeric_handle_predicate(
    tensor: &runmat_builtins::Tensor,
    builtin: &'static str,
    predicate: fn(f64, &'static str) -> bool,
) -> Value {
    let data = tensor
        .materialize_f64()
        .iter()
        .map(|&handle| u8::from(predicate(handle, builtin)))
        .collect();
    Value::LogicalArray(
        LogicalArray::new(data, tensor.shape.clone()).expect("logical shape from tensor"),
    )
}

fn handle_is_graphics(handle: f64, builtin: &'static str) -> bool {
    if !handle.is_finite() || handle < 0.0 {
        return false;
    }
    crate::builtins::plotting::properties::resolve_plot_handle(&Value::Num(handle), builtin).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isgraphics_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = ISGRAPHICS_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"tf = isgraphics(h)"));
        assert!(labels.contains(&"tf = isgraphics(h, type)"));
    }

    #[test]
    fn isgraphics_vectorizes_numeric_handle_arrays() {
        let handles = runmat_builtins::Tensor::new(vec![f64::NAN, -1.0], vec![2, 1]).unwrap();
        let result = isgraphics_builtin(vec![Value::Tensor(handles)]).unwrap();
        match result {
            Value::LogicalArray(logical) => {
                assert_eq!(logical.shape, vec![2, 1]);
                assert_eq!(logical.data, vec![0, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[test]
    fn isgraphics_vectorizes_typed_handles_without_a_floating_mirror() {
        let handles = runmat_builtins::Tensor::new_integer(
            runmat_builtins::IntegerStorage::I16(vec![-1, -2]),
            vec![2, 1],
        )
        .unwrap();
        let result = isgraphics_builtin(vec![Value::Tensor(handles)]).unwrap();
        match result {
            Value::LogicalArray(logical) => assert_eq!(logical.data, vec![0, 0]),
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[test]
    fn compatibility_mode_treats_integer_arrays_as_nonobjects() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let handles = runmat_builtins::Tensor::new_integer(
            runmat_builtins::IntegerStorage::U64(vec![u64::MAX]),
            vec![1, 1],
        )
        .unwrap();
        let Value::LogicalArray(result) = isgraphics_builtin(vec![Value::Tensor(handles)]).unwrap()
        else {
            panic!("expected logical array");
        };
        assert_eq!(result.data, vec![0]);
    }

    #[test]
    fn isgraphics_validates_arity_and_type_selector() {
        assert!(isgraphics_builtin(Vec::new()).is_err());
        assert!(
            isgraphics_builtin(vec![Value::Num(0.0), Value::from("root"), Value::Num(1.0)])
                .is_err()
        );
        assert_eq!(
            isgraphics_builtin(vec![Value::Num(0.0), Value::from("root")]).unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            isgraphics_builtin(vec![Value::Num(0.0), Value::from("figure")]).unwrap(),
            Value::Bool(false)
        );
        let handles = runmat_builtins::Tensor::new(vec![0.0, f64::NAN], vec![2, 1]).unwrap();
        let Value::LogicalArray(result) =
            isgraphics_builtin(vec![Value::Tensor(handles), Value::from("root")]).unwrap()
        else {
            panic!("expected logical array");
        };
        assert_eq!(result.data, vec![1, 0]);
    }
}
