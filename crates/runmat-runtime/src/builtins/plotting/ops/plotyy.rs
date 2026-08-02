//! MATLAB-compatible legacy `plotyy` builtin.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    Tensor, Value,
};
use runmat_macros::runtime_builtin;

use super::line::{append_line_objects_to_figure, handles_value, parse_line_args};
use super::state::{
    append_active_plot, current_axes_state, prepare_plotyy_axes, register_line3_handle,
    register_line_handle, select_axes_for_figure, set_log_modes_for_axes, PlotRenderOptions,
};
use crate::builtins::plotting::type_resolvers::handle_scalar_type;
use crate::{BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "plotyy";

const OUTPUT_AX_H1_H2: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Two axes handles [left right].",
    },
    BuiltinParamDescriptor {
        name: "h1",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Line handle or handles for the left axes.",
    },
    BuiltinParamDescriptor {
        name: "h2",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Line handle or handles for the right axes.",
    },
];

const INPUTS_X1_Y1_X2_Y2: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "X1",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X data for the left axes.",
    },
    BuiltinParamDescriptor {
        name: "Y1",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y data for the left axes.",
    },
    BuiltinParamDescriptor {
        name: "X2",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X data for the right axes.",
    },
    BuiltinParamDescriptor {
        name: "Y2",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y data for the right axes.",
    },
];

const INPUTS_WITH_FUN: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "args",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "X1, Y1, X2, Y2 followed by one plotting function selector.",
}];

const INPUTS_WITH_FUNS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "args",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "X1, Y1, X2, Y2 followed by left and right plotting function selectors.",
}];

const SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "[ax,h1,h2] = plotyy(X1,Y1,X2,Y2)",
        inputs: &INPUTS_X1_Y1_X2_Y2,
        outputs: &OUTPUT_AX_H1_H2,
    },
    BuiltinSignatureDescriptor {
        label: "[ax,h1,h2] = plotyy(X1,Y1,X2,Y2,fun)",
        inputs: &INPUTS_WITH_FUN,
        outputs: &OUTPUT_AX_H1_H2,
    },
    BuiltinSignatureDescriptor {
        label: "[ax,h1,h2] = plotyy(X1,Y1,X2,Y2,fun1,fun2)",
        inputs: &INPUTS_WITH_FUNS,
        outputs: &OUTPUT_AX_H1_H2,
    },
];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PLOTYY.INVALID_ARGUMENT",
    identifier: Some("RunMat:plotyy:InvalidArgument"),
    when: "Input count, plotting function selector, or line data is invalid.",
    message: "plotyy: invalid argument",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PLOTYY.INTERNAL",
    identifier: Some("RunMat:plotyy:Internal"),
    when: "Internal plotting state update fails.",
    message: "plotyy: internal operation failed",
};

const ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID_ARGUMENT, ERROR_INTERNAL];

pub const PLOTYY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

#[derive(Clone, Copy, Debug)]
enum PlotSelector {
    Plot,
    SemilogX,
    SemilogY,
    LogLog,
}

#[runtime_builtin(
    name = "plotyy",
    category = "plotting",
    summary = "Plot two data sets with independent left and right Y axes.",
    keywords = "plotyy,plotting,dual y axis,legacy",
    sink = true,
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    descriptor(crate::builtins::plotting::plotyy::PLOTYY_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::plotyy"
)]
pub async fn plotyy_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.len() < 4 || args.len() > 6 {
        return Err(invalid(
            "expected X1, Y1, X2, Y2 and optional plotting functions",
        ));
    }

    let x1 = args[0].clone();
    let y1 = args[1].clone();
    let x2 = args[2].clone();
    let y2 = args[3].clone();
    let (left_selector, right_selector) = match args.len() {
        4 => (PlotSelector::Plot, PlotSelector::Plot),
        5 => {
            let selector = selector_from_value(&args[4])?;
            (selector, selector)
        }
        6 => (
            selector_from_value(&args[4])?,
            selector_from_value(&args[5])?,
        ),
        _ => unreachable!(),
    };

    let current_axes = current_axes_state();
    let left_line = parse_plot_line(current_axes.handle, current_axes.active_index, x1, y1).await?;
    let right_line =
        parse_plot_line(current_axes.handle, current_axes.active_index, x2, y2).await?;

    let (figure, left_axes_index, right_axes_index, left_axes, right_axes) =
        prepare_plotyy_axes().map_err(map_internal)?;
    apply_selector_modes(figure, left_axes_index, left_selector)?;
    apply_selector_modes(figure, right_axes_index, right_selector)?;

    let left_plot_indices = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let right_plot_indices = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let left_plot_indices_slot = std::rc::Rc::clone(&left_plot_indices);
    let right_plot_indices_slot = std::rc::Rc::clone(&right_plot_indices);
    let mut left_objects = Some(left_line.objects);
    let mut right_objects = Some(right_line.objects);
    let render_result = append_active_plot(
        BUILTIN_NAME,
        PlotRenderOptions::default(),
        move |figure_state, _| {
            left_plot_indices_slot.borrow_mut().extend(
                append_line_objects_to_figure(
                    figure_state,
                    left_axes_index,
                    left_objects
                        .take()
                        .expect("left plotyy line objects consumed exactly once"),
                )
                .into_iter(),
            );
            right_plot_indices_slot.borrow_mut().extend(
                append_line_objects_to_figure(
                    figure_state,
                    right_axes_index,
                    right_objects
                        .take()
                        .expect("right plotyy line objects consumed exactly once"),
                )
                .into_iter(),
            );
            Ok(())
        },
    );
    if let Err(err) = render_result {
        let lower = err.to_string().to_lowercase();
        if !(lower.contains("plotting is unavailable") || lower.contains("non-main thread")) {
            return Err(map_internal(err));
        }
    }

    let h1 = handles_value(
        left_plot_indices
            .borrow()
            .iter()
            .map(|(plot_index, is_3d)| {
                if *is_3d {
                    register_line3_handle(figure, left_axes_index, *plot_index)
                } else {
                    register_line_handle(figure, left_axes_index, *plot_index)
                }
            })
            .collect(),
    );
    let h2 = handles_value(
        right_plot_indices
            .borrow()
            .iter()
            .map(|(plot_index, is_3d)| {
                if *is_3d {
                    register_line3_handle(figure, right_axes_index, *plot_index)
                } else {
                    register_line_handle(figure, right_axes_index, *plot_index)
                }
            })
            .collect(),
    );
    select_axes_for_figure(figure, left_axes_index).map_err(map_internal)?;
    let ax = handles_tensor(vec![left_axes, right_axes]);
    outputs(ax, h1, h2)
}

async fn parse_plot_line(
    figure: super::state::FigureHandle,
    axes_index: usize,
    x: Value,
    y: Value,
) -> BuiltinResult<super::line::ParsedLineCall> {
    parse_line_args(Some((figure, axes_index)), vec![x, y])
        .await
        .map_err(|err| {
            RuntimeError::from(
                crate::build_runtime_error(format!("plotyy: {}", err.message()))
                    .with_builtin(BUILTIN_NAME)
                    .with_identifier(ERROR_INVALID_ARGUMENT.identifier.unwrap_or("RunMat:plotyy"))
                    .with_source(err)
                    .build(),
            )
        })
}

fn outputs(ax: Value, h1: Value, h2: Value) -> BuiltinResult<Value> {
    let values = vec![ax, h1, h2];
    match crate::output_count::current_output_count() {
        Some(0) => Ok(Value::OutputList(Vec::new())),
        Some(count) => {
            if count > 3 {
                return Err(invalid("requested more than three outputs"));
            }
            Ok(crate::output_count::output_list_with_padding(count, values))
        }
        None => Ok(values.into_iter().next().unwrap_or(Value::Num(0.0))),
    }
}

fn selector_from_value(value: &Value) -> BuiltinResult<PlotSelector> {
    let text = match value {
        Value::String(text) => text.clone(),
        Value::CharArray(chars) if chars.rows == 1 => chars.data.iter().collect::<String>(),
        Value::FunctionHandle(name) => name.clone(),
        Value::ExternalFunctionHandle(name) => name.clone(),
        Value::MethodFunctionHandle(name) => name.clone(),
        _ => {
            return Err(invalid(
                "plotting function must be a function handle or function name",
            ))
        }
    };
    match text
        .trim()
        .trim_start_matches('@')
        .to_ascii_lowercase()
        .as_str()
    {
        "plot" | "line" => Ok(PlotSelector::Plot),
        "semilogx" => Ok(PlotSelector::SemilogX),
        "semilogy" => Ok(PlotSelector::SemilogY),
        "loglog" => Ok(PlotSelector::LogLog),
        other => Err(invalid(format!("unsupported plotting function `{other}`"))),
    }
}

fn apply_selector_modes(
    figure: super::state::FigureHandle,
    axes_index: usize,
    selector: PlotSelector,
) -> BuiltinResult<()> {
    let (x_log, y_log) = match selector {
        PlotSelector::Plot => (false, false),
        PlotSelector::SemilogX => (true, false),
        PlotSelector::SemilogY => (false, true),
        PlotSelector::LogLog => (true, true),
    };
    set_log_modes_for_axes(figure, axes_index, x_log, y_log).map_err(map_internal)
}

fn handles_tensor(handles: Vec<f64>) -> Value {
    let len = handles.len();
    Value::Tensor(Tensor::new(handles, vec![1, len]).expect("plotyy handle row"))
}

fn invalid(message: impl Into<String>) -> RuntimeError {
    let mut builder = crate::build_runtime_error(format!("plotyy: {}", message.into()))
        .with_builtin(BUILTIN_NAME);
    if let Some(identifier) = ERROR_INVALID_ARGUMENT.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_internal(err: impl std::error::Error + Send + Sync + 'static) -> RuntimeError {
    let mut builder = crate::build_runtime_error(format!("{}: {err}", ERROR_INTERNAL.message))
        .with_builtin(BUILTIN_NAME);
    if let Some(identifier) = ERROR_INTERNAL.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.with_source(err).build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, current_axes_state, reset_hold_state_for_run};
    use futures::executor::block_on;

    fn setup() -> crate::builtins::plotting::PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    fn tensor(data: &[f64]) -> Value {
        Value::Tensor(Tensor::new(data.to_vec(), vec![1, data.len()]).expect("plotyy row vector"))
    }

    #[test]
    fn plotyy_returns_axes_by_default_and_sets_axis_locations() {
        let _guard = setup();
        let out = block_on(plotyy_builtin(vec![
            tensor(&[1.0, 2.0, 3.0]),
            tensor(&[10.0, 20.0, 30.0]),
            tensor(&[1.0, 2.0, 3.0]),
            tensor(&[100.0, 400.0, 900.0]),
        ]))
        .unwrap();
        let Value::Tensor(ax) = out else {
            panic!("expected axes handle vector");
        };
        assert_eq!(ax.materialize_f64().len(), 2);
        let left_location = get_builtin(vec![
            Value::Num(ax.materialize_f64()[0]),
            Value::String("YAxisLocation".into()),
        ])
        .unwrap();
        let right_location = get_builtin(vec![
            Value::Num(ax.materialize_f64()[1]),
            Value::String("YAxisLocation".into()),
        ])
        .unwrap();
        assert_eq!(left_location, Value::String("left".into()));
        assert_eq!(right_location, Value::String("right".into()));
        assert_eq!(current_axes_state().active_index, 0);
    }

    #[test]
    fn plotyy_returns_three_outputs_when_requested() {
        let _guard = setup();
        let _outputs = crate::output_count::push_output_count(Some(3));
        let out = block_on(plotyy_builtin(vec![
            tensor(&[1.0, 2.0]),
            tensor(&[3.0, 4.0]),
            tensor(&[1.0, 2.0]),
            tensor(&[30.0, 40.0]),
        ]))
        .unwrap();
        let Value::OutputList(values) = out else {
            panic!("expected output list");
        };
        assert_eq!(values.len(), 3);
        assert!(matches!(&values[0], Value::Tensor(ax) if ax.materialize_f64().len() == 2));
        assert!(matches!(&values[1], Value::Num(_) | Value::Tensor(_)));
        assert!(matches!(&values[2], Value::Num(_) | Value::Tensor(_)));
    }

    #[test]
    fn plotyy_function_selectors_set_log_modes_per_axes() {
        let _guard = setup();
        let out = block_on(plotyy_builtin(vec![
            tensor(&[1.0, 10.0]),
            tensor(&[3.0, 4.0]),
            tensor(&[1.0, 10.0]),
            tensor(&[30.0, 40.0]),
            Value::String("semilogx".into()),
            Value::String("semilogy".into()),
        ]))
        .unwrap();
        let Value::Tensor(ax) = out else {
            panic!("expected axes handle vector");
        };
        assert_eq!(
            get_builtin(vec![
                Value::Num(ax.materialize_f64()[0]),
                Value::String("XScale".into())
            ])
            .unwrap(),
            Value::String("log".into())
        );
        assert_eq!(
            get_builtin(vec![
                Value::Num(ax.materialize_f64()[0]),
                Value::String("YScale".into())
            ])
            .unwrap(),
            Value::String("linear".into())
        );
        assert_eq!(
            get_builtin(vec![
                Value::Num(ax.materialize_f64()[1]),
                Value::String("XScale".into())
            ])
            .unwrap(),
            Value::String("linear".into())
        );
        assert_eq!(
            get_builtin(vec![
                Value::Num(ax.materialize_f64()[1]),
                Value::String("YScale".into())
            ])
            .unwrap(),
            Value::String("log".into())
        );
    }
}
