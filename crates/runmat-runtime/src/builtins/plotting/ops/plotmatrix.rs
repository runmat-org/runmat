//! MATLAB-compatible `plotmatrix` builtin.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use super::histogram::histogram_builtin;
use super::op_common::line_inputs::NumericInput;
use super::properties::{resolve_plot_handle, PlotHandle};
use super::scatter::scatter_builtin;
use super::state::{
    configure_subplot, create_axes_for_figure, current_axes_state, encode_axes_handle,
    select_axes_for_figure, FigureHandle,
};
use super::style::value_as_string;
use crate::BuiltinResult;

const BUILTIN_NAME: &str = "plotmatrix";

type AxesTarget = Option<(FigureHandle, usize)>;
type PlotMatrixArgs = (AxesTarget, Vec<Value>);

const PARAM_S: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "S",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Matrix of scatter graphics handles for off-diagonal plots.",
};

const PARAM_AX: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "AX",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Matrix of axes handles used by the plot matrix.",
};

const PARAM_BIG_AX: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "BigAx",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Compatibility axes handle for labels spanning the plot matrix.",
};

const PARAM_H: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "H",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Vector of diagonal histogram handles for the one-input form.",
};

const PARAM_H_AX: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "HAx",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Vector of diagonal histogram axes handles for the one-input form.",
};

const OUTPUT_S: [BuiltinParamDescriptor; 1] = [PARAM_S];
const OUTPUT_S_AX: [BuiltinParamDescriptor; 2] = [PARAM_S, PARAM_AX];
const OUTPUT_S_AX_BIGAX: [BuiltinParamDescriptor; 3] = [PARAM_S, PARAM_AX, PARAM_BIG_AX];
const OUTPUT_MULTI: [BuiltinParamDescriptor; 5] =
    [PARAM_S, PARAM_AX, PARAM_BIG_AX, PARAM_H, PARAM_H_AX];

const PARAM_X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Matrix whose columns are variables and rows are observations.",
};

const PARAM_Y: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Matrix whose columns are variables and rows are observations.",
};

const PARAM_LINE_SPEC: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "lineSpec",
    ty: BuiltinParamType::StyleSpec,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Optional marker/color style applied to scatter plots.",
};

const INPUT_X: [BuiltinParamDescriptor; 1] = [PARAM_X];
const INPUT_X_LINE_SPEC: [BuiltinParamDescriptor; 2] = [PARAM_X, PARAM_LINE_SPEC];
const INPUT_X_Y: [BuiltinParamDescriptor; 2] = [PARAM_X, PARAM_Y];
const INPUT_LINE_SPEC: [BuiltinParamDescriptor; 3] = [PARAM_X, PARAM_Y, PARAM_LINE_SPEC];

const SIGNATURES: [BuiltinSignatureDescriptor; 8] = [
    BuiltinSignatureDescriptor {
        label: "S = plotmatrix(X)",
        inputs: &INPUT_X,
        outputs: &OUTPUT_S,
    },
    BuiltinSignatureDescriptor {
        label: "S = plotmatrix(X, LineSpec)",
        inputs: &INPUT_X_LINE_SPEC,
        outputs: &OUTPUT_S,
    },
    BuiltinSignatureDescriptor {
        label: "S = plotmatrix(X, Y)",
        inputs: &INPUT_X_Y,
        outputs: &OUTPUT_S,
    },
    BuiltinSignatureDescriptor {
        label: "S = plotmatrix(X, Y, LineSpec)",
        inputs: &INPUT_LINE_SPEC,
        outputs: &OUTPUT_S,
    },
    BuiltinSignatureDescriptor {
        label: "S = plotmatrix(ax, ...)",
        inputs: &INPUT_LINE_SPEC,
        outputs: &OUTPUT_S,
    },
    BuiltinSignatureDescriptor {
        label: "[S, AX] = plotmatrix(...)",
        inputs: &INPUT_LINE_SPEC,
        outputs: &OUTPUT_S_AX,
    },
    BuiltinSignatureDescriptor {
        label: "[S, AX, BigAx] = plotmatrix(...)",
        inputs: &INPUT_LINE_SPEC,
        outputs: &OUTPUT_S_AX_BIGAX,
    },
    BuiltinSignatureDescriptor {
        label: "[S, AX, BigAx, H, HAx] = plotmatrix(X)",
        inputs: &INPUT_X,
        outputs: &OUTPUT_MULTI,
    },
];

const ERR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PLOTMATRIX.INVALID_ARGUMENT",
    identifier: Some("RunMat:plotmatrix:InvalidArgument"),
    when: "Inputs are nonnumeric, observation counts differ, or style arguments are malformed.",
    message: "plotmatrix: invalid argument",
};

const ERR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PLOTMATRIX.INTERNAL",
    identifier: Some("RunMat:plotmatrix:Internal"),
    when: "Internal subplot, scatter, or histogram rendering fails.",
    message: "plotmatrix: internal operation failed",
};

const ERRORS: [BuiltinErrorDescriptor; 2] = [ERR_INVALID_ARGUMENT, ERR_INTERNAL];

pub const PLOTMATRIX_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

fn plotmatrix_type(_args: &[Type], _ctx: &runmat_builtins::ResolveContext) -> Type {
    Type::Unknown
}

#[derive(Clone)]
struct MatrixColumns {
    columns: Vec<Vec<f64>>,
    observations: usize,
}

struct PlotMatrixOutputs {
    s: Value,
    ax: Value,
    big_ax: Value,
    h: Value,
    h_ax: Value,
}

#[runtime_builtin(
    name = "plotmatrix",
    category = "plotting",
    summary = "Create a scatter plot matrix.",
    keywords = "plotmatrix,scatter,matrix,plotting,statistics",
    sink = true,
    suppress_auto_output = true,
    type_resolver(plotmatrix_type),
    descriptor(crate::builtins::plotting::plotmatrix::PLOTMATRIX_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::plotmatrix"
)]
pub async fn plotmatrix_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let parsed = parse_args(args).await?;
    let outputs = render_plotmatrix(parsed).await?;
    let all_outputs = vec![
        outputs.s,
        outputs.ax,
        outputs.big_ax,
        outputs.h,
        outputs.h_ax,
    ];

    match crate::output_count::current_output_count() {
        Some(0) => Ok(Value::OutputList(Vec::new())),
        Some(out_count) => Ok(crate::output_count::output_list_with_padding(
            out_count,
            all_outputs,
        )),
        None => Ok(all_outputs.into_iter().next().unwrap_or_else(empty_matrix)),
    }
}

struct ParsedPlotMatrix {
    axes_target: AxesTarget,
    x: MatrixColumns,
    y: MatrixColumns,
    one_input: bool,
    style: Option<String>,
}

async fn parse_args(args: Vec<Value>) -> BuiltinResult<ParsedPlotMatrix> {
    if args.is_empty() || args.len() > 4 {
        return Err(invalid(
            "expected plotmatrix(X), plotmatrix(X,lineSpec), plotmatrix(X,Y), plotmatrix(X,Y,lineSpec), or plotmatrix(ax,...)",
        ));
    }
    let (axes_target, args) = split_leading_axes_target(args)?;
    if args.is_empty() || args.len() > 3 {
        return Err(invalid(
            "expected data after optional axes handle and at most one LineSpec",
        ));
    }

    let mut iter = args.into_iter();
    let x = matrix_columns(iter.next().unwrap()).await?;
    let mut style = None;
    let (y, one_input) = match iter.next() {
        Some(value) if value_as_string(&value).is_some() => {
            style = value_as_string(&value);
            (x.clone(), true)
        }
        Some(value) => (matrix_columns(value).await?, false),
        None => (x.clone(), true),
    };
    if x.observations != y.observations {
        return Err(invalid(
            "X and Y must have the same number of observations (rows)",
        ));
    }
    if let Some(value) = iter.next() {
        if style.is_some() {
            return Err(invalid("only one lineSpec argument is supported"));
        }
        style = Some(
            value_as_string(&value)
                .ok_or_else(|| invalid("lineSpec must be a string or character vector"))?,
        );
    }
    Ok(ParsedPlotMatrix {
        axes_target,
        x,
        y,
        one_input,
        style,
    })
}

fn split_leading_axes_target(args: Vec<Value>) -> BuiltinResult<PlotMatrixArgs> {
    let mut iter = args.into_iter();
    let Some(first) = iter.next() else {
        return Ok((None, Vec::new()));
    };
    if let Ok(PlotHandle::Axes(handle, axes_index)) = resolve_plot_handle(&first, BUILTIN_NAME) {
        let rest = iter.collect::<Vec<_>>();
        if rest.is_empty() {
            return Err(invalid("expected data after axes handle"));
        }
        return Ok((Some((handle, axes_index)), rest));
    }
    let mut rest = Vec::with_capacity(iter.size_hint().0 + 1);
    rest.push(first);
    rest.extend(iter);
    Ok((None, rest))
}

async fn matrix_columns(value: Value) -> BuiltinResult<MatrixColumns> {
    let tensor = NumericInput::from_value(value, BUILTIN_NAME)?
        .into_tensor_async(BUILTIN_NAME)
        .await?;
    if tensor.data.is_empty() {
        return Err(invalid("input matrices must be nonempty"));
    }
    let rows = tensor.rows();
    let cols = tensor.cols();
    let (observations, variables) = if rows == 1 || cols == 1 {
        (tensor.data.len(), 1)
    } else {
        (rows, cols)
    };
    if observations == 0 || variables == 0 {
        return Err(invalid("input matrices must be nonempty"));
    }

    let columns = if variables == 1 {
        vec![tensor.data]
    } else {
        (0..variables)
            .map(|col| {
                (0..observations)
                    .map(|row| tensor.data[col * rows + row])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    };
    Ok(MatrixColumns {
        columns,
        observations,
    })
}

async fn render_plotmatrix(parsed: ParsedPlotMatrix) -> BuiltinResult<PlotMatrixOutputs> {
    if let Some((figure, axes_index)) = parsed.axes_target {
        select_axes_for_figure(figure, axes_index)
            .map_err(|err| internal(format!("axes target selection failed: {err}")))?;
    }

    let rows = parsed.x.columns.len();
    let cols = parsed.y.columns.len();
    let mut scatter_handles = vec![f64::NAN; rows * cols];
    let mut axes_handles = vec![f64::NAN; rows * cols];
    let mut hist_handles = Vec::new();
    let mut hist_axes = Vec::new();

    for row in 0..rows {
        for col in 0..cols {
            let flat = row + col * rows;
            configure_subplot(rows, cols, row * cols + col)
                .map_err(|err| internal(format!("subplot setup failed: {err}")))?;
            let axes = current_axes_state();
            let ax_handle = encode_axes_handle(axes.handle, axes.active_index);
            axes_handles[flat] = ax_handle;

            if parsed.one_input && row == col {
                let handle = histogram_builtin(vec![
                    Value::Num(ax_handle),
                    vector_value(parsed.x.columns[row].clone()),
                ])
                .await?;
                hist_handles.push(handle);
                hist_axes.push(ax_handle);
            } else {
                let mut rest = vec![
                    Value::Num(ax_handle),
                    vector_value(parsed.x.columns[row].clone()),
                    vector_value(parsed.y.columns[col].clone()),
                ];
                if let Some(style) = &parsed.style {
                    rest.push(Value::String(style.clone()));
                }
                let handle = scatter_builtin(rest.remove(0), rest.remove(0), rest)
                    .await
                    .map_err(|err| {
                        if err.identifier().is_some() {
                            err
                        } else {
                            internal(err.message)
                        }
                    })?;
                scatter_handles[flat] = handle;
            }
        }
    }

    let big_ax = match parsed.axes_target {
        Some((figure, axes_index)) => {
            select_axes_for_figure(figure, axes_index)
                .map_err(|err| internal(format!("big axes selection failed: {err}")))?;
            encode_axes_handle(figure, axes_index)
        }
        None => {
            let (figure, axes_index) = create_axes_for_figure(None)
                .map_err(|err| internal(format!("big axes creation failed: {err}")))?;
            encode_axes_handle(figure, axes_index)
        }
    };

    Ok(PlotMatrixOutputs {
        s: matrix_value(scatter_handles, rows, cols),
        ax: matrix_value(axes_handles, rows, cols),
        big_ax: Value::Num(big_ax),
        h: vector_value(hist_handles),
        h_ax: vector_value(hist_axes),
    })
}

fn vector_value(data: Vec<f64>) -> Value {
    let len = data.len();
    matrix_value(data, 1, len)
}

fn matrix_value(data: Vec<f64>, rows: usize, cols: usize) -> Value {
    Value::Tensor(Tensor {
        data,
        rows,
        cols,
        shape: vec![rows, cols],
        dtype: runmat_builtins::NumericDType::F64,
    })
}

fn empty_matrix() -> Value {
    matrix_value(Vec::new(), 0, 0)
}

fn invalid(message: impl AsRef<str>) -> crate::RuntimeError {
    let message = format!("{}: {}", ERR_INVALID_ARGUMENT.message, message.as_ref());
    let builder = crate::build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .with_identifier(ERR_INVALID_ARGUMENT.identifier.unwrap());
    builder.build()
}

fn internal(message: impl AsRef<str>) -> crate::RuntimeError {
    let message = format!("{}: {}", ERR_INTERNAL.message, message.as_ref());
    let builder = crate::build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .with_identifier(ERR_INTERNAL.identifier.unwrap());
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, clone_figure, reset_hold_state_for_run};
    use futures::executor::block_on;

    fn setup() -> crate::builtins::plotting::state::PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    #[test]
    fn plotmatrix_one_input_builds_square_grid_with_diagonal_histograms() {
        let _guard = setup();
        let _out_guard = crate::output_count::push_output_count(Some(5));
        let output = block_on(plotmatrix_builtin(vec![matrix(
            vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            3,
            2,
        )]))
        .unwrap();
        let Value::OutputList(outputs) = output else {
            panic!("expected output list");
        };
        assert_tensor_shape(&outputs[0], 2, 2);
        assert_tensor_shape(&outputs[1], 2, 2);
        assert_tensor_shape(&outputs[3], 1, 2);
        assert_tensor_shape(&outputs[4], 1, 2);

        let scatter = tensor_data(&outputs[0]);
        assert!(scatter[0].is_nan());
        assert!(scatter[1].is_finite());
        assert!(scatter[2].is_finite());
        assert!(scatter[3].is_nan());
        assert_eq!(
            get_builtin(vec![Value::Num(scatter[1]), Value::String("Type".into())]).unwrap(),
            Value::String("scatter".into())
        );

        let hist = tensor_data(&outputs[3]);
        assert_eq!(
            get_builtin(vec![Value::Num(hist[0]), Value::String("Type".into())]).unwrap(),
            Value::String("histogram".into())
        );
        assert_eq!(current_big_axes_handle(), numeric_scalar(&outputs[2]));

        let figure = clone_figure(crate::builtins::plotting::current_figure_handle()).unwrap();
        assert_eq!(figure.axes_count(), 5);
    }

    #[test]
    fn plotmatrix_two_input_returns_requested_handle_and_axes_matrices() {
        let _guard = setup();
        let _out_guard = crate::output_count::push_output_count(Some(5));
        let output = block_on(plotmatrix_builtin(vec![
            matrix(vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0], 3, 2),
            matrix(vec![4.0, 5.0, 6.0], 3, 1),
            Value::String("r+".into()),
        ]))
        .unwrap();
        let Value::OutputList(outputs) = output else {
            panic!("expected output list");
        };
        assert_eq!(outputs.len(), 5);
        assert_tensor_shape(&outputs[0], 2, 1);
        assert_tensor_shape(&outputs[1], 2, 1);
        assert_eq!(current_big_axes_handle(), numeric_scalar(&outputs[2]));
        assert_tensor_shape(&outputs[3], 1, 0);
        assert_tensor_shape(&outputs[4], 1, 0);

        let h = first_tensor_value(&outputs[0]);
        assert_eq!(
            get_builtin(vec![Value::Num(h), Value::String("Type".into())]).unwrap(),
            Value::String("scatter".into())
        );
        let marker = get_builtin(vec![Value::Num(h), Value::String("Marker".into())]).unwrap();
        assert_eq!(marker, Value::String("+".into()));
    }

    #[test]
    fn plotmatrix_accepts_line_spec_without_y_argument() {
        let _guard = setup();
        let output = block_on(plotmatrix_builtin(vec![
            matrix(vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0], 3, 2),
            Value::String("g+".into()),
        ]))
        .unwrap();
        let Value::Tensor(handles) = output else {
            panic!("expected scatter matrix");
        };
        assert!(handles.data[1].is_finite());
        assert_eq!(
            get_builtin(vec![
                Value::Num(handles.data[1]),
                Value::String("Marker".into())
            ])
            .unwrap(),
            Value::String("+".into())
        );
    }

    #[test]
    fn plotmatrix_accepts_leading_axes_target_as_big_axes() {
        let _guard = setup();
        let (_figure, axes_index) = create_axes_for_figure(None).unwrap();
        let target = encode_axes_handle(
            crate::builtins::plotting::current_figure_handle(),
            axes_index,
        );
        let _out_guard = crate::output_count::push_output_count(Some(3));
        let output = block_on(plotmatrix_builtin(vec![
            Value::Num(target),
            matrix(vec![1.0, 2.0, 3.0], 3, 1),
        ]))
        .unwrap();
        let Value::OutputList(outputs) = output else {
            panic!("expected output list");
        };
        assert_eq!(numeric_scalar(&outputs[2]), target);
        assert_eq!(current_big_axes_handle(), target);
    }

    #[test]
    fn plotmatrix_rejects_mismatched_observation_counts() {
        let _guard = setup();
        let err = block_on(plotmatrix_builtin(vec![
            matrix(vec![1.0, 2.0, 3.0], 3, 1),
            matrix(vec![1.0, 2.0], 2, 1),
        ]))
        .unwrap_err();
        assert!(err.message.contains("same number of observations"));
    }

    fn matrix(data: Vec<f64>, rows: usize, cols: usize) -> Value {
        Value::Tensor(Tensor {
            data,
            rows,
            cols,
            shape: vec![rows, cols],
            dtype: runmat_builtins::NumericDType::F64,
        })
    }

    fn assert_tensor_shape(value: &Value, rows: usize, cols: usize) {
        let Value::Tensor(tensor) = value else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.rows, rows);
        assert_eq!(tensor.cols, cols);
    }

    fn first_tensor_value(value: &Value) -> f64 {
        let Value::Tensor(tensor) = value else {
            panic!("expected tensor");
        };
        tensor.data[0]
    }

    fn tensor_data(value: &Value) -> &[f64] {
        let Value::Tensor(tensor) = value else {
            panic!("expected tensor");
        };
        &tensor.data
    }

    fn numeric_scalar(value: &Value) -> f64 {
        let Value::Num(number) = value else {
            panic!("expected numeric scalar");
        };
        *number
    }

    fn current_big_axes_handle() -> f64 {
        let axes = current_axes_state();
        encode_axes_handle(axes.handle, axes.active_index)
    }
}
