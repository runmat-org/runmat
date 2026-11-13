//! MATLAB-compatible `plotmatrix` builtin.
use runmat_builtins::{
    PLOTMATRIX_ERR_INTERNAL as ERR_INTERNAL,
    PLOTMATRIX_ERR_INVALID_ARGUMENT as ERR_INVALID_ARGUMENT,
};

use runmat_macros::runtime_builtin;
use runmat_value::{Tensor, Value};

use super::histogram::histogram_builtin;
use super::op_common::line_inputs::NumericInput;
use super::scatter::scatter_builtin;
use super::state::{
    configure_subplot, create_axes_for_figure, current_axes_state, encode_axes_handle,
    select_axes_for_figure, FigureHandle,
};
use super::style::value_as_string;
use crate::builtins::common::tensor as tensor_utils;
use crate::BuiltinResult;

const BUILTIN_NAME: &str = "plotmatrix";

type AxesTarget = Option<(FigureHandle, usize)>;
type PlotMatrixArgs = (AxesTarget, crate::call::BuiltinArguments);

#[derive(Clone)]
struct MatrixColumns {
    columns: Vec<Tensor>,
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
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::plotmatrix"
)]
pub async fn plotmatrix_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if matches!(crate::output_count::current_output_count(),Some(n) if n>5) {
        return Err(crate::runtime_error::semantic_error(
            "RunMat:plotting:TooManyOutputs",
            "plotmatrix: expected at most five outputs",
        ));
    }
    let work = super::coordinate_admission::CoordinateWork::admit(runtime.as_ref(), BUILTIN_NAME)?;
    let _arguments = args.fork();
    let result = async {
        let parsed = parse_args(args).await?;
        work.check()?;
        let outputs = render_plotmatrix(runtime.as_ref(), &work, parsed).await?;
        let all_outputs = vec![
            outputs.s,
            outputs.ax,
            outputs.big_ax,
            outputs.h,
            outputs.h_ax,
        ];

        work.finish(match crate::output_count::current_output_count() {
            Some(0) => Ok(Value::output_list(Vec::new())),
            Some(out_count) => Ok(crate::output_count::output_list_with_padding(
                out_count,
                all_outputs,
            )),
            None => Ok(all_outputs.into_iter().next().unwrap_or_else(empty_matrix)),
        })
    }
    .await;
    work.finish(result)
}

struct ParsedPlotMatrix {
    axes_target: AxesTarget,
    x: MatrixColumns,
    y: MatrixColumns,
    one_input: bool,
    style: Option<String>,
}

async fn parse_args(args: crate::call::BuiltinArguments) -> BuiltinResult<ParsedPlotMatrix> {
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
        Some(value)
            if value_as_string(&value)
                .map_err(|source| {
                    crate::builtins::plotting::plotting_error_with_source(
                        "plotmatrix",
                        source.to_string(),
                        source,
                    )
                })?
                .is_some() =>
        {
            style = value_as_string(&value).map_err(|source| {
                crate::builtins::plotting::plotting_error_with_source(
                    "plotmatrix",
                    source.to_string(),
                    source,
                )
            })?;
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
                .map_err(|source| {
                    crate::builtins::plotting::plotting_error_with_source(
                        "plotmatrix",
                        source.to_string(),
                        source,
                    )
                })?
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

fn split_leading_axes_target(args: crate::call::BuiltinArguments) -> BuiltinResult<PlotMatrixArgs> {
    super::op_common::split_leading_axes_arguments(args, BUILTIN_NAME)
}

async fn matrix_columns(value: Value) -> BuiltinResult<MatrixColumns> {
    let tensor = NumericInput::from_value(value, BUILTIN_NAME)?
        .into_tensor_async(BUILTIN_NAME)
        .await?;
    if tensor_utils::tensor_element_len(&tensor) == 0 {
        return Err(invalid("input matrices must be nonempty"));
    }
    let rows = tensor.rows();
    let cols = tensor.cols();
    let (observations, variables) = if rows == 1 || cols == 1 {
        (tensor_utils::tensor_element_len(&tensor), 1)
    } else {
        (rows, cols)
    };
    if observations == 0 || variables == 0 {
        return Err(invalid("input matrices must be nonempty"));
    }

    let columns = if variables == 1 {
        vec![tensor]
    } else {
        (0..variables)
            .map(|col| {
                let indices = (0..observations)
                    .map(|row| col * rows + row)
                    .collect::<Vec<_>>();
                super::op_common::line_inputs::select_native(
                    &tensor,
                    &indices,
                    vec![observations, 1],
                    BUILTIN_NAME,
                )
            })
            .collect::<BuiltinResult<Vec<_>>>()?
    };
    Ok(MatrixColumns {
        columns,
        observations,
    })
}

async fn render_plotmatrix(
    runtime: Option<&crate::context::RuntimeContext>,
    work: &super::coordinate_admission::CoordinateWork<'_>,
    parsed: ParsedPlotMatrix,
) -> BuiltinResult<PlotMatrixOutputs> {
    // Child constructors return one handle each, independent of the outer
    // matrix's requested output prefix. The canonical projection guard is
    // retained through child awaits and restores that outer request on drop.
    let _child_outputs = crate::output_count::push_output_count(Some(1))?;
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
            work.check()?;
            let flat = row + col * rows;
            configure_subplot(rows, cols, row * cols + col)
                .map_err(|err| internal(format!("subplot setup failed: {err}")))?;
            let axes = current_axes_state();
            let ax_handle = encode_axes_handle(axes.handle, axes.active_index);
            axes_handles[flat] = ax_handle;

            if parsed.one_input && row == col {
                let histogram_values = [
                    Value::Num(ax_handle),
                    Value::Tensor(parsed.x.columns[row].clone()),
                ];
                let arguments =
                    crate::call::BuiltinArguments::from_borrowed(&histogram_values, runtime)?;
                let handle = histogram_builtin(runtime.cloned(), arguments).await?;
                hist_handles.push(handle);
                hist_axes.push(ax_handle);
            } else {
                let mut rest = vec![
                    Value::Num(ax_handle),
                    Value::Tensor(parsed.x.columns[row].clone()),
                    Value::Tensor(parsed.y.columns[col].clone()),
                ];
                if let Some(style) = &parsed.style {
                    rest.push(Value::string(style.clone()));
                }
                let arguments = crate::call::BuiltinArguments::from_borrowed(&rest, runtime)?;
                let handle = scatter_builtin(runtime.cloned(), arguments)
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
    Value::Tensor(Tensor::new(data, vec![rows, cols]).expect("plotmatrix result"))
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
    async fn plotmatrix_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
        super::plotmatrix_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(args),
        )
        .await
    }
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, clone_figure, reset_hold_state_for_run};
    use futures::executor::block_on;
    use runmat_value::IntegerStorage;

    fn setup() -> crate::builtins::plotting::state::PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    fn cleared_int_value(storage: IntegerStorage, rows: usize, cols: usize) -> Value {
        let tensor = Tensor::new_integer(storage, vec![rows, cols]).expect("integer tensor");
        Value::Tensor(tensor)
    }

    #[test]
    fn matrix_columns_reads_typed_integer_storage_without_mirror() {
        let columns = block_on(matrix_columns(cleared_int_value(
            IntegerStorage::I16(vec![1, 2, 10, 20]),
            2,
            2,
        )))
        .expect("columns");

        assert_eq!(columns.observations, 2);
        assert_eq!(columns.columns.len(), 2);
        assert_eq!(
            columns.columns[0].integer_storage(),
            Some(&IntegerStorage::I16(vec![1, 2]))
        );
        assert_eq!(
            columns.columns[1].integer_storage(),
            Some(&IntegerStorage::I16(vec![10, 20]))
        );
    }

    #[test]
    fn matrix_columns_preserve_every_integer_storage_class() {
        let cases = [
            (
                IntegerStorage::I8(vec![1, 2, 3, 4]),
                IntegerStorage::I8(vec![1, 2]),
            ),
            (
                IntegerStorage::I16(vec![1, 2, 3, 4]),
                IntegerStorage::I16(vec![1, 2]),
            ),
            (
                IntegerStorage::I32(vec![1, 2, 3, 4]),
                IntegerStorage::I32(vec![1, 2]),
            ),
            (
                IntegerStorage::I64(vec![1, 2, 3, 4]),
                IntegerStorage::I64(vec![1, 2]),
            ),
            (
                IntegerStorage::U8(vec![1, 2, 3, 4]),
                IntegerStorage::U8(vec![1, 2]),
            ),
            (
                IntegerStorage::U16(vec![1, 2, 3, 4]),
                IntegerStorage::U16(vec![1, 2]),
            ),
            (
                IntegerStorage::U32(vec![1, 2, 3, 4]),
                IntegerStorage::U32(vec![1, 2]),
            ),
            (
                IntegerStorage::U64(vec![1, 2, 3, 4]),
                IntegerStorage::U64(vec![1, 2]),
            ),
        ];
        for (input, expected) in cases {
            let columns = block_on(matrix_columns(cleared_int_value(input, 2, 2))).unwrap();
            assert_eq!(columns.columns[0].integer_storage(), Some(&expected));
        }
    }

    #[test]
    fn plotmatrix_scatter_properties_preserve_wide_uint64_columns() {
        let _guard = setup();
        let wide = 9_007_199_254_740_993_u64;
        let output = block_on(plotmatrix_builtin(vec![cleared_int_value(
            IntegerStorage::U64(vec![wide, wide + 1, wide + 2, wide + 3]),
            2,
            2,
        )]))
        .unwrap();
        let Value::Tensor(handles) = output else {
            panic!("expected handle matrix")
        };
        let handle = handles.materialize_f64()[1];
        let Value::Tensor(x_data) =
            get_builtin(vec![Value::Num(handle), Value::from("XData")]).unwrap()
        else {
            panic!("expected XData tensor")
        };
        assert!(matches!(
            x_data.integer_storage(),
            Some(IntegerStorage::U64(values))
                if values == &vec![wide, wide + 1] || values == &vec![wide + 2, wide + 3]
        ));
    }

    #[test]
    fn plotmatrix_one_input_builds_square_grid_with_diagonal_histograms() {
        let _guard = setup();
        let _out_guard =
            crate::output_count::push_output_count(Some(5)).expect("output-count scope");
        let output = block_on(plotmatrix_builtin(vec![matrix(
            vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            3,
            2,
        )]))
        .unwrap();
        assert_eq!(crate::output_count::current_output_count(), Some(5));
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
        let _out_guard =
            crate::output_count::push_output_count(Some(5)).expect("output-count scope");
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
        assert!(handles.materialize_f64()[1].is_finite());
        assert_eq!(
            get_builtin(vec![
                Value::Num(handles.materialize_f64()[1]),
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
        let _out_guard =
            crate::output_count::push_output_count(Some(3)).expect("output-count scope");
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
        Value::Tensor(Tensor::new(data, vec![rows, cols]).expect("plotmatrix test matrix"))
    }

    fn assert_tensor_shape(value: &Value, rows: usize, cols: usize) {
        let Value::Tensor(tensor) = value else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.rows(), rows);
        assert_eq!(tensor.cols(), cols);
    }

    fn first_tensor_value(value: &Value) -> f64 {
        let Value::Tensor(tensor) = value else {
            panic!("expected tensor");
        };
        tensor.materialize_f64()[0]
    }

    fn tensor_data(value: &Value) -> Vec<f64> {
        let Value::Tensor(tensor) = value else {
            panic!("expected tensor");
        };
        tensor.materialize_f64()
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
