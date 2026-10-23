//! MATLAB-compatible `sphere` builtin.
use runmat_builtins::BuiltinErrorDescriptor;
use runmat_builtins::{
    SPHERE_DEFAULT_N as DEFAULT_N, SPHERE_MAX_COORDINATE_ELEMENTS as MAX_COORDINATE_ELEMENTS,
    SPHERE_WHOLE_COUNT_TOLERANCE,
};
pub use runmat_builtins::{
    SPHERE_ERROR_ARG_COUNT, SPHERE_ERROR_INTERNAL, SPHERE_ERROR_INVALID_N,
    SPHERE_ERROR_OUTPUT_COUNT, SPHERE_ERROR_TOO_LARGE, SPHERE_TYPED_N_EXTENSION,
};

use runmat_macros::runtime_builtin;
use runmat_plot::plots::SurfacePlot;
use runmat_value::{IntValue, Tensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::plotting::state::{render_active_plot, PlotRenderOptions};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "sphere";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::sphere")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "sphere",
    op_kind: GpuOpKind::PlotRender,
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "sphere generates dense host coordinate tensors and renders statement-form calls as a 3-D surface mesh.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::plotting::sphere")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "sphere",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "sphere materializes coordinate tensors and terminates fusion chains.",
};

fn sphere_error(
    descriptor: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", descriptor.message, detail.as_ref()))
        .with_builtin(BUILTIN_NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "sphere",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::sphere"
)]
pub async fn sphere_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    let work = super::coordinate_admission::CoordinateWork::admit(runtime.as_ref(), "sphere")?;
    let _arguments = args.fork();
    work.finish(sphere_impl(args, &work).await)
}

async fn sphere_impl(
    args: crate::call::BuiltinArguments,
    work: &super::coordinate_admission::CoordinateWork<'_>,
) -> crate::BuiltinResult<Value> {
    if matches!(crate::output_count::current_output_count(), Some(n) if n > 3) {
        return Err(sphere_error(
            &SPHERE_ERROR_OUTPUT_COUNT,
            "requested output count exceeds [X,Y,Z]",
        ));
    }

    if let Some(value) = args.first().filter(|_| args.len() == 1) {
        if crate::builtins::common::validation::value_has_native_integer_class(value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &SPHERE_TYPED_N_EXTENSION,
                BUILTIN_NAME,
            )?;
        }
    }

    let n = match args.len() {
        0 => DEFAULT_N,
        1 => parse_n_value(&args[0]).await?,
        _ => {
            return Err(sphere_error(
                &SPHERE_ERROR_ARG_COUNT,
                "use sphere or sphere(n)",
            ))
        }
    };

    if is_plot_form_call() {
        let (x, y, z) = make_sphere_tensors(n, true, true, true)?;
        work.check()?;
        render_sphere_plot(x, y, z, work)?;
        return Ok(Value::output_list(Vec::new()));
    }

    build_output(n)
}

fn is_plot_form_call() -> bool {
    matches!(crate::output_count::current_output_count(), Some(0))
        || (crate::output_context::requested_output_count() == Some(0)
            && crate::output_count::current_output_count().is_none())
}

fn build_output(n: usize) -> BuiltinResult<Value> {
    match crate::output_count::current_output_count() {
        Some(0) => Ok(Value::output_list(Vec::new())),
        Some(1) => {
            let (x, _, _) = make_sphere_tensors(n, true, false, false)?;
            Ok(Value::output_list(vec![Value::Tensor(x)]))
        }
        Some(2) => {
            let (x, y, _) = make_sphere_tensors(n, true, true, false)?;
            Ok(Value::output_list(vec![Value::Tensor(x), Value::Tensor(y)]))
        }
        Some(3) => {
            let (x, y, z) = make_sphere_tensors(n, true, true, true)?;
            Ok(Value::output_list(vec![
                Value::Tensor(x),
                Value::Tensor(y),
                Value::Tensor(z),
            ]))
        }
        Some(_) => Err(sphere_error(
            &SPHERE_ERROR_OUTPUT_COUNT,
            "requested output count exceeds [X,Y,Z]",
        )),
        None => {
            let (x, _, _) = make_sphere_tensors(n, true, false, false)?;
            Ok(Value::Tensor(x))
        }
    }
}

fn render_sphere_plot(
    x: Tensor,
    y: Tensor,
    z: Tensor,
    work: &super::coordinate_admission::CoordinateWork<'_>,
) -> BuiltinResult<()> {
    let x_grid = tensor_to_plot_grid(x)?;
    let y_grid = tensor_to_plot_grid(y)?;
    let z_grid = tensor_to_plot_grid(z)?;
    let mut plot = Some(
        SurfacePlot::from_coordinate_grids(x_grid, y_grid, z_grid)
            .map_err(|err| sphere_error(&SPHERE_ERROR_INTERNAL, err))?,
    );
    let opts = PlotRenderOptions {
        axis_equal: true,
        ..Default::default()
    };
    work.check()?;
    let render_result = render_active_plot(BUILTIN_NAME, opts, move |figure, axes| {
        figure.add_surface_plot_on_axes(plot.take().expect("sphere plot consumed once"), axes);
        Ok(())
    });

    if let Err(err) = render_result {
        if crate::builtins::plotting::state::presentation_unavailable(&err) {
            return Ok(());
        }
        let mapped = sphere_error(
            &SPHERE_ERROR_INTERNAL,
            format!("could not render sphere surface: {}", err.message),
        );
        return Err(build_runtime_error(mapped.message)
            .with_builtin(BUILTIN_NAME)
            .with_identifier(
                SPHERE_ERROR_INTERNAL
                    .identifier
                    .expect("sphere internal identifier"),
            )
            .with_source(err)
            .build());
    }
    Ok(())
}

fn tensor_to_plot_grid(tensor: Tensor) -> BuiltinResult<Vec<Vec<f64>>> {
    let rows = tensor.rows();
    let cols = tensor.cols();
    if rows == 0 || cols == 0 || tensor.len() != rows * cols {
        return Err(sphere_error(
            &SPHERE_ERROR_INTERNAL,
            "coordinate tensor has invalid shape",
        ));
    }
    let values = tensor.materialize_f64();
    let mut grid = vec![vec![0.0; rows]; cols];
    for (col, col_values) in grid.iter_mut().enumerate() {
        for (row, cell) in col_values.iter_mut().enumerate() {
            *cell = values[row + rows * col];
        }
    }
    Ok(grid)
}

async fn parse_n_value(value: &Value) -> BuiltinResult<usize> {
    if let Some(raw) = exact_integer_scalar(value) {
        return parse_n_integer(&raw);
    }
    let Some(raw) = tensor::scalar_f64_from_value_async(value)
        .await
        .map_err(|err| sphere_error(&SPHERE_ERROR_INVALID_N, err))?
    else {
        return Err(sphere_error(
            &SPHERE_ERROR_INVALID_N,
            "n must contain exactly one numeric value",
        ));
    };
    parse_n_number(raw)
}

fn parse_n_number(raw: f64) -> BuiltinResult<usize> {
    if !raw.is_finite() {
        return Err(sphere_error(&SPHERE_ERROR_INVALID_N, "n must be finite"));
    }
    let rounded = raw.round();
    if (rounded - raw).abs() > SPHERE_WHOLE_COUNT_TOLERANCE
        || rounded < 0.0
        || rounded > usize::MAX as f64
        || (usize::BITS == 64 && rounded == usize::MAX as f64)
    {
        return Err(sphere_error(
            &SPHERE_ERROR_INVALID_N,
            "n must be a non-negative integer representable on this platform",
        ));
    }
    let n = rounded as usize;
    validate_grid_size(n)?;
    Ok(n)
}

fn exact_integer_scalar(value: &Value) -> Option<IntValue> {
    match value {
        Value::Int(value) => Some(value.clone()),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => tensor
            .integer_storage()
            .and_then(|storage| storage.value_at(0)),
        _ => None,
    }
}

fn parse_n_integer(raw: &IntValue) -> BuiltinResult<usize> {
    let Some(n) = raw.try_to_usize() else {
        return Err(sphere_error(
            &SPHERE_ERROR_INVALID_N,
            "n must be a non-negative integer representable on this platform",
        ));
    };
    validate_grid_size(n)?;
    Ok(n)
}

fn validate_grid_size(n: usize) -> BuiltinResult<usize> {
    let size = n
        .checked_add(1)
        .ok_or_else(|| sphere_error(&SPHERE_ERROR_TOO_LARGE, "n + 1 overflowed"))?;
    let elements = size.checked_mul(size).ok_or_else(|| {
        sphere_error(
            &SPHERE_ERROR_TOO_LARGE,
            "coordinate grid element count overflowed",
        )
    })?;
    if elements > MAX_COORDINATE_ELEMENTS {
        return Err(sphere_error(
            &SPHERE_ERROR_TOO_LARGE,
            format!("maximum supported coordinate elements is {MAX_COORDINATE_ELEMENTS}"),
        ));
    }
    Ok(size)
}

fn make_sphere_tensors(
    n: usize,
    want_x: bool,
    want_y: bool,
    want_z: bool,
) -> BuiltinResult<(Tensor, Tensor, Tensor)> {
    let size = validate_grid_size(n)?;
    let elements = size * size;
    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut z = Vec::new();
    if want_x {
        x.try_reserve_exact(elements)
            .map_err(|err| sphere_error(&SPHERE_ERROR_TOO_LARGE, err.to_string()))?;
    }
    if want_y {
        y.try_reserve_exact(elements)
            .map_err(|err| sphere_error(&SPHERE_ERROR_TOO_LARGE, err.to_string()))?;
    }
    if want_z {
        z.try_reserve_exact(elements)
            .map_err(|err| sphere_error(&SPHERE_ERROR_TOO_LARGE, err.to_string()))?;
    }

    for col in 0..size {
        let theta = sphere_theta(col, n);
        let (sin_theta, cos_theta) = theta.sin_cos();
        for row in 0..size {
            let phi = sphere_phi(row, n);
            let (sin_phi, cos_phi) = phi.sin_cos();
            if want_x {
                let x_val = cos_phi * cos_theta;
                x.push(if x_val.abs() < 1e-15 { 0.0 } else { x_val });
            }
            if want_y {
                let y_val = cos_phi * sin_theta;
                y.push(if y_val.abs() < 1e-15 { 0.0 } else { y_val });
            }
            if want_z {
                z.push(if sin_phi.abs() < 1e-15 { 0.0 } else { sin_phi });
            }
        }
    }

    let shape = vec![size, size];
    let x = make_optional_tensor(x, shape.clone(), want_x)?;
    let y = make_optional_tensor(y, shape.clone(), want_y)?;
    let z = make_optional_tensor(z, shape, want_z)?;
    Ok((x, y, z))
}

fn make_optional_tensor(data: Vec<f64>, shape: Vec<usize>, wanted: bool) -> BuiltinResult<Tensor> {
    if wanted {
        Tensor::new(data, shape).map_err(|err| sphere_error(&SPHERE_ERROR_INTERNAL, err))
    } else {
        Tensor::new(Vec::new(), vec![0, 0]).map_err(|err| sphere_error(&SPHERE_ERROR_INTERNAL, err))
    }
}

fn sphere_theta(col: usize, n: usize) -> f64 {
    if n == 0 {
        -std::f64::consts::PI
    } else {
        -std::f64::consts::PI + 2.0 * std::f64::consts::PI * (col as f64) / (n as f64)
    }
}

fn sphere_phi(row: usize, n: usize) -> f64 {
    if n == 0 {
        -std::f64::consts::FRAC_PI_2
    } else {
        -std::f64::consts::FRAC_PI_2 + std::f64::consts::PI * (row as f64) / (n as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    async fn sphere_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
        super::sphere_builtin(
            crate::context::legacy::active(),
            crate::builtins::common::test_support::builtin_arguments(args),
        )
        .await
    }

    use futures::executor::block_on;

    fn call(args: Vec<Value>) -> BuiltinResult<Value> {
        block_on(sphere_builtin(args))
    }

    fn output_tensors(value: Value) -> Vec<Tensor> {
        let Value::OutputList(outputs) = value else {
            panic!("expected output list, got {value:?}");
        };
        outputs
            .try_into_owned()
            .expect("unique unbounded sphere fixture outputs")
            .into_iter()
            .map(|value| match value {
                Value::Tensor(tensor) => tensor,
                other => panic!("expected tensor output, got {other:?}"),
            })
            .collect()
    }

    #[test]
    fn sphere_default_single_output_shape() {
        let value = call(vec![]).expect("sphere");
        let Value::Tensor(tensor) = value else {
            panic!("expected X tensor");
        };
        assert_eq!(tensor.shape(), &[21, 21]);
    }

    #[test]
    fn sphere_n_reads_typed_integer_tensor_exactly() {
        let n = runmat_value::Tensor::new_integer(
            runmat_value::IntegerStorage::U64(vec![42]),
            vec![1, 1],
        )
        .expect("typed n");
        assert_eq!(block_on(parse_n_value(&Value::Tensor(n))).unwrap(), 42);

        let negative = runmat_value::Tensor::new_integer(
            runmat_value::IntegerStorage::I64(vec![-1]),
            vec![1, 1],
        )
        .expect("negative n");
        assert!(block_on(parse_n_value(&Value::Tensor(negative))).is_err());

        let too_large = runmat_value::Tensor::new_integer(
            runmat_value::IntegerStorage::U64(vec![u64::MAX]),
            vec![1, 1],
        )
        .expect("large n");
        assert!(block_on(parse_n_value(&Value::Tensor(too_large))).is_err());
    }

    #[test]
    fn sphere_typed_integer_dimension_is_gated_before_generation() {
        let n = runmat_value::Tensor::new_integer(
            runmat_value::IntegerStorage::U8(vec![4]),
            vec![1, 1],
        )
        .expect("typed n");
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let err = call(vec![Value::Tensor(n)]).expect_err("typed n must be gated");
        assert_eq!(
            err.identifier(),
            Some("RunMat:compatibility:SphereTypedFaceCountExtension")
        );
    }

    #[test]
    fn sphere_three_outputs_are_unit_sphere_coordinates() {
        let _guard = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
        let outputs = output_tensors(call(vec![Value::Num(4.0)]).expect("sphere"));
        assert_eq!(outputs.len(), 3);
        for tensor in &outputs {
            assert_eq!(tensor.shape(), &[5, 5]);
        }

        let x = &outputs[0];
        let y = &outputs[1];
        let z = &outputs[2];
        for i in 0..x.materialize_f64().len() {
            let radius = (x.materialize_f64()[i] * x.materialize_f64()[i]
                + y.materialize_f64()[i] * y.materialize_f64()[i]
                + z.materialize_f64()[i] * z.materialize_f64()[i])
                .sqrt();
            assert!(
                (radius - 1.0).abs() < 1e-12,
                "point {i} is not on unit sphere: {radius}"
            );
        }

        let rows = 5;
        assert_eq!(z.materialize_f64()[0], -1.0);
        assert_eq!(z.materialize_f64()[rows - 1], 1.0);
        for row in 0..rows {
            assert!((x.materialize_f64()[row] - x.materialize_f64()[row + 4 * rows]).abs() < 1e-12);
            assert!((y.materialize_f64()[row] - y.materialize_f64()[row + 4 * rows]).abs() < 1e-12);
            assert!((z.materialize_f64()[row] - z.materialize_f64()[row + 4 * rows]).abs() < 1e-12);
        }
    }

    #[test]
    fn sphere_two_outputs_return_x_and_y_only() {
        let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let outputs = output_tensors(call(vec![Value::Num(3.0)]).expect("sphere"));
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].shape(), &[4, 4]);
        assert_eq!(outputs[1].shape(), &[4, 4]);
    }

    #[test]
    fn sphere_zero_faces_is_single_unit_point() {
        let _guard = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
        let outputs = output_tensors(call(vec![Value::Num(0.0)]).expect("sphere"));
        assert_eq!(outputs[0].shape(), &[1, 1]);
        assert_eq!(outputs[0].materialize_f64(), vec![0.0]);
        assert_eq!(outputs[1].materialize_f64(), vec![0.0]);
        assert_eq!(outputs[2].materialize_f64(), vec![-1.0]);
    }

    #[test]
    fn sphere_statement_form_plots_and_returns_no_outputs() {
        let _plot_guard = crate::builtins::plotting::lock_plot_test_context();
        crate::builtins::plotting::reset_plot_state();
        let _output_guard =
            crate::output_count::push_output_count(Some(0)).expect("output-count scope");
        let value = call(vec![Value::Num(4.0)]).expect("sphere plot");
        assert!(matches!(value, Value::OutputList(outputs) if outputs.is_empty()));
        let figure = crate::builtins::plotting::clone_figure(
            crate::builtins::plotting::current_figure_handle(),
        )
        .expect("figure");
        assert!(figure.axis_equal);
    }

    #[test]
    fn sphere_rejects_invalid_n() {
        let err = call(vec![Value::Num(2.5)]).unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:sphere:InvalidArgument"));

        let err = call(vec![Value::Num(-1.0)]).unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:sphere:InvalidArgument"));

        let err = call(vec![Value::Num(f64::INFINITY)]).unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:sphere:InvalidArgument"));

        let boundary = if usize::BITS == 64 {
            usize::MAX as f64
        } else {
            (usize::MAX as f64) + 1.0
        };
        let err = call(vec![Value::Num(boundary)]).unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:sphere:InvalidArgument"));
    }

    #[test]
    fn sphere_rejects_too_many_args_and_outputs() {
        let err = call(vec![Value::Num(1.0), Value::Num(2.0)]).unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:sphere:InvalidArgument"));

        let _guard = crate::output_count::push_output_count(Some(4)).expect("output-count scope");
        let err = call(vec![]).unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:sphere:OutputCount"));
    }

    #[test]
    fn sphere_rejects_unbounded_grid_size() {
        let err = call(vec![Value::Num(10_000.0)]).unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:sphere:OutOfMemory"));
    }

    #[test]
    fn sphere_type_infers_default_and_literal_shapes() {
        use runmat_types::*;
        for (arguments, literals, size) in [
            (vec![], vec![], 21),
            (
                vec![ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                }))],
                vec![LiteralValue::Number(4.0)],
                5,
            ),
        ] {
            let request = CallRequest {
                arguments,
                literals: LiteralContext::new(literals),
                outputs: OutputSelection::new(RequestedOutputCount::Exactly(3)),
            };
            let result = runmat_builtins::infer_catalog_call(
                &runmat_builtins::SPHERE_CATALOG_ENTRY,
                &request,
            );
            assert!(result.diagnostics.is_empty());
            assert_eq!(result.outputs.len(), 3);
            assert!(result.outputs.iter().all(|fact| fact.shape
                == ShapeFact::from(vec![Some(size), Some(size)])
                && fact.residency == ResidencyFact::Host));
        }
    }
}
