//! MATLAB-compatible `sphere` builtin.

use runmat_builtins::shape_rules::element_count_if_known;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    LiteralValue, ResolveContext, Type,
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
const DEFAULT_N: usize = 20;
const MAX_COORDINATE_ELEMENTS: usize = 4_000_000;

const SPHERE_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];
const SPHERE_INPUTS_N: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "n",
    ty: BuiltinParamType::IntegerScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("20"),
    description: "Number of surface faces per dimension; outputs have size (n+1)-by-(n+1).",
}];

const SPHERE_OUTPUT_X: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "X coordinates on the unit sphere.",
}];
const SPHERE_OUTPUT_XY: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X coordinates on the unit sphere.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y coordinates on the unit sphere.",
    },
];
const SPHERE_OUTPUT_XYZ: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X coordinates on the unit sphere.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y coordinates on the unit sphere.",
    },
    BuiltinParamDescriptor {
        name: "Z",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Z coordinates on the unit sphere.",
    },
];

const SPHERE_SIGNATURES: [BuiltinSignatureDescriptor; 8] = [
    BuiltinSignatureDescriptor {
        label: "sphere()",
        inputs: &SPHERE_INPUTS_NONE,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "sphere(n)",
        inputs: &SPHERE_INPUTS_N,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "X = sphere()",
        inputs: &SPHERE_INPUTS_NONE,
        outputs: &SPHERE_OUTPUT_X,
    },
    BuiltinSignatureDescriptor {
        label: "X = sphere(n)",
        inputs: &SPHERE_INPUTS_N,
        outputs: &SPHERE_OUTPUT_X,
    },
    BuiltinSignatureDescriptor {
        label: "[X,Y] = sphere()",
        inputs: &SPHERE_INPUTS_NONE,
        outputs: &SPHERE_OUTPUT_XY,
    },
    BuiltinSignatureDescriptor {
        label: "[X,Y] = sphere(n)",
        inputs: &SPHERE_INPUTS_N,
        outputs: &SPHERE_OUTPUT_XY,
    },
    BuiltinSignatureDescriptor {
        label: "[X,Y,Z] = sphere()",
        inputs: &SPHERE_INPUTS_NONE,
        outputs: &SPHERE_OUTPUT_XYZ,
    },
    BuiltinSignatureDescriptor {
        label: "[X,Y,Z] = sphere(n)",
        inputs: &SPHERE_INPUTS_N,
        outputs: &SPHERE_OUTPUT_XYZ,
    },
];

const SPHERE_ERROR_OUTPUT_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPHERE.OUTPUT_COUNT",
    identifier: Some("RunMat:sphere:OutputCount"),
    when: "More than three outputs are requested.",
    message: "sphere: too many output arguments; maximum is 3",
};
const SPHERE_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPHERE.ARG_COUNT",
    identifier: Some("RunMat:sphere:InvalidArgument"),
    when: "More than one input argument is supplied.",
    message: "sphere: expected zero or one input argument",
};
const SPHERE_ERROR_INVALID_N: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPHERE.INVALID_N",
    identifier: Some("RunMat:sphere:InvalidArgument"),
    when: "The n argument is not a finite non-negative integer scalar.",
    message: "sphere: n must be a finite non-negative integer scalar",
};
const SPHERE_ERROR_TOO_LARGE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPHERE.N_TOO_LARGE",
    identifier: Some("RunMat:sphere:OutOfMemory"),
    when: "The requested coordinate grid is too large for RunMat's bounded allocation policy.",
    message: "sphere: requested grid is too large",
};
const SPHERE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPHERE.INTERNAL",
    identifier: Some("RunMat:sphere:Internal"),
    when: "Coordinate tensor construction or plotting fails unexpectedly.",
    message: "sphere: internal operation failed",
};
const SPHERE_ERRORS: [BuiltinErrorDescriptor; 5] = [
    SPHERE_ERROR_OUTPUT_COUNT,
    SPHERE_ERROR_ARG_COUNT,
    SPHERE_ERROR_INVALID_N,
    SPHERE_ERROR_TOO_LARGE,
    SPHERE_ERROR_INTERNAL,
];

pub const SPHERE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SPHERE_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SPHERE_ERRORS,
};

const SPHERE_TYPED_N_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "sphere-typed-face-count",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "sphere with a typed-integer face count is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:SphereTypedFaceCountExtension"),
};
pub const SPHERE_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [SPHERE_TYPED_N_EXTENSION];
const SPHERE_INTEGER_N_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "n",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target calls n a positive integer but does not enumerate typed storage classes. RunMat accepts every exact nonnegative integer class behind a compatibility gate; ordinary host double integer values remain documented behavior.",
    }];
pub const SPHERE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "[X,Y,Z] = sphere(integer_n)",
        inputs: &SPHERE_INTEGER_N_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The typed face count is a gated RunMat extension. It is decoded exactly, checked against allocation bounds, and controls only output shape; coordinate generation and statement-form plotting use host double values.",
    }];

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

fn sphere_type(args: &[Type], ctx: &ResolveContext) -> Type {
    match args {
        [] => Type::tensor_with_shape(vec![DEFAULT_N + 1, DEFAULT_N + 1]),
        [arg] => sphere_n_type(arg, ctx),
        _ => Type::Unknown,
    }
}

fn sphere_n_type(arg: &Type, ctx: &ResolveContext) -> Type {
    if let Some(n) = sphere_literal_n(ctx) {
        if let Some(size) = n.checked_add(1) {
            return Type::Tensor {
                shape: Some(vec![Some(size), Some(size)]),
            };
        }
        return Type::Unknown;
    }

    match arg {
        Type::Num | Type::Int | Type::Bool => Type::Tensor {
            shape: Some(vec![None, None]),
        },
        Type::Tensor { shape: Some(shape) } | Type::Logical { shape: Some(shape) } => {
            let element_count = element_count_if_known(shape);
            if element_count != Some(1) && element_count.is_some() {
                Type::Unknown
            } else {
                Type::Tensor {
                    shape: Some(vec![None, None]),
                }
            }
        }
        Type::Tensor { shape: None } | Type::Logical { shape: None } => Type::Tensor {
            shape: Some(vec![None, None]),
        },
        Type::Unknown => Type::Unknown,
        _ => Type::Unknown,
    }
}

fn sphere_literal_n(ctx: &ResolveContext) -> Option<usize> {
    match ctx.literal_args.first() {
        Some(LiteralValue::Number(value)) => parse_n_number(*value).ok(),
        Some(LiteralValue::Bool(value)) => Some(usize::from(*value)),
        _ => None,
    }
}

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
    category = "plotting",
    summary = "Generate or plot unit-sphere coordinate grids.",
    keywords = "sphere,plotting,3d,surface,unit sphere",
    sink = true,
    suppress_auto_output = true,
    type_resolver(sphere_type),
    descriptor(crate::builtins::plotting::sphere::SPHERE_DESCRIPTOR),
    extensions(crate::builtins::plotting::sphere::SPHERE_EXTENSIONS),
    integer_capabilities(crate::builtins::plotting::sphere::SPHERE_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::plotting::sphere"
)]
async fn sphere_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if matches!(crate::output_count::current_output_count(), Some(n) if n > 3) {
        return Err(sphere_error(
            &SPHERE_ERROR_OUTPUT_COUNT,
            "requested output count exceeds [X,Y,Z]",
        ));
    }

    if let [value] = args.as_slice() {
        if crate::builtins::common::validation::value_has_native_integer_class(value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &SPHERE_TYPED_N_EXTENSION,
                BUILTIN_NAME,
            )?;
        }
    }

    let n = match args.as_slice() {
        [] => DEFAULT_N,
        [value] => parse_n_value(value).await?,
        _ => {
            return Err(sphere_error(
                &SPHERE_ERROR_ARG_COUNT,
                "use sphere or sphere(n)",
            ))
        }
    };

    if is_plot_form_call() {
        let (x, y, z) = make_sphere_tensors(n, true, true, true)?;
        render_sphere_plot(x, y, z)?;
        return Ok(Value::OutputList(Vec::new()));
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
        Some(0) => Ok(Value::OutputList(Vec::new())),
        Some(1) => {
            let (x, _, _) = make_sphere_tensors(n, true, false, false)?;
            Ok(Value::OutputList(vec![Value::Tensor(x)]))
        }
        Some(2) => {
            let (x, y, _) = make_sphere_tensors(n, true, true, false)?;
            Ok(Value::OutputList(vec![Value::Tensor(x), Value::Tensor(y)]))
        }
        Some(3) => {
            let (x, y, z) = make_sphere_tensors(n, true, true, true)?;
            Ok(Value::OutputList(vec![
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

fn render_sphere_plot(x: Tensor, y: Tensor, z: Tensor) -> BuiltinResult<()> {
    let x_grid = tensor_to_plot_grid(x)?;
    let y_grid = tensor_to_plot_grid(y)?;
    let z_grid = tensor_to_plot_grid(z)?;
    let mut plot = Some(
        SurfacePlot::from_coordinate_grids(x_grid, y_grid, z_grid)
            .map_err(|err| sphere_error(&SPHERE_ERROR_INTERNAL, err))?,
    );
    let opts = PlotRenderOptions {
        title: "Sphere",
        x_label: "X",
        y_label: "Y",
        axis_equal: true,
        ..Default::default()
    };
    let render_result = render_active_plot(BUILTIN_NAME, opts, move |figure, axes| {
        figure.add_surface_plot_on_axes(plot.take().expect("sphere plot consumed once"), axes);
        Ok(())
    });

    if let Err(err) = render_result {
        let lower = err.to_string().to_lowercase();
        if lower.contains("plotting is unavailable") || lower.contains("non-main thread") {
            return Ok(());
        }
        return Err(sphere_error(
            &SPHERE_ERROR_INTERNAL,
            format!("could not render sphere surface: {}", err.message),
        ));
    }
    Ok(())
}

fn tensor_to_plot_grid(tensor: Tensor) -> BuiltinResult<Vec<Vec<f64>>> {
    let rows = tensor.rows;
    let cols = tensor.cols;
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
    if (rounded - raw).abs() > 1e-6
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
    use futures::executor::block_on;

    fn call(args: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::sphere_builtin(args))
    }

    fn output_tensors(value: Value) -> Vec<Tensor> {
        let Value::OutputList(outputs) = value else {
            panic!("expected output list, got {value:?}");
        };
        outputs
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
        assert_eq!(tensor.shape, vec![21, 21]);
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
        let _guard = crate::output_count::push_output_count(Some(3));
        let outputs = output_tensors(call(vec![Value::Num(4.0)]).expect("sphere"));
        assert_eq!(outputs.len(), 3);
        for tensor in &outputs {
            assert_eq!(tensor.shape, vec![5, 5]);
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
        let _guard = crate::output_count::push_output_count(Some(2));
        let outputs = output_tensors(call(vec![Value::Num(3.0)]).expect("sphere"));
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].shape, vec![4, 4]);
        assert_eq!(outputs[1].shape, vec![4, 4]);
    }

    #[test]
    fn sphere_zero_faces_is_single_unit_point() {
        let _guard = crate::output_count::push_output_count(Some(3));
        let outputs = output_tensors(call(vec![Value::Num(0.0)]).expect("sphere"));
        assert_eq!(outputs[0].shape, vec![1, 1]);
        assert_eq!(outputs[0].materialize_f64(), vec![0.0]);
        assert_eq!(outputs[1].materialize_f64(), vec![0.0]);
        assert_eq!(outputs[2].materialize_f64(), vec![-1.0]);
    }

    #[test]
    fn sphere_statement_form_plots_and_returns_no_outputs() {
        let _plot_guard = crate::builtins::plotting::lock_plot_test_context();
        crate::builtins::plotting::reset_plot_state();
        let _output_guard = crate::output_count::push_output_count(Some(0));
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

        let _guard = crate::output_count::push_output_count(Some(4));
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
        assert_eq!(
            sphere_type(&[], &ResolveContext::new(Vec::new())),
            Type::Tensor {
                shape: Some(vec![Some(21), Some(21)]),
            }
        );
        assert_eq!(
            sphere_type(
                &[Type::Num],
                &ResolveContext::new(vec![LiteralValue::Number(4.0)])
            ),
            Type::Tensor {
                shape: Some(vec![Some(5), Some(5)]),
            }
        );
    }
}
