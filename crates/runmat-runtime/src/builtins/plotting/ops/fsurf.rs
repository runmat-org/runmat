//! MATLAB-compatible `fsurf` builtin.
use runmat_builtins::{BuiltinErrorDescriptor, BuiltinExtensionDescriptor};
pub use runmat_builtins::{
    FSURF_ERROR_EVALUATION, FSURF_ERROR_INTERNAL, FSURF_ERROR_INVALID_ARGUMENT,
    FSURF_INTEGER_AXES_EXTENSION as INTEGER_AXES_EXTENSION,
    FSURF_INTEGER_CALLBACK_OUTPUT_EXTENSION as INTEGER_CALLBACK_OUTPUT_EXTENSION,
    FSURF_INTEGER_DOMAIN_EXTENSION as INTEGER_DOMAIN_EXTENSION,
    FSURF_INTEGER_MESH_DENSITY_EXTENSION as INTEGER_MESH_DENSITY_EXTENSION,
    FSURF_INTEGER_STYLE_PROPERTY_EXTENSION as INTEGER_STYLE_PROPERTY_EXTENSION,
    FSURF_LOGICAL_NUMERIC_INPUT_EXTENSION as LOGICAL_NUMERIC_INPUT_EXTENSION,
    FSURF_RESIDENT_CALLBACK_OUTPUT_EXTENSION as RESIDENT_CALLBACK_OUTPUT_EXTENSION,
    FSURF_SINGLE_NUMERIC_INPUT_EXTENSION as SINGLE_NUMERIC_INPUT_EXTENSION,
};

use runmat_macros::runtime_builtin;
use runmat_plot::plots::{ColorMap, ShadingMode, SurfacePlot};
use runmat_value::{IntValue, NumericDType, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor as tensor_utils;

use crate::{build_runtime_error, BuiltinResult, RuntimeError};

use super::op_common::{apply_axes_target, split_leading_axes_arguments, AxesTarget};
use super::plotting_error;
use super::state::PlotRenderOptions;
use super::state::{
    current_figure_handle, register_function_surface_handle, render_active_plot,
    FunctionSurfaceFunctionRef, FunctionSurfaceFunctionState,
};
use super::style::{parse_surface_style_args, value_as_string, SurfaceStyleDefaults};

const BUILTIN_NAME: &str = "fsurf";
const DEFAULT_DOMAIN: Domain = Domain {
    x_min: -5.0,
    x_max: 5.0,
    y_min: -5.0,
    y_max: 5.0,
};
const DEFAULT_MESH_DENSITY: usize = 35;
const MAX_MESH_DENSITY: usize = 400;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::fsurf")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "fsurf",
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
    notes: "fsurf samples arbitrary MATLAB function handles on the host, then renders through the existing surface plot pipeline.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::plotting::fsurf")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "fsurf",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "fsurf performs callback sampling and terminates fusion graphs.",
};

#[runtime_builtin(
    name = "fsurf",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::fsurf"
)]
pub async fn fsurf_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<f64> {
    if matches!(crate::output_count::current_output_count(), Some(n) if n > 1) {
        return Err(crate::runtime_error::semantic_error(
            "RunMat:plotting:TooManyOutputs",
            "fsurf: expected at most one output",
        ));
    }
    let work = super::coordinate_admission::CoordinateWork::admit(runtime.as_ref(), "fsurf")?;
    let _arguments = args.fork();
    work.finish(fsurf_impl(runtime.as_ref(), args, &work).await)
}

async fn fsurf_impl(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
    work: &super::coordinate_admission::CoordinateWork<'_>,
) -> crate::BuiltinResult<f64> {
    preflight_fsurf_extensions(args.view())?;
    let parsed = parse_fsurf_args(args).map_err(map_fsurf_invalid)?;
    let surface = sample_surface(&parsed, runtime, work).await?;
    render_fsurf(surface, &parsed, runtime, work).map_err(map_fsurf_internal)
}

fn preflight_fsurf_extensions(args: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<()> {
    gate_typed_integer_axes_alias(args)?;
    let mut values = args
        .iter()
        .skip_while(|value| !is_function_handle(value))
        .skip_while(|value| is_function_handle(value));
    if let Some(value) = values
        .next()
        .filter(|value| !matches!(value, Value::String(_) | Value::CharArray(_)))
    {
        if is_typed_integer_value(value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &INTEGER_DOMAIN_EXTENSION,
                BUILTIN_NAME,
            )?;
        }
        if matches!(value, Value::Bool(_) | Value::LogicalArray(_)) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &LOGICAL_NUMERIC_INPUT_EXTENSION,
                BUILTIN_NAME,
            )?;
        }
        if matches!(value, Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &SINGLE_NUMERIC_INPUT_EXTENSION,
                BUILTIN_NAME,
            )?;
        }
    }
    for index in 0..args.len().saturating_sub(1) {
        let pair = args.range(index..index + 2)?;
        let Some(key) = value_as_string(&pair[0]).map_err(|source| {
            crate::builtins::plotting::plotting_error_with_source(
                "fsurf",
                source.to_string(),
                source,
            )
        })?
        else {
            continue;
        };
        if key.trim().eq_ignore_ascii_case("MeshDensity") {
            if is_typed_integer_value(&pair[1]) {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &INTEGER_MESH_DENSITY_EXTENSION,
                    BUILTIN_NAME,
                )?;
            } else if matches!(&pair[1], Value::Bool(_) | Value::LogicalArray(_)) {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &LOGICAL_NUMERIC_INPUT_EXTENSION,
                    BUILTIN_NAME,
                )?;
            } else if matches!(&pair[1], Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32)
            {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &SINGLE_NUMERIC_INPUT_EXTENSION,
                    BUILTIN_NAME,
                )?;
            }
        } else {
            preflight_surface_style_extension(key.trim(), &pair[1])?;
        }
    }
    Ok(())
}

fn preflight_surface_style_extension(key: &str, value: &Value) -> BuiltinResult<()> {
    let key = key.to_ascii_lowercase();
    if is_typed_integer_value(value) {
        match key.as_str() {
            "facealpha" | "alpha" | "facecolor" | "colormap" => {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &INTEGER_STYLE_PROPERTY_EXTENSION,
                    BUILTIN_NAME,
                )?;
                ensure_integer_value_exact_f64(value, "surface style property")?;
            }
            "flattenz" | "lighting" | "visible" => {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &INTEGER_STYLE_PROPERTY_EXTENSION,
                    BUILTIN_NAME,
                )?;
            }
            _ => {}
        }
    } else if matches!(value, Value::Bool(_) | Value::LogicalArray(_)) {
        if matches!(key.as_str(), "flattenz" | "lighting" | "visible") {
            crate::compatibility::ensure_builtin_extension_enabled(
                &LOGICAL_NUMERIC_INPUT_EXTENSION,
                BUILTIN_NAME,
            )?;
        }
    } else if matches!(value, Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32)
        && matches!(
            key.as_str(),
            "facealpha" | "alpha" | "facecolor" | "colormap"
        )
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &SINGLE_NUMERIC_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(())
}

fn is_typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

fn gate_typed_integer_axes_alias(args: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<()> {
    let Some(first) = args.first() else {
        return Ok(());
    };
    if !is_typed_integer_value(first) {
        return Ok(());
    }
    let Ok(super::properties::PlotHandle::Axes(handle, axes_index)) =
        super::properties::resolve_plot_handle(first, BUILTIN_NAME)
    else {
        return Ok(());
    };
    let expected = super::state::encode_axes_handle(handle, axes_index) as u64;
    let actual = exact_integer_scalar(first).and_then(|value| value.try_to_u64());
    if actual != Some(expected) {
        return Err(fsurf_invalid(
            "typed-integer axes alias must equal the canonical encoded axes handle",
        ));
    }
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &INTEGER_AXES_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(())
}

fn ensure_integer_value_exact_f64(value: &Value, role: &str) -> BuiltinResult<()> {
    match value {
        Value::Int(value) => checked_integer_f64(value, role).map(|_| ()),
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => tensor
            .integer_storage()
            .expect("checked integer storage")
            .exact_values()
            .into_iter()
            .try_for_each(|value| checked_integer_f64(&value, role).map(|_| ())),
        _ => Ok(()),
    }
}

#[derive(Clone)]
enum SurfaceFunction {
    Explicit(Value),
    Parametric {
        x: Box<Value>,
        y: Box<Value>,
        z: Box<Value>,
    },
}

#[derive(Clone, Copy)]
struct Domain {
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
}

struct ParsedFsurf {
    target_axes: AxesTarget,
    function: SurfaceFunction,
    domain: Domain,
    mesh_density: usize,
    style_args: Vec<Value>,
}

fn fsurf_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", error.message, detail.as_ref()))
        .with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn fsurf_invalid(detail: impl AsRef<str>) -> RuntimeError {
    fsurf_error_with_detail(&FSURF_ERROR_INVALID_ARGUMENT, detail)
}

fn map_fsurf_invalid(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    let mapped = fsurf_error_with_detail(&FSURF_ERROR_INVALID_ARGUMENT, err.message.clone());
    let mut builder = crate::build_runtime_error(mapped.message)
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = mapped.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_fsurf_eval(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    let mapped = fsurf_error_with_detail(&FSURF_ERROR_EVALUATION, err.message.clone());
    let mut builder = crate::build_runtime_error(mapped.message)
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = mapped.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_fsurf_internal(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    let mapped = fsurf_error_with_detail(&FSURF_ERROR_INTERNAL, err.message.clone());
    let mut builder = crate::build_runtime_error(mapped.message)
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = mapped.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn parse_fsurf_args(args: crate::call::BuiltinArguments) -> BuiltinResult<ParsedFsurf> {
    if args.is_empty() {
        return Err(fsurf_invalid("expected a function handle"));
    }

    let (target_axes, mut values) = split_leading_axes_arguments(args, BUILTIN_NAME)?;
    if values.is_empty() {
        return Err(fsurf_invalid(
            "expected a function handle after axes handle",
        ));
    }

    let mut function_values = [None, None, None];
    for slot in &mut function_values {
        if !values.first().is_some_and(is_function_handle) {
            break;
        }
        *slot = values.pop_front();
    }

    let function = match function_values {
        [Some(function), None, None] => SurfaceFunction::Explicit(function),
        [Some(x), Some(y), Some(z)] => SurfaceFunction::Parametric {
            x: Box::new(x),
            y: Box::new(y),
            z: Box::new(z),
        },
        [None, None, None] => return Err(fsurf_invalid("expected a function handle")),
        _ => {
            return Err(fsurf_invalid(
                "expected either one function handle or three parametric function handles",
            ))
        }
    };

    let mut domain = DEFAULT_DOMAIN;
    if values.first().is_some_and(is_numeric_domain_value) {
        domain = parse_domain(&values.pop_front().expect("checked argument prefix"))?;
    }
    let (mesh_density, style_args) = split_mesh_density(values)?;

    Ok(ParsedFsurf {
        target_axes,
        function,
        domain,
        mesh_density,
        style_args,
    })
}

fn is_function_handle(value: &Value) -> bool {
    matches!(
        value,
        Value::ScopedFunctionHandle(_)
            | Value::FunctionHandle(_)
            | Value::ExternalFunctionHandle(_)
            | Value::MethodFunctionHandle(_)
            | Value::BoundFunctionHandle(_)
            | Value::Closure(_)
    )
}

fn is_numeric_domain_value(value: &Value) -> bool {
    matches!(
        value,
        Value::Tensor(_) | Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::LogicalArray(_)
    )
}

fn parse_domain(value: &Value) -> BuiltinResult<Domain> {
    let values = numeric_vector(value)?;
    match values.as_slice() {
        [lo, hi] => domain_from_values(*lo, *hi, *lo, *hi),
        [x_min, x_max, y_min, y_max] => domain_from_values(*x_min, *x_max, *y_min, *y_max),
        _ => Err(fsurf_invalid(
            "domain must be a two-element or four-element numeric vector",
        )),
    }
}

fn domain_from_values(x_min: f64, x_max: f64, y_min: f64, y_max: f64) -> BuiltinResult<Domain> {
    if ![x_min, x_max, y_min, y_max]
        .iter()
        .all(|value| value.is_finite())
    {
        return Err(fsurf_invalid("domain limits must be finite"));
    }
    if x_min >= x_max || y_min >= y_max {
        return Err(fsurf_invalid(
            "domain lower bounds must be less than upper bounds",
        ));
    }
    Ok(Domain {
        x_min,
        x_max,
        y_min,
        y_max,
    })
}

fn numeric_vector(value: &Value) -> BuiltinResult<Vec<f64>> {
    match value {
        Value::Num(n) => Ok(vec![*n]),
        Value::Int(i) => Ok(vec![checked_integer_f64(i, "domain bound")?]),
        Value::Bool(b) => Ok(vec![if *b { 1.0 } else { 0.0 }]),
        Value::LogicalArray(array) => Ok(array
            .data
            .iter()
            .map(|value| if *value == 0 { 0.0 } else { 1.0 })
            .collect()),
        Value::Tensor(tensor) => {
            if let Some(storage) = tensor.integer_storage() {
                storage
                    .exact_values()
                    .into_iter()
                    .map(|value| checked_integer_f64(&value, "domain bound"))
                    .collect()
            } else {
                Ok(tensor_utils::tensor_values_f64(tensor))
            }
        }
        other => Err(fsurf_invalid(format!(
            "expected numeric domain vector, got {other:?}"
        ))),
    }
}

fn checked_integer_f64(value: &IntValue, role: &str) -> BuiltinResult<f64> {
    if crate::builtins::common::validation::integer_is_exact_f64(value) {
        Ok(value.to_f64())
    } else {
        Err(fsurf_invalid(format!(
            "integer {role} must be exactly representable as double"
        )))
    }
}

fn split_mesh_density(args: crate::call::BuiltinArguments) -> BuiltinResult<(usize, Vec<Value>)> {
    let mut mesh_density = DEFAULT_MESH_DENSITY;
    let mut style_args = Vec::new();
    let mut idx = 0usize;
    while idx < args.len() {
        let Some(key) = value_as_string(&args[idx]).map_err(|source| {
            crate::builtins::plotting::plotting_error_with_source(
                "fsurf",
                source.to_string(),
                source,
            )
        })?
        else {
            return Err(fsurf_invalid("name-value option names must be strings"));
        };
        if idx + 1 >= args.len() {
            return Err(fsurf_invalid("name-value arguments must come in pairs"));
        }
        if key.trim().eq_ignore_ascii_case("MeshDensity") {
            mesh_density = parse_mesh_density(&args[idx + 1])?;
        } else {
            style_args.push(args[idx].clone());
            style_args.push(args[idx + 1].clone());
        }
        idx += 2;
    }
    Ok((mesh_density, style_args))
}

fn parse_mesh_density(value: &Value) -> BuiltinResult<usize> {
    if let Some(count) = exact_integer_scalar(value) {
        return parse_mesh_density_integer(&count);
    }
    let values = numeric_vector(value)?;
    if values.len() != 1 {
        return Err(fsurf_invalid("MeshDensity must be a scalar"));
    }
    let raw = values[0];
    if !raw.is_finite() {
        return Err(fsurf_invalid("MeshDensity must be finite"));
    }
    let rounded = raw.round();
    if (rounded - raw).abs() > 1.0e-9
        || rounded < 2.0
        || rounded > usize::MAX as f64
        || (usize::BITS == 64 && rounded == usize::MAX as f64)
    {
        return Err(fsurf_invalid(
            "MeshDensity must be an integer greater than or equal to 2",
        ));
    }
    let density = rounded as usize;
    if density > MAX_MESH_DENSITY {
        return Err(fsurf_invalid(format!(
            "MeshDensity must be at most {MAX_MESH_DENSITY}"
        )));
    }
    Ok(density)
}

fn exact_integer_scalar(value: &Value) -> Option<IntValue> {
    match value {
        Value::Int(value) => Some(value.clone()),
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(tensor) => tensor
            .integer_storage()
            .and_then(|storage| storage.value_at(0)),
        _ => None,
    }
}

fn parse_mesh_density_integer(value: &IntValue) -> BuiltinResult<usize> {
    let Some(density) = value.try_to_usize() else {
        return Err(fsurf_invalid(
            "MeshDensity must be an integer greater than or equal to 2",
        ));
    };
    if density < 2 {
        return Err(fsurf_invalid(
            "MeshDensity must be an integer greater than or equal to 2",
        ));
    }
    if density > MAX_MESH_DENSITY {
        return Err(fsurf_invalid(format!(
            "MeshDensity must be at most {MAX_MESH_DENSITY}"
        )));
    }
    Ok(density)
}

async fn sample_surface(
    parsed: &ParsedFsurf,
    runtime: Option<&crate::context::RuntimeContext>,
    work: &super::coordinate_admission::CoordinateWork<'_>,
) -> BuiltinResult<SurfacePlot> {
    let x_axis = linspace(
        parsed.domain.x_min,
        parsed.domain.x_max,
        parsed.mesh_density,
    );
    let y_axis = linspace(
        parsed.domain.y_min,
        parsed.domain.y_max,
        parsed.mesh_density,
    );
    let mut z_grid = vec![vec![0.0; y_axis.len()]; x_axis.len()];
    let mut x_grid = Vec::new();
    let mut y_grid = Vec::new();

    match &parsed.function {
        SurfaceFunction::Explicit(function) => {
            for (i, &x) in x_axis.iter().enumerate() {
                for (j, &y) in y_axis.iter().enumerate() {
                    z_grid[i][j] = call_surface_function(runtime, work, function, x, y).await?;
                }
            }
        }
        SurfaceFunction::Parametric { x, y, z } => {
            x_grid = vec![vec![0.0; y_axis.len()]; x_axis.len()];
            y_grid = vec![vec![0.0; y_axis.len()]; x_axis.len()];
            for (i, &u) in x_axis.iter().enumerate() {
                for (j, &v) in y_axis.iter().enumerate() {
                    x_grid[i][j] = call_surface_function(runtime, work, x, u, v).await?;
                    y_grid[i][j] = call_surface_function(runtime, work, y, u, v).await?;
                    z_grid[i][j] = call_surface_function(runtime, work, z, u, v).await?;
                }
            }
        }
    }

    let defaults = SurfaceStyleDefaults::new(
        ColorMap::Parula,
        ShadingMode::Smooth,
        false,
        1.0,
        false,
        true,
    );
    let style = parse_surface_style_args(
        BUILTIN_NAME,
        crate::call::FunctionArgumentView::from_slice(&parsed.style_args),
        defaults,
    )
    .map_err(map_fsurf_invalid)?;
    let mut surface = match &parsed.function {
        SurfaceFunction::Explicit(_) => SurfacePlot::new(x_axis, y_axis, z_grid)
            .map_err(|err| plotting_error(BUILTIN_NAME, format!("fsurf: {err}")))?,
        SurfaceFunction::Parametric { .. } => {
            SurfacePlot::from_coordinate_grids(x_grid, y_grid, z_grid)
                .map_err(|err| plotting_error(BUILTIN_NAME, format!("fsurf: {err}")))?
        }
    };
    style.apply_to_plot(&mut surface);
    Ok(surface)
}

async fn call_surface_function(
    runtime: Option<&crate::context::RuntimeContext>,
    work: &super::coordinate_admission::CoordinateWork<'_>,
    function: &Value,
    a: f64,
    b: f64,
) -> BuiltinResult<f64> {
    work.check()?;
    let callback = crate::canonicalize_callback_handle_for_semantic_resolution(function.clone())?;
    let arguments = [Value::Num(a), Value::Num(b)];
    let value = match runtime {
        Some(runtime) => {
            crate::call_feval_async_with_outputs_in_context(runtime, callback, &arguments, 1).await
        }
        None => crate::call_feval_async_with_outputs(callback, &arguments, 1).await,
    }
    .map_err(map_fsurf_eval)?;
    work.check()?;
    if matches!(value.value(), Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &INTEGER_CALLBACK_OUTPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if matches!(value.value(), Value::GpuTensor(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &RESIDENT_CALLBACK_OUTPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    value
        .with_owned_async(|value| async move {
            let value = crate::dispatcher::gather_if_needed_async(&value)
                .await
                .map_err(map_fsurf_eval)?;
            work.check()?;
            surface_value_to_scalar(value).map_err(map_fsurf_eval)
        })
        .await
}

fn surface_value_to_scalar(value: Value) -> BuiltinResult<f64> {
    match value {
        Value::Num(value) => Ok(value),
        Value::Int(value) => checked_integer_callback_f64(&value),
        Value::Bool(value) => {
            gate_callback_extension(&LOGICAL_NUMERIC_INPUT_EXTENSION)?;
            Ok(if value { 1.0 } else { 0.0 })
        }
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(&tensor) => {
            if let Some(value) = tensor
                .integer_storage()
                .and_then(|storage| storage.value_at(0))
            {
                checked_integer_callback_f64(&value)
            } else {
                if tensor.numeric_dtype() == NumericDType::F32 {
                    gate_callback_extension(&SINGLE_NUMERIC_INPUT_EXTENSION)?;
                }
                Ok(tensor_utils::tensor_value_f64(&tensor, 0))
            }
        }
        Value::LogicalArray(array) if array.data.len() == 1 => {
            gate_callback_extension(&LOGICAL_NUMERIC_INPUT_EXTENSION)?;
            Ok(if array.data[0] != 0 { 1.0 } else { 0.0 })
        }
        other => Err(fsurf_error_with_detail(
            &FSURF_ERROR_EVALUATION,
            format!("function output must be a scalar real numeric value, got {other:?}"),
        )),
    }
}

fn gate_callback_extension(extension: &BuiltinExtensionDescriptor) -> BuiltinResult<()> {
    crate::compatibility::ensure_builtin_extension_enabled(extension, BUILTIN_NAME)
}

fn checked_integer_callback_f64(value: &IntValue) -> BuiltinResult<f64> {
    crate::compatibility::ensure_builtin_extension_enabled(
        &INTEGER_CALLBACK_OUTPUT_EXTENSION,
        BUILTIN_NAME,
    )?;
    if crate::builtins::common::validation::integer_is_exact_f64(value) {
        Ok(value.to_f64())
    } else {
        Err(fsurf_error_with_detail(
            &FSURF_ERROR_EVALUATION,
            "integer callback result must be exactly representable as double",
        ))
    }
}

fn linspace(start: f64, stop: f64, count: usize) -> Vec<f64> {
    if count <= 1 {
        return vec![start];
    }
    let step = (stop - start) / (count as f64 - 1.0);
    (0..count).map(|idx| start + step * idx as f64).collect()
}

fn render_fsurf(
    surface: SurfacePlot,
    parsed: &ParsedFsurf,
    runtime: Option<&crate::context::RuntimeContext>,
    work: &super::coordinate_admission::CoordinateWork<'_>,
) -> BuiltinResult<f64> {
    work.check()?;
    apply_axes_target(parsed.target_axes, BUILTIN_NAME)?;
    let function_state = function_surface_state(runtime, &parsed.function)?;
    let mut surface = Some(surface);
    let plot_index_out = std::rc::Rc::new(std::cell::RefCell::new(None));
    let plot_index_slot = std::rc::Rc::clone(&plot_index_out);
    let figure_handle = current_figure_handle();
    let target_axes_index = parsed.target_axes.map(|(_, axes)| axes);
    let render_result = render_active_plot(
        BUILTIN_NAME,
        PlotRenderOptions {
            axis_equal: false,
            ..Default::default()
        },
        move |figure, axes| {
            let axes = target_axes_index.unwrap_or(axes);
            let plot_index =
                figure.add_surface_plot_on_axes(surface.take().expect("fsurf consumed once"), axes);
            *plot_index_slot.borrow_mut() = Some((axes, plot_index));
            Ok(())
        },
    );
    let Some((axes, plot_index)) = *plot_index_out.borrow() else {
        return render_result.map(|_| f64::NAN);
    };
    let handle = register_function_surface_handle(
        figure_handle,
        axes,
        plot_index,
        parsed.mesh_density,
        (parsed.domain.x_min, parsed.domain.x_max),
        (parsed.domain.y_min, parsed.domain.y_max),
        function_state,
    );
    if let Err(err) = render_result {
        if crate::builtins::plotting::state::presentation_unavailable(&err) {
            return Ok(handle);
        }
        return Err(err);
    }
    Ok(handle)
}

fn function_surface_state(
    runtime: Option<&crate::context::RuntimeContext>,
    function: &SurfaceFunction,
) -> BuiltinResult<FunctionSurfaceFunctionState> {
    match function {
        SurfaceFunction::Explicit(function) => Ok(FunctionSurfaceFunctionState::Explicit(
            function_surface_ref(runtime, function, BUILTIN_NAME)?,
        )),
        SurfaceFunction::Parametric { x, y, z } => {
            let x = function_surface_ref(runtime, x, BUILTIN_NAME)?;
            let y = function_surface_ref(runtime, y, BUILTIN_NAME)?;
            let z = function_surface_ref(runtime, z, BUILTIN_NAME)?;
            Ok(FunctionSurfaceFunctionState::Parametric { x, y, z })
        }
    }
}

pub(crate) fn function_surface_ref(
    runtime: Option<&crate::context::RuntimeContext>,
    function: &Value,
    builtin: &'static str,
) -> BuiltinResult<FunctionSurfaceFunctionRef> {
    Ok(match function {
        Value::FunctionHandle(name) => {
            FunctionSurfaceFunctionRef::FunctionHandle(name.name().to_owned())
        }
        Value::ExternalFunctionHandle(name) => {
            FunctionSurfaceFunctionRef::ExternalFunctionHandle(name.name().to_owned())
        }
        Value::MethodFunctionHandle(name) => {
            FunctionSurfaceFunctionRef::MethodFunctionHandle(name.name().to_owned())
        }
        Value::BoundFunctionHandle(_) | Value::Closure(_) => {
            FunctionSurfaceFunctionRef::OwnedCallable(super::state::callbacks::retain_in_context(
                runtime,
                function.clone(),
                builtin,
            )?)
        }
        _ => FunctionSurfaceFunctionRef::FunctionHandle("<invalid>".into()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::{
        FSURF_DESCRIPTOR, FSURF_EXTENSIONS as EXTENSIONS,
        FSURF_INTEGER_CAPABILITIES as INTEGER_CAPABILITIES,
    };
    fn preflight_fsurf_extensions(args: &[Value]) -> crate::BuiltinResult<()> {
        super::preflight_fsurf_extensions(crate::call::FunctionArgumentView::from_slice(args))
    }

    async fn fsurf_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
        super::fsurf_builtin(
            crate::context::legacy::active(),
            crate::builtins::common::test_support::builtin_arguments(args),
        )
        .await
    }

    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, clone_figure, current_figure_handle};
    use futures::executor::block_on;
    use runmat_plot::plots::PlotElement;
    use std::sync::Arc;

    struct TestFunctions {
        runtime: crate::context::RuntimeContext,
        bindings: Vec<runmat_value::BoundFunctionBinding>,
        _bound: crate::user_functions::BoundFunctionInvokerGuard,
    }

    impl TestFunctions {
        fn value(&self, name: &str, function: usize) -> Value {
            let binding = self
                .bindings
                .iter()
                .find(|binding| binding.function == function)
                .expect("selected surface callback");
            crate::user_functions::test_support::exact_local_value(name, binding.clone())
        }
    }

    fn selected_functions(functions: &[usize]) -> TestFunctions {
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let bindings = functions
            .iter()
            .map(|function| {
                crate::user_functions::test_support::exact_local_binding(&runtime, *function)
            })
            .collect::<Vec<_>>();
        let entry = runtime.enter();
        let bound = crate::user_functions::test_support::install_bound_forwarders(bindings.clone());
        drop(entry);
        TestFunctions {
            runtime,
            bindings,
            _bound: bound,
        }
    }

    fn with_test_function(
        f: impl Fn(f64, f64) -> f64 + Send + Sync + 'static,
    ) -> (TestFunctions, crate::user_functions::FunctionInvokerGuard) {
        let fixture = selected_functions(&[1]);
        let entry = fixture.runtime.enter();
        let f = Arc::new(f);
        let numeric = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            move |function, args, _requested_outputs| {
                assert_eq!(function, 1);
                let f = Arc::clone(&f);
                let a = match &args[0] {
                    Value::Num(value) => *value,
                    other => panic!("expected first scalar argument, got {other:?}"),
                };
                let b = match &args[1] {
                    Value::Num(value) => *value,
                    other => panic!("expected second scalar argument, got {other:?}"),
                };
                Box::pin(async move {
                    crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(
                        f(a, b),
                    ))
                })
            },
        )))
        .expect("call-port scope");
        drop(entry);
        (fixture, numeric)
    }

    #[test]
    fn fsurf_numeric_vector_reads_typed_integer_storage_exactly() {
        let domain = runmat_value::Tensor::new_integer(
            runmat_value::IntegerStorage::I16(vec![-2, 2]),
            vec![1, 2],
        )
        .expect("typed domain vector");

        assert_eq!(
            numeric_vector(&Value::Tensor(domain)).expect("numeric vector"),
            vec![-2.0, 2.0]
        );

        let wide = runmat_value::Tensor::new_integer(
            runmat_value::IntegerStorage::U64(vec![(1_u64 << 53) + 1, (1_u64 << 53) + 3]),
            vec![1, 2],
        )
        .expect("wide typed domain vector");
        assert!(numeric_vector(&Value::Tensor(wide)).is_err());
    }

    #[test]
    fn fsurf_gates_integer_forms_independently() {
        let function = Value::BoundFunctionHandle(runmat_value::BoundFunctionHandle::from_data(
            runmat_value::BoundFunctionHandleData {
                name: "surface".into(),
                function: 1,
                authority: runmat_value::BoundFunctionAuthority::unavailable(1),
                source_role: runmat_types::FunctionSourceRole::Unknown,
                source_file: None,
            },
        ));
        let domain = Value::Tensor(
            runmat_value::Tensor::new_integer(
                runmat_value::IntegerStorage::I16(vec![-2, 2]),
                vec![1, 2],
            )
            .unwrap(),
        );
        let mesh = Value::Int(IntValue::U8(35));
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);

        let domain_err = preflight_fsurf_extensions(&[function.clone(), domain])
            .expect_err("strict mode rejects integer domain");
        assert_eq!(
            domain_err.identifier(),
            INTEGER_DOMAIN_EXTENSION.error_identifier
        );

        let mesh_err =
            preflight_fsurf_extensions(&[function, Value::String("MeshDensity".into()), mesh])
                .expect_err("strict mode rejects integer mesh density");
        assert_eq!(
            mesh_err.identifier(),
            INTEGER_MESH_DENSITY_EXTENSION.error_identifier
        );

        let callback_err = surface_value_to_scalar(Value::Int(IntValue::I8(1)))
            .expect_err("strict mode rejects integer callback result");
        assert_eq!(
            callback_err.identifier(),
            INTEGER_CALLBACK_OUTPUT_EXTENSION.error_identifier
        );

        let padded_mesh_err = preflight_fsurf_extensions(&[
            Value::BoundFunctionHandle(runmat_value::BoundFunctionHandle::from_data(
                runmat_value::BoundFunctionHandleData {
                    name: "surface".into(),
                    function: 1,
                    authority: runmat_value::BoundFunctionAuthority::unavailable(1),
                    source_role: runmat_types::FunctionSourceRole::Unknown,
                    source_file: None,
                },
            )),
            Value::String(" MeshDensity ".into()),
            Value::Int(IntValue::U8(35)),
        ])
        .expect_err("normalized integer mesh density remains gated");
        assert_eq!(
            padded_mesh_err.identifier(),
            INTEGER_MESH_DENSITY_EXTENSION.error_identifier
        );

        let style_err = preflight_fsurf_extensions(&[
            Value::BoundFunctionHandle(runmat_value::BoundFunctionHandle::from_data(
                runmat_value::BoundFunctionHandleData {
                    name: "surface".into(),
                    function: 1,
                    authority: runmat_value::BoundFunctionAuthority::unavailable(1),
                    source_role: runmat_types::FunctionSourceRole::Unknown,
                    source_file: None,
                },
            )),
            Value::String("FaceAlpha".into()),
            Value::Int(IntValue::U8(1)),
        ])
        .expect_err("integer style remains independently gated");
        assert_eq!(
            style_err.identifier(),
            INTEGER_STYLE_PROPERTY_EXTENSION.error_identifier
        );
    }

    #[test]
    fn fsurf_declares_all_integer_roles_and_extensions() {
        assert_eq!(INTEGER_CAPABILITIES.len(), 5);
        assert_eq!(EXTENSIONS.len(), 8);
        assert!(INTEGER_CAPABILITIES.iter().all(|capability| capability
            .inputs
            .iter()
            .all(|input| input.classes.len() == 8)));
    }

    #[test]
    fn fsurf_all_integer_classes_use_exact_domain_and_callback_boundaries() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        for storage in [
            runmat_value::IntegerStorage::I8(vec![1, 2]),
            runmat_value::IntegerStorage::I16(vec![1, 2]),
            runmat_value::IntegerStorage::I32(vec![1, 2]),
            runmat_value::IntegerStorage::I64(vec![1, 2]),
            runmat_value::IntegerStorage::U8(vec![1, 2]),
            runmat_value::IntegerStorage::U16(vec![1, 2]),
            runmat_value::IntegerStorage::U32(vec![1, 2]),
            runmat_value::IntegerStorage::U64(vec![1, 2]),
        ] {
            let tensor = runmat_value::Tensor::new_integer(storage, vec![1, 2]).unwrap();
            assert_eq!(
                numeric_vector(&Value::Tensor(tensor)).unwrap(),
                vec![1.0, 2.0]
            );
        }
        for value in [
            IntValue::I8(2),
            IntValue::I16(2),
            IntValue::I32(2),
            IntValue::I64(2),
            IntValue::U8(2),
            IntValue::U16(2),
            IntValue::U32(2),
            IntValue::U64((1_u64 << 53) + 2),
        ] {
            let expected = value.to_f64();
            assert_eq!(
                surface_value_to_scalar(Value::Int(value)).unwrap(),
                expected
            );
        }
    }

    #[test]
    fn fsurf_integer_axes_alias_must_be_canonical() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        let _ = clear_figure(None);
        let axes = crate::builtins::plotting::gca::gca_builtin(Vec::new()).unwrap();
        let Value::Num(encoded) = axes else {
            panic!("numeric axes representation");
        };
        let function = Value::BoundFunctionHandle(runmat_value::BoundFunctionHandle::from_data(
            runmat_value::BoundFunctionHandleData {
                name: "surface".into(),
                function: 1,
                authority: runmat_value::BoundFunctionAuthority::unavailable(1),
                source_role: runmat_types::FunctionSourceRole::Unknown,
                source_file: None,
            },
        ));
        {
            let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = preflight_fsurf_extensions(&[
                Value::Int(IntValue::U64(encoded as u64)),
                function.clone(),
            ])
            .expect_err("canonical typed axes alias is gated");
            assert_eq!(error.identifier(), INTEGER_AXES_EXTENSION.error_identifier);
        }
        {
            let _enabled = crate::compatibility::push_runmat_extensions_enabled(true);
            let wrapped = (encoded as u64) + (1_u64 << 52);
            let error = preflight_fsurf_extensions(&[Value::Int(IntValue::U64(wrapped)), function])
                .expect_err("wrapped axes aliases are noncanonical");
            assert_eq!(error.identifier(), FSURF_ERROR_INVALID_ARGUMENT.identifier);
        }
    }

    #[test]
    fn fsurf_resident_integer_callback_is_gated_before_provider_access() {
        let fixture = selected_functions(&[1]);
        let _entry = fixture.runtime.enter();
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |function, _args, _requested_outputs| {
                assert_eq!(function, 1);
                Box::pin(async {
                    let handle = runmat_accelerate_api::test_support::TestAllocationOwner::new()
                        .allocate_zeroed(
                            vec![1, 1],
                            runmat_accelerate_api::NumericElementType::U64,
                            runmat_accelerate_api::GpuTensorStorage::Real,
                        )
                        .unwrap()
                        .handle;
                    crate::user_functions::test_support::rooted_single_value_sequence(
                        Value::GpuTensor(handle),
                    )
                })
            },
        )))
        .expect("call-port scope");
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let work = super::super::coordinate_admission::CoordinateWork::admit(
            Some(&fixture.runtime),
            "fsurf",
        )
        .unwrap();
        let error = block_on(fixture.runtime.scope(call_surface_function(
            Some(&fixture.runtime),
            &work,
            &fixture.value("surface", 1),
            0.0,
            0.0,
        )))
        .expect_err("resident integer callback is rejected before gather");
        assert_eq!(
            error.identifier(),
            INTEGER_CALLBACK_OUTPUT_EXTENSION.error_identifier
        );
    }

    #[test]
    fn fsurf_logical_and_single_callback_samples_are_gated() {
        let single = runmat_value::Tensor::from_numeric_storage(
            runmat_value::NumericStorage::F32(vec![1.0]),
            vec![1, 1],
        )
        .unwrap();
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let logical_error = surface_value_to_scalar(Value::Bool(true))
            .expect_err("logical callback sample is gated");
        assert_eq!(
            logical_error.identifier(),
            LOGICAL_NUMERIC_INPUT_EXTENSION.error_identifier
        );
        let single_error = surface_value_to_scalar(Value::Tensor(single))
            .expect_err("single callback sample is gated");
        assert_eq!(
            single_error.identifier(),
            SINGLE_NUMERIC_INPUT_EXTENSION.error_identifier
        );
    }

    #[test]
    fn fsurf_style_preflight_is_property_specific() {
        let _enabled = crate::compatibility::push_runmat_extensions_enabled(true);
        preflight_surface_style_extension("FaceAlpha", &Value::Int(IntValue::U8(1)))
            .expect("integer alpha extension");
        preflight_surface_style_extension("Visible", &Value::Int(IntValue::U64(u64::MAX)))
            .expect("boolean-style integers do not cross binary64");
        preflight_surface_style_extension("DisplayName", &Value::Int(IntValue::U8(1)))
            .expect("unsupported text value is not misclassified as an integer style extension");

        let defaults = SurfaceStyleDefaults::new(
            ColorMap::Parula,
            ShadingMode::Smooth,
            false,
            1.0,
            false,
            true,
        );
        let style = parse_surface_style_args(
            BUILTIN_NAME,
            crate::call::FunctionArgumentView::from_slice(&[
                Value::String("FaceAlpha".into()),
                Value::Int(IntValue::U8(1)),
                Value::String("Visible".into()),
                Value::Int(IntValue::U64(u64::MAX)),
            ]),
            defaults,
        )
        .expect("enabled supported integer styles");
        assert_eq!(style.alpha, 1.0);
        assert_eq!(style.visible, Some(true));
    }

    #[test]
    fn fsurf_samples_explicit_function_surface() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        let _ = clear_figure(None);
        let (fixture, _numeric) = with_test_function(|x, y| x * x + y);
        let _entry = fixture.runtime.enter();
        let prior_roots = fixture.runtime.plot_callback_root_count();
        let prior_direct_roots = super::super::state::callbacks::direct_root_count();

        let handle = block_on(fixture.runtime.scope(fsurf_builtin(vec![
            fixture.value("surface", 1),
            Value::Tensor(runmat_value::Tensor::new(vec![0.0, 1.0, 0.0, 2.0], vec![1, 4]).unwrap()),
            Value::String("MeshDensity".into()),
            Value::Num(3.0),
            Value::String("DisplayName".into()),
            Value::String("parabola".into()),
        ])))
        .unwrap();

        let ty = get_builtin(vec![Value::Num(handle), Value::String("Type".into())]).unwrap();
        assert_eq!(ty, Value::String("functionsurface".into()));
        let callback = get_builtin(vec![Value::Num(handle), Value::String("Function".into())])
            .expect("owning session callback");
        assert_eq!(callback, fixture.value("surface", 1));
        assert_eq!(
            super::super::state::callbacks::direct_root_count(),
            prior_direct_roots
        );
        assert_eq!(fixture.runtime.plot_callback_root_count(), prior_roots + 1);
        let mesh_density = get_builtin(vec![
            Value::Num(handle),
            Value::String("MeshDensity".into()),
        ])
        .unwrap();
        assert_eq!(mesh_density, Value::Num(3.0));
        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Surface(surface) = fig.plots().next().unwrap() else {
            panic!("expected surface plot");
        };
        assert_eq!(surface.x_data.len(), 3);
        assert_eq!(surface.y_data.len(), 3);
        assert_eq!(surface.label.as_deref(), Some("parabola"));
        let z = surface.z_data.as_ref().unwrap();
        assert!((z[2][2] - 3.0).abs() < 1.0e-12);
        clear_figure(Some(current_figure_handle())).expect("clear callback scene");
        assert_eq!(fixture.runtime.plot_callback_root_count(), prior_roots);
    }

    #[test]
    fn sibling_session_clear_revokes_scene_callback_without_owning_its_value() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        let _ = clear_figure(None);
        let (fixture, _numeric) = with_test_function(|x, y| x + y);
        let first = fixture.runtime.clone();
        let second = crate::context::RuntimeContext::new(std::rc::Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ));
        let handle = block_on(first.scope(fsurf_builtin(vec![
            fixture.value("surface", 1),
            Value::Tensor(runmat_value::Tensor::new(vec![0.0, 1.0, 0.0, 1.0], vec![1, 4]).unwrap()),
            Value::String("MeshDensity".into()),
            Value::Num(3.0),
        ])))
        .expect("first session surface");
        assert_eq!(first.plot_callback_root_count(), 1);
        {
            let _active = second.enter();
            let error = get_builtin(vec![Value::Num(handle), Value::String("Function".into())])
                .expect_err("sibling cannot observe original callable");
            assert!(error.message().contains("another or ended session"));
            clear_figure(Some(current_figure_handle())).expect("sibling clears shared scene");
        }
        assert_eq!(first.plot_callback_root_count(), 0);
    }

    #[test]
    fn fsurf_supports_axes_target_and_parametric_surface() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        let _ = clear_figure(None);
        let ax = crate::builtins::plotting::subplot::subplot_builtin(
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(2.0),
        )
        .unwrap();
        let fixture = selected_functions(&[11, 12, 13]);
        let _entry = fixture.runtime.enter();
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |function, args, _requested_outputs| {
                let u = match &args[0] {
                    Value::Num(value) => *value,
                    other => panic!("expected first scalar argument, got {other:?}"),
                };
                let v = match &args[1] {
                    Value::Num(value) => *value,
                    other => panic!("expected second scalar argument, got {other:?}"),
                };
                Box::pin(async move {
                    crate::user_functions::test_support::rooted_single_value_sequence(
                        match function {
                            11 => Value::Num(u),
                            12 => Value::Num(v),
                            13 => Value::Num(u + v),
                            other => panic!("unexpected function id {other}"),
                        },
                    )
                })
            },
        )))
        .expect("call-port scope");

        let handle = block_on(fixture.runtime.scope(fsurf_builtin(vec![
            Value::Num(ax),
            fixture.value("x", 11),
            fixture.value("y", 12),
            fixture.value("z", 13),
            Value::Tensor(
                runmat_value::Tensor::new(vec![-1.0, 1.0, 2.0, 4.0], vec![1, 4]).unwrap(),
            ),
            Value::String("MeshDensity".into()),
            Value::Num(2.0),
        ])))
        .unwrap();

        let ty = get_builtin(vec![Value::Num(handle), Value::String("Type".into())]).unwrap();
        assert_eq!(ty, Value::String("functionsurface".into()));
        let fig = clone_figure(current_figure_handle()).unwrap();
        assert_eq!(fig.plot_axes_indices()[0], 1);
        let PlotElement::Surface(surface) = fig.plots().next().unwrap() else {
            panic!("expected surface plot");
        };
        assert!(surface.x_grid.is_some());
        assert!(surface.y_grid.is_some());
        let z = surface.z_data.as_ref().unwrap();
        assert!((z[0][0] - 1.0).abs() < 1.0e-12);
        assert!((z[1][1] - 5.0).abs() < 1.0e-12);
    }

    #[test]
    fn fsurf_allows_nonfinite_explicit_samples() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        let _ = clear_figure(None);
        let (fixture, _numeric) = with_test_function(|x, y| {
            if x == 0.0 && y == 0.0 {
                f64::NAN
            } else {
                x + y
            }
        });
        let _entry = fixture.runtime.enter();

        let handle = block_on(fixture.runtime.scope(fsurf_builtin(vec![
            fixture.value("surface", 1),
            Value::Tensor(runmat_value::Tensor::new(vec![0.0, 1.0, 0.0, 1.0], vec![1, 4]).unwrap()),
            Value::String("MeshDensity".into()),
            Value::Num(2.0),
        ])))
        .unwrap();

        let ty = get_builtin(vec![Value::Num(handle), Value::String("Type".into())]).unwrap();
        assert_eq!(ty, Value::String("functionsurface".into()));
        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Surface(surface) = fig.plots().next().unwrap() else {
            panic!("expected surface plot");
        };
        let z = surface.z_data.as_ref().unwrap();
        assert!(z[0][0].is_nan());
        assert_eq!(z[1][1], 2.0);
    }

    #[test]
    fn fsurf_rejects_invalid_mesh_density() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        let (fixture, _numeric) = with_test_function(|x, y| x + y);
        let _entry = fixture.runtime.enter();
        let err = block_on(fixture.runtime.scope(fsurf_builtin(vec![
            Value::BoundFunctionHandle(runmat_value::BoundFunctionHandle::from_data(
                runmat_value::BoundFunctionHandleData {
                    name: "surface".into(),
                    function: 1,
                    authority: runmat_value::BoundFunctionAuthority::unavailable(1),
                    source_role: runmat_types::FunctionSourceRole::Unknown,
                    source_file: None,
                },
            )),
            Value::String("MeshDensity".into()),
            Value::Num(1.0),
        ])))
        .expect_err("expected mesh-density validation error");
        assert_eq!(err.identifier(), FSURF_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn fsurf_mesh_density_reads_typed_integer_tensor_exactly() {
        let exact = runmat_value::Tensor::new_integer(
            runmat_value::IntegerStorage::U64(vec![400]),
            vec![1, 1],
        )
        .expect("typed density");
        assert_eq!(parse_mesh_density(&Value::Tensor(exact)).unwrap(), 400);

        let too_large = runmat_value::Tensor::new_integer(
            runmat_value::IntegerStorage::U64(vec![401]),
            vec![1, 1],
        )
        .expect("large density");
        assert!(parse_mesh_density(&Value::Tensor(too_large)).is_err());

        let negative = runmat_value::Tensor::new_integer(
            runmat_value::IntegerStorage::I64(vec![-1]),
            vec![1, 1],
        )
        .expect("negative density");
        assert!(parse_mesh_density(&Value::Tensor(negative)).is_err());

        let boundary = if usize::BITS == 64 {
            usize::MAX as f64
        } else {
            (usize::MAX as f64) + 1.0
        };
        assert!(parse_mesh_density(&Value::Num(boundary)).is_err());
    }

    #[test]
    fn fsurf_function_scalar_reads_typed_integer_storage_exactly() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor = runmat_value::Tensor::new_integer(
            runmat_value::IntegerStorage::I16(vec![12]),
            vec![1, 1],
        )
        .unwrap();

        assert_eq!(
            surface_value_to_scalar(Value::Tensor(tensor)).unwrap(),
            12.0
        );

        let wide = runmat_value::Tensor::new_integer(
            runmat_value::IntegerStorage::U64(vec![(1_u64 << 53) + 1]),
            vec![1, 1],
        )
        .unwrap();
        assert!(surface_value_to_scalar(Value::Tensor(wide)).is_err());
    }

    #[test]
    fn fsurf_descriptor_covers_parametric_and_axes_forms() {
        let labels: Vec<&str> = FSURF_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"h = fsurf(f)"));
        assert!(labels.contains(&"h = fsurf(fx, fy, fz)"));
        assert!(labels.contains(&"h = fsurf(ax, ___)"));
    }
}
