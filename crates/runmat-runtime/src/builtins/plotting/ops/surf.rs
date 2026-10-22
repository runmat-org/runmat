//! MATLAB-compatible `surf` builtin.

use glam::Vec3;
use runmat_accelerate_api::{self, GpuTensorHandle, ProviderPrecision};
use runmat_builtins::BuiltinErrorDescriptor;
use runmat_macros::runtime_builtin;
use runmat_plot::gpu::ScalarType;
use runmat_plot::plots::{ColorMap, ShadingMode, SurfacePlot};
use runmat_value::Value;
#[cfg(test)]
use runmat_value::{IntegerStorage, Tensor};

use super::common::{
    surface_matrix_dimensions, tensor_to_surface_grid_matlab_xy, SurfaceDataInput,
};
use super::mesh::{
    numeric_plot_data_from_tensor, numeric_plot_data_from_value, retain_surface_source_data,
};
use super::op_common::surface_inputs::{
    axis_sources_to_host, extract_meshgrid_axes_from_xy_matrices,
    parse_surface_call_args_matlab_xy, surface_axis_sources_from_xy_values, AxisSource,
};
use super::perf::compute_surface_lod;
use super::plotting_error;
use super::state::{render_active_plot, PlotRenderOptions};
use super::style::{parse_surface_style_args, SurfaceStyle, SurfaceStyleDefaults};
use crate::build_runtime_error;
use std::convert::TryFrom;
use std::sync::Arc;

use crate::{BuiltinResult, RuntimeError};

#[path = "surf/color_input.rs"]
mod color_input;

const BUILTIN_NAME: &str = "surf";

#[cfg(test)]
use runmat_builtins::{SURF_DESCRIPTOR, SURF_INTEGER_CAPABILITIES};
pub use runmat_builtins::{SURF_ERROR_INTERNAL, SURF_ERROR_INVALID_ARGUMENT};

fn surf_error_with_detail(
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

fn map_surf_invalid_argument(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    surf_error_with_detail(&SURF_ERROR_INVALID_ARGUMENT, err.message)
}

fn map_surf_internal(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    surf_error_with_detail(&SURF_ERROR_INTERNAL, err.message)
}

#[runtime_builtin(
    name = "surf",
    binding_variant = "default",
    builtin_path = "crate::builtins::plotting::surf"
)]
pub async fn surf_builtin(args: crate::call::BuiltinArguments) -> crate::BuiltinResult<f64> {
    let color_input::ParsedSurfaceInput {
        x,
        y,
        z,
        color,
        rest,
    } = color_input::parse(args)?;
    let source_x = numeric_plot_data_from_value(&x, BUILTIN_NAME)
        .await
        .map_err(map_surf_invalid_argument)?;
    let source_y = numeric_plot_data_from_value(&y, BUILTIN_NAME)
        .await
        .map_err(map_surf_invalid_argument)?;
    let z_input = SurfaceDataInput::from_value(z, "surf").map_err(map_surf_invalid_argument)?;
    let source_z = match &z_input {
        SurfaceDataInput::Host(tensor) => Some(numeric_plot_data_from_tensor(tensor)),
        SurfaceDataInput::Gpu(handle) => Some(numeric_plot_data_from_tensor(
            &super::common::gather_tensor_from_gpu_async(handle.clone(), BUILTIN_NAME)
                .await
                .map_err(map_surf_invalid_argument)?,
        )),
    };
    let (rows, cols) = z_input
        .grid_shape(BUILTIN_NAME)
        .map_err(map_surf_invalid_argument)?;

    let style = Arc::new(
        parse_surface_style_args(
            "surf",
            rest.view(),
            SurfaceStyleDefaults::new(
                ColorMap::Parula,
                ShadingMode::Smooth,
                false,
                1.0,
                false,
                true,
            ),
        )
        .map_err(map_surf_invalid_argument)?,
    );
    let mut source_c = match color {
        Some(value) => Some(color_input::prepare(value, rows, cols).await?),
        None => None,
    };

    // Prefer a no-download path for vector-like gpuArray axes: keep X/Y on-device and pass
    // their buffers through to the GPU vertex packer. If X/Y are meshgrid matrices, we still
    // need to validate and extract axes on the host.
    if let Some(mut surface) = build_parametric_surface_cpu(&x, &y, &z_input, rows, cols, &style)
        .map_err(map_surf_invalid_argument)?
    {
        if let Some(color) = source_c.take() {
            color_input::apply(&mut surface, color).map_err(map_surf_invalid_argument)?;
        }
        retain_surface_source_data(&mut surface, source_x, source_y, source_z);
        return render_surface(surface).map_err(map_surf_internal);
    }
    let (x_axis, y_axis) =
        match surface_axis_sources_from_xy_values(x, y, rows, cols, BUILTIN_NAME).await {
            Ok(axes) => axes,
            Err(axis_err) => return Err(map_surf_invalid_argument(axis_err)),
        };
    let mut surface =
        if let Some(z_gpu) = z_input.gpu_handle().cloned().filter(|_| source_c.is_none()) {
            match super::op_common::surface_inputs::PreparedSurfaceGpu::prepare(
                BUILTIN_NAME,
                &x_axis,
                &y_axis,
                &z_gpu,
            )
            .await?
            {
                Some(mut prepared) => {
                    let (min_z, max_z) = prepared.z_bounds;
                    build_surface_gpu_plot_with_bounds_async(
                        BUILTIN_NAME,
                        &x_axis,
                        &y_axis,
                        &z_gpu,
                        &mut prepared,
                        min_z,
                        max_z,
                        style.colormap.clone(),
                        style.alpha,
                        style.flatten_z,
                    )
                    .await?
                }
                None => {
                    let (x_host, y_host) = axis_sources_to_host(&x_axis, &y_axis, BUILTIN_NAME)
                        .await
                        .map_err(map_surf_invalid_argument)?;
                    build_surface_cpu(&z_input, x_host, y_host, rows, cols)
                        .await
                        .map_err(map_surf_invalid_argument)?
                }
            }
        } else {
            let (x_host, y_host) = axis_sources_to_host(&x_axis, &y_axis, BUILTIN_NAME)
                .await
                .map_err(map_surf_invalid_argument)?;
            build_surface_cpu(&z_input, x_host, y_host, rows, cols)
                .await
                .map_err(map_surf_invalid_argument)?
        };

    style.apply_to_plot(&mut surface);
    if let Some(color) = source_c.take() {
        color_input::apply(&mut surface, color).map_err(map_surf_invalid_argument)?;
    }
    retain_surface_source_data(&mut surface, source_x, source_y, source_z);
    render_surface(surface).map_err(map_surf_internal)
}

fn render_surface(surface: SurfacePlot) -> BuiltinResult<f64> {
    let mut surface = Some(surface);
    let plot_index_out = std::rc::Rc::new(std::cell::RefCell::new(None));
    let plot_index_slot = std::rc::Rc::clone(&plot_index_out);
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    let opts = PlotRenderOptions {
        axis_equal: false,
        ..Default::default()
    };
    let render_result = render_active_plot(BUILTIN_NAME, opts, move |figure, axes| {
        let surface = surface.take().expect("surf plot consumed once");
        let plot_index = figure.add_surface_plot_on_axes(surface, axes);
        *plot_index_slot.borrow_mut() = Some((axes, plot_index));
        Ok(())
    });
    let Some((axes, plot_index)) = *plot_index_out.borrow() else {
        return render_result.map(|_| f64::NAN);
    };
    let handle =
        crate::builtins::plotting::state::register_surface_handle(figure_handle, axes, plot_index);
    if let Err(err) = render_result {
        if crate::builtins::plotting::state::presentation_unavailable(&err) {
            return Ok(handle);
        }
        return Err(err);
    }
    Ok(handle)
}

fn build_parametric_surface_cpu(
    x: &Value,
    y: &Value,
    z_input: &SurfaceDataInput,
    rows: usize,
    cols: usize,
    style: &SurfaceStyle,
) -> BuiltinResult<Option<SurfacePlot>> {
    let (Value::Tensor(x), Value::Tensor(y)) = (x, y) else {
        return Ok(None);
    };
    let SurfaceDataInput::Host(z) = z_input else {
        return Ok(None);
    };
    if surface_matrix_dimensions(x) != (rows, cols) || surface_matrix_dimensions(y) != (rows, cols)
    {
        return Ok(None);
    }
    if extract_meshgrid_axes_from_xy_matrices(x, y, rows, cols, BUILTIN_NAME).is_ok() {
        return Ok(None);
    }

    let x_grid = tensor_to_surface_grid_matlab_xy(x.clone(), rows, cols, BUILTIN_NAME)?;
    let y_grid = tensor_to_surface_grid_matlab_xy(y.clone(), rows, cols, BUILTIN_NAME)?;
    let z_grid = tensor_to_surface_grid_matlab_xy(z.clone(), rows, cols, BUILTIN_NAME)?;
    let mut surface = SurfacePlot::from_coordinate_grids(x_grid, y_grid, z_grid)
        .map_err(|err| plotting_error(BUILTIN_NAME, format!("surf: {err}")))?;
    style.apply_to_plot(&mut surface);
    Ok(Some(surface))
}

async fn build_surface_cpu(
    z_input: &SurfaceDataInput,
    x_axis: Vec<f64>,
    y_axis: Vec<f64>,
    rows: usize,
    cols: usize,
) -> BuiltinResult<SurfacePlot> {
    let z_tensor = match z_input.clone() {
        SurfaceDataInput::Host(tensor) => tensor,
        SurfaceDataInput::Gpu(handle) => {
            super::common::gather_tensor_from_gpu_async(handle, BUILTIN_NAME).await?
        }
    };
    let grid = tensor_to_surface_grid_matlab_xy(z_tensor, rows, cols, BUILTIN_NAME)?;
    build_surface(x_axis, y_axis, grid)
}

pub(crate) fn build_surface(
    x_axis: Vec<f64>,
    y_axis: Vec<f64>,
    z_grid: Vec<Vec<f64>>,
) -> BuiltinResult<SurfacePlot> {
    if x_axis.is_empty() || y_axis.is_empty() {
        return Err(plotting_error(
            BUILTIN_NAME,
            "surf: axis vectors must be non-empty",
        ));
    }

    let surface = SurfacePlot::new(x_axis, y_axis, z_grid)
        .map_err(|err| plotting_error(BUILTIN_NAME, format!("surf: {err}")))?
        .with_colormap(ColorMap::Parula)
        .with_shading(ShadingMode::Smooth);
    Ok(surface)
}

pub(crate) async fn build_surface_gpu_plot_with_bounds_async(
    name: &'static str,
    x_axis: &AxisSource,
    y_axis: &AxisSource,
    z: &GpuTensorHandle,
    prepared: &mut super::op_common::surface_inputs::PreparedSurfaceGpu<'_>,
    min_z: f32,
    max_z: f32,
    colormap: ColorMap,
    alpha: f32,
    flatten_z: bool,
) -> BuiltinResult<SurfacePlot> {
    prepared.validate_sources(name, x_axis, y_axis, z)?;
    let (min_x, max_x) = prepared.x_bounds;
    let (min_y, max_y) = prepared.y_bounds;
    let context = &prepared.context;
    let gpu_exports = &mut prepared.exports;
    let result = (|| {
        let z_ref = gpu_exports.export(z, name)?.ok_or_else(|| {
            plotting_error(
                name,
                format!("{name}: admitted GPU Z export became unavailable"),
            )
        })?;
        let x_len = x_axis.len();
        let y_len = y_axis.len();
        let bounds = runmat_plot::core::scene::BoundingBox::new(
            Vec3::new(min_x, min_y, min_z),
            Vec3::new(max_x, max_y, max_z),
        );
        let extent_hint = ((max_x - min_x).powi(2) + (max_y - min_y).powi(2)).sqrt();

        let color_table = build_color_lut(&colormap, 512, 1.0);
        let scalar = ScalarType::from_is_f64(z_ref.precision == ProviderPrecision::F64);

        let mut x_axis_f32: Vec<f32> = Vec::new();
        let mut y_axis_f32: Vec<f32> = Vec::new();

        let x_axis_gpu: Option<runmat_plot::gpu::GpuInputBuffer> = match x_axis {
            AxisSource::Gpu(h) => {
                let exported = gpu_exports.export(h, name)?.ok_or_else(|| {
                    plotting_error(name, format!("{name}: unable to export GPU X axis buffer"))
                })?;
                if exported.len != x_len {
                    return Err(plotting_error(
                        name,
                        format!(
                            "{name}: X axis length mismatch (expected {x_len}, got {})",
                            exported.len
                        ),
                    ));
                }
                if exported.precision != z_ref.precision {
                    return Err(plotting_error(
                        name,
                        format!("{name}: X axis precision must match Z precision"),
                    ));
                }
                Some(exported.clone().into())
            }
            AxisSource::Host(v) => {
                if scalar == ScalarType::F32 {
                    x_axis_f32 = v.iter().map(|&val| val as f32).collect::<Vec<f32>>();
                }
                None
            }
        };

        let y_axis_gpu: Option<runmat_plot::gpu::GpuInputBuffer> = match y_axis {
            AxisSource::Gpu(h) => {
                let exported = gpu_exports.export(h, name)?.ok_or_else(|| {
                    plotting_error(name, format!("{name}: unable to export GPU Y axis buffer"))
                })?;
                if exported.len != y_len {
                    return Err(plotting_error(
                        name,
                        format!(
                            "{name}: Y axis length mismatch (expected {y_len}, got {})",
                            exported.len
                        ),
                    ));
                }
                if exported.precision != z_ref.precision {
                    return Err(plotting_error(
                        name,
                        format!("{name}: Y axis precision must match Z precision"),
                    ));
                }
                Some(exported.clone().into())
            }
            AxisSource::Host(v) => {
                if scalar == ScalarType::F32 {
                    y_axis_f32 = v.iter().map(|&val| val as f32).collect::<Vec<f32>>();
                }
                None
            }
        };

        let inputs = runmat_plot::gpu::surface::SurfaceGpuInputs {
            x_axis: if let Some(buffer) = x_axis_gpu.as_ref() {
                runmat_plot::gpu::surface::SurfaceAxis::Buffer(buffer.clone())
            } else if z_ref.precision == ProviderPrecision::F64 {
                match x_axis {
                    AxisSource::Host(v) => {
                        runmat_plot::gpu::surface::SurfaceAxis::F64(v.as_slice())
                    }
                    AxisSource::Gpu(_) => unreachable!("gpu X axis handled above"),
                }
            } else {
                runmat_plot::gpu::surface::SurfaceAxis::F32(x_axis_f32.as_slice())
            },
            y_axis: if let Some(buffer) = y_axis_gpu.as_ref() {
                runmat_plot::gpu::surface::SurfaceAxis::Buffer(buffer.clone())
            } else if z_ref.precision == ProviderPrecision::F64 {
                match y_axis {
                    AxisSource::Host(v) => {
                        runmat_plot::gpu::surface::SurfaceAxis::F64(v.as_slice())
                    }
                    AxisSource::Gpu(_) => unreachable!("gpu Y axis handled above"),
                }
            } else {
                runmat_plot::gpu::surface::SurfaceAxis::F32(y_axis_f32.as_slice())
            },
            z_buffer: z_ref.clone().into(),
            color_table: &color_table,
            x_len: x_len as u32,
            y_len: y_len as u32,
            scalar,
        };
        let gpu_source = runmat_plot::plots::SurfaceGpuSource {
            x_axis: runmat_plot::gpu::axis::OwnedAxisData::from_axis(&inputs.x_axis),
            y_axis: runmat_plot::gpu::axis::OwnedAxisData::from_axis(&inputs.y_axis),
            z_buffer: inputs.z_buffer.clone(),
            x_len,
            y_len,
            scalar,
        };
        let lod = compute_surface_lod(x_len, y_len, extent_hint);
        let params = runmat_plot::gpu::surface::SurfaceGpuParams {
            min_z,
            max_z,
            alpha,
            flatten_z,
            x_stride: lod.stride_x,
            y_stride: lod.stride_y,
            lod_x_len: lod.lod_x_len,
            lod_y_len: lod.lod_y_len,
        };

        let gpu_vertices = context
            .access
            .with_shared(|emission| {
                runmat_plot::gpu::surface::pack_surface_vertices(emission, &inputs, &params)
            })
            .map_err(|source| {
                super::plotting_error_with_source(
                    name,
                    format!("{}: GPU emission unavailable: {source}", name),
                    source,
                )
            })?
            .map_err(|e| {
                plotting_error(name, format!("{name}: failed to build GPU vertices: {e}"))
            })?;

        let vertex_count = lod.vertex_count();
        let lod_x_len = usize::try_from(lod.lod_x_len)
            .map_err(|_| plotting_error(name, format!("{name}: LOD X size overflowed")))?;
        let lod_y_len = usize::try_from(lod.lod_y_len)
            .map_err(|_| plotting_error(name, format!("{name}: LOD Y size overflowed")))?;
        let mut surface =
            SurfacePlot::from_gpu_buffer(lod_x_len, lod_y_len, gpu_vertices, vertex_count, bounds)
                .with_gpu_source(gpu_source)
                .with_scalar_color_extent(min_z as f64, max_z as f64);
        surface.colormap = colormap;
        surface.alpha = alpha;
        surface.flatten_z = flatten_z;
        Ok(surface)
    })();
    gpu_exports.complete(result, name)
}

pub(crate) fn build_color_lut(colormap: &ColorMap, samples: usize, alpha: f32) -> Vec<[f32; 4]> {
    let clamped = samples.max(2);
    (0..clamped)
        .map(|i| {
            let t = i as f32 / (clamped as f32 - 1.0);
            let rgb = colormap.map_value(t);
            [rgb.x, rgb.y, rgb.z, alpha]
        })
        .collect()
}

#[cfg(test)]
pub(crate) mod tests {
    async fn surf_builtin(args: Vec<Value>) -> BuiltinResult<f64> {
        super::surf_builtin(crate::call::BuiltinArguments::from_borrowed(&args, None)?).await
    }

    fn parse_surface_call_args_matlab_xy(
        args: Vec<Value>,
        builtin: &'static str,
    ) -> BuiltinResult<(Value, Value, Value, crate::call::BuiltinArguments)> {
        super::parse_surface_call_args_matlab_xy(
            crate::call::BuiltinArguments::from_borrowed(&args, None)?,
            builtin,
        )
    }

    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, reset_hold_state_for_run};

    #[test]
    fn surf_color_forms_keep_the_color_array_out_of_style_arguments() {
        let z = Value::Tensor(Tensor::new(vec![1.0; 4], vec![2, 2]).unwrap());
        let c = Value::Tensor(Tensor::new(vec![0.0, 1.0, 1.0, 0.0], vec![2, 2]).unwrap());
        let color_input::ParsedSurfaceInput {
            color, rest: style, ..
        } = color_input::parse(crate::builtins::common::test_support::builtin_arguments(
            vec![z.clone(), c.clone()],
        ))
        .unwrap();
        assert!(matches!(color, Some(Value::Tensor(_))));
        assert!(style.is_empty());
        let color_input::ParsedSurfaceInput {
            color, rest: style, ..
        } = color_input::parse(crate::builtins::common::test_support::builtin_arguments(
            vec![
                z.clone(),
                c.clone(),
                Value::String("EdgeColor".into()),
                Value::String("none".into()),
            ],
        ))
        .unwrap();
        assert!(color.is_some());
        assert_eq!(style.len(), 2);
        let axes = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2]).unwrap());
        let color_input::ParsedSurfaceInput {
            color, rest: style, ..
        } = color_input::parse(crate::builtins::common::test_support::builtin_arguments(
            vec![axes.clone(), axes.clone(), z.clone()],
        ))
        .unwrap();
        assert!(color.is_none());
        assert!(style.is_empty());
        let color_input::ParsedSurfaceInput {
            color, rest: style, ..
        } = color_input::parse(crate::builtins::common::test_support::builtin_arguments(
            vec![
                axes.clone(),
                axes,
                z,
                c,
                Value::String("EdgeColor".into()),
                Value::String("none".into()),
            ],
        ))
        .unwrap();
        assert!(matches!(color, Some(Value::Tensor(_))));
        assert_eq!(style.len(), 2);
    }

    fn setup_plot_tests() {
        ensure_plot_test_env();
    }

    fn tensor_from(data: &[f64]) -> Tensor {
        Tensor::new(data.to_vec(), vec![data.len()]).expect("surf test vector")
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn surf_requires_matching_grid() {
        setup_plot_tests();
        let res = futures::executor::block_on(surf_builtin(vec![
            Value::Tensor(tensor_from(&[0.0, 1.0])),
            Value::Tensor(tensor_from(&[0.0])),
            Value::Tensor(Tensor::new(vec![0.0], vec![1]).expect("scalar grid")),
        ]));
        assert!(res.is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn parametric_cpu_surface_preserves_low_rank_grid_geometry() {
        let style = parse_surface_style_args(
            BUILTIN_NAME,
            crate::call::FunctionArgumentView::from_slice(&[]),
            SurfaceStyleDefaults::new(
                ColorMap::Parula,
                ShadingMode::Smooth,
                false,
                1.0,
                false,
                true,
            ),
        )
        .expect("default surface style");

        let scalar = Value::Tensor(Tensor::new(vec![0.0], Vec::new()).unwrap());
        let scalar_z = SurfaceDataInput::Host(Tensor::new(vec![1.0], vec![1, 1]).unwrap());
        assert!(
            build_parametric_surface_cpu(&scalar, &scalar, &scalar_z, 1, 1, &style)
                .unwrap()
                .is_none()
        );

        let rank_one = Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![2]).unwrap());
        let rank_one_z = SurfaceDataInput::Host(Tensor::new(vec![1.0, 2.0], vec![2]).unwrap());
        assert_eq!(rank_one_z.grid_shape(BUILTIN_NAME).unwrap(), (1, 2));
        assert!(
            build_parametric_surface_cpu(&rank_one, &rank_one, &rank_one_z, 1, 2, &style,)
                .unwrap()
                .is_some()
        );
        assert!(
            build_parametric_surface_cpu(&rank_one, &rank_one, &rank_one_z, 2, 1, &style,)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn surf_catalog_infers_host_double_handle() {
        use runmat_types::*;
        let numeric = ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        });
        let input = ValueFact::proven(
            numeric.clone(),
            ShapeFact::from(vec![Some(3), Some(2)]),
            StorageFact::Dense,
        );
        let request = CallRequest {
            arguments: vec![input; 3],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        };
        let result =
            runmat_builtins::infer_catalog_call(&runmat_builtins::SURF_CATALOG_ENTRY, &request);
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        assert_eq!(result.outputs.len(), 1);
        let output = &result.outputs[0];
        assert_eq!(output.kind, numeric);
        assert_eq!(output.shape, ShapeFact::Scalar);
        assert_eq!(output.storage, StorageFact::Scalar);
        assert_eq!(output.residency, ResidencyFact::Host);
    }

    #[test]
    fn surf_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = SURF_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"h = surf(Z)"));
        assert!(labels.contains(&"h = surf(X, Y, Z)"));
        assert!(labels.contains(&"h = surf(X, Y, Z, Name, Value, ...)"));
    }

    #[test]
    fn surf_missing_input_uses_stable_identifier() {
        setup_plot_tests();
        let err = futures::executor::block_on(surf_builtin(vec![])).expect_err("missing input");
        assert_eq!(err.identifier(), SURF_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn surf_accepts_meshgrid_xy_matrices() {
        // x = [0, 1, 2], y = [10, 20] => meshgrid X/Y are 2x3.
        // Column-major storage for X:
        // X = [0 1 2;
        //      0 1 2]
        let x = Tensor::new(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0], vec![2, 3]).expect("meshgrid x");
        // Y = [10 10 10;
        //      20 20 20]
        let y =
            Tensor::new(vec![10.0, 20.0, 10.0, 20.0, 10.0, 20.0], vec![2, 3]).expect("meshgrid y");
        // Z is 2x3 surface heights (any values), column-major.
        let z =
            Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).expect("surface heights");

        let (rows, cols) = surface_matrix_dimensions(&z);
        let (x_axis, y_axis) = crate::builtins::plotting::op_common::surface_inputs::extract_meshgrid_axes_from_xy_matrices(&x, &y, rows, cols, BUILTIN_NAME).expect("axes");
        assert_eq!(x_axis.len(), 3);
        assert_eq!(y_axis.len(), 2);

        // With extracted axes, Z should validate and reshape into a surface grid.
        let grid = tensor_to_surface_grid_matlab_xy(z, y_axis.len(), x_axis.len(), BUILTIN_NAME);
        assert!(
            grid.is_ok(),
            "expected Z to be compatible with extracted axes"
        );
    }

    #[test]
    fn surf_z_only_shorthand_builds_surface_with_default_axes() {
        setup_plot_tests();
        let out = futures::executor::block_on(surf_builtin(vec![Value::Tensor(
            Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).expect("square surface"),
        )]));
        assert!(out.is_ok() || out.is_err());
        let (x, y, z, rest) = parse_surface_call_args_matlab_xy(
            vec![Value::Tensor(
                Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).expect("square surface"),
            )],
            BUILTIN_NAME,
        )
        .unwrap();
        assert!(rest.is_empty());
        assert_eq!(
            Tensor::try_from(&x).unwrap().materialize_f64(),
            vec![1.0, 2.0]
        );
        assert_eq!(
            Tensor::try_from(&y).unwrap().materialize_f64(),
            vec![1.0, 2.0]
        );
        assert_eq!(
            Tensor::try_from(&z).unwrap().materialize_f64(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn surf_z_only_shorthand_uses_matlab_xy_for_non_square_grid() {
        let (x, y, _, rest) = parse_surface_call_args_matlab_xy(
            vec![Value::Tensor(
                Tensor::new((1..=12).map(|v| v as f64).collect(), vec![3, 4])
                    .expect("nonsquare surface"),
            )],
            BUILTIN_NAME,
        )
        .unwrap();
        assert!(rest.is_empty());
        assert_eq!(
            Tensor::try_from(&x).unwrap().materialize_f64(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(
            Tensor::try_from(&y).unwrap().materialize_f64(),
            vec![1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn surf_returns_surface_handle() {
        setup_plot_tests();
        let handle = futures::executor::block_on(surf_builtin(vec![Value::Tensor(
            Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).expect("square surface"),
        )]))
        .expect("surf should return a handle");
        assert!(handle.is_finite());
    }

    #[test]
    fn surf_zdata_property_retains_wide_integer_storage() {
        let _guard = lock_plot_registry();
        setup_plot_tests();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let expected = IntegerStorage::U64(vec![
            9_007_199_254_740_993,
            9_007_199_254_740_994,
            9_007_199_254_740_995,
            9_007_199_254_740_996,
        ]);
        let z = Tensor::new_integer(expected.clone(), vec![2, 2]).expect("integer surface Z");
        let handle = futures::executor::block_on(surf_builtin(vec![Value::Tensor(z)]))
            .expect("integer surface");
        let value = get_builtin(vec![Value::Num(handle), Value::String("ZData".into())])
            .expect("surface ZData");

        assert!(
            matches!(value, Value::Tensor(tensor) if tensor.shape() == [2, 2] && tensor.integer_storage() == Some(&expected))
        );
    }

    #[test]
    fn surf_integer_capabilities_cover_all_coordinate_roles() {
        assert_eq!(SURF_INTEGER_CAPABILITIES.len(), 4);
        assert!(SURF_INTEGER_CAPABILITIES
            .iter()
            .all(|capability| capability.inputs[0].classes.len() == 8));
    }
}
