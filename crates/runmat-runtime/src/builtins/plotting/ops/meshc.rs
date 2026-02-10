//! MATLAB-compatible `meshc` builtin (mesh with contour).

use log::warn;
use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::{ColorMap, ShadingMode};

use super::common::{numeric_vector, tensor_to_surface_grid, SurfaceDataInput};
use super::contour::{
    build_contour_gpu_plot, build_contour_plot, default_level_count, ContourLevelSpec,
    ContourLineColor,
};
use super::mesh::build_mesh_surface;
use super::state::{render_active_plot, PlotRenderOptions};

use super::style::{parse_surface_style_args, SurfaceStyleDefaults};
use super::surf::build_surface_gpu_plot;
use crate::builtins::plotting::type_resolvers::string_type;
use std::sync::Arc;

const BUILTIN_NAME: &str = "meshc";

#[runtime_builtin(
    name = "meshc",
    category = "plotting",
    summary = "Render a MATLAB-compatible mesh with contour overlay.",
    keywords = "meshc,plotting,mesh,contour",
    sink = true,
    suppress_auto_output = true,
    type_resolver(string_type),
    builtin_path = "crate::builtins::plotting::meshc"
)]
pub async fn meshc_builtin(
    x: Tensor,
    y: Tensor,
    z: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<String> {
    let x_axis = numeric_vector(x);
    let y_axis = numeric_vector(y);
    let z_input = SurfaceDataInput::from_value(z, "meshc")?;
    let style = Arc::new(parse_surface_style_args(
        "meshc",
        &rest,
        SurfaceStyleDefaults::new(
            ColorMap::Turbo,
            ShadingMode::Faceted,
            true,
            1.0,
            false,
            true,
        ),
    )?);
    let opts = PlotRenderOptions {
        title: "Mesh with Contours",
        x_label: "X",
        y_label: "Y",
        axis_equal: false,
        ..Default::default()
    };
    let level_spec = ContourLevelSpec::Count(default_level_count());
    let contour_map = style.colormap;
    let (mut surface, contour) = if let Some(z_gpu) = z_input.gpu_handle().cloned() {
        match build_surface_gpu_plot(
            BUILTIN_NAME,
            &x_axis,
            &y_axis,
            &z_gpu,
            style.colormap,
            style.alpha,
            style.flatten_z,
        )
        .await
        {
            Ok(surface_gpu) => {
                let mut surface = surface_gpu.with_wireframe(true);
                style.apply_to_plot(&mut surface);
                let base_z = surface.bounds().min.z;
                match build_contour_gpu_plot(
                    BUILTIN_NAME,
                    &x_axis,
                    &y_axis,
                    &z_gpu,
                    contour_map,
                    base_z,
                    &level_spec,
                    &ContourLineColor::Auto,
                ) {
                    Ok(contour) => (surface, contour),
                    Err(err) => {
                        warn!("meshc contour GPU path unavailable: {err}");
                        let z_tensor =
                            super::common::gather_tensor_from_gpu_async(z_gpu, BUILTIN_NAME)
                                .await?;
                        let grid = tensor_to_surface_grid(
                            z_tensor,
                            x_axis.len(),
                            y_axis.len(),
                            BUILTIN_NAME,
                        )?;
                        let base_z = surface.bounds().min.z;
                        let contour = build_contour_plot(
                            BUILTIN_NAME,
                            &x_axis,
                            &y_axis,
                            &grid,
                            contour_map,
                            base_z,
                            &level_spec,
                            &ContourLineColor::Auto,
                        )?;
                        (surface, contour)
                    }
                }
            }
            Err(err) => {
                warn!("meshc surface GPU path unavailable: {err}");
                let z_tensor =
                    super::common::gather_tensor_from_gpu_async(z_gpu, BUILTIN_NAME).await?;
                let grid =
                    tensor_to_surface_grid(z_tensor, x_axis.len(), y_axis.len(), BUILTIN_NAME)?;
                let mut surface = build_mesh_surface(x_axis.clone(), y_axis.clone(), grid.clone())?;
                style.apply_to_plot(&mut surface);
                let base_z = surface.bounds().min.z;
                let contour = build_contour_plot(
                    BUILTIN_NAME,
                    &x_axis,
                    &y_axis,
                    &grid,
                    contour_map,
                    base_z,
                    &level_spec,
                    &ContourLineColor::Auto,
                )?;
                (surface, contour)
            }
        }
    } else {
        let grid = tensor_to_surface_grid(
            z_input.into_tensor(BUILTIN_NAME)?,
            x_axis.len(),
            y_axis.len(),
            BUILTIN_NAME,
        )?;
        let mut surface = build_mesh_surface(x_axis.clone(), y_axis.clone(), grid.clone())?;
        style.apply_to_plot(&mut surface);
        let base_z = surface.bounds().min.z;
        let contour = build_contour_plot(
            BUILTIN_NAME,
            &x_axis,
            &y_axis,
            &grid,
            contour_map,
            base_z,
            &level_spec,
            &ContourLineColor::Auto,
        )?;
        (surface, contour)
    };

    surface = surface
        .with_colormap(ColorMap::Turbo)
        .with_wireframe(true)
        .with_shading(ShadingMode::Faceted);

    let mut surface_opt = Some(surface);
    let mut contour_opt = Some(contour);
    let rendered = render_active_plot(BUILTIN_NAME, opts, move |figure, axes| {
        let surface = surface_opt.take().expect("meshc surface consumed once");
        let contour = contour_opt.take().expect("meshc contour consumed once");
        figure.add_surface_plot_on_axes(surface, axes);
        figure.add_contour_plot_on_axes(contour, axes);
        Ok(())
    })?;
    Ok(rendered)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::plotting::tests::ensure_plot_test_env;
    use runmat_builtins::{ResolveContext, Type};

    fn setup_plot_tests() {
        ensure_plot_test_env();
    }

    fn tensor_from(data: &[f64]) -> Tensor {
        Tensor {
            data: data.to_vec(),
            shape: vec![data.len()],
            rows: data.len(),
            cols: 1,
            dtype: runmat_builtins::NumericDType::F64,
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn meshc_requires_matching_grid() {
        setup_plot_tests();
        let res = futures::executor::block_on(meshc_builtin(
            tensor_from(&[0.0]),
            tensor_from(&[0.0, 1.0]),
            Value::Tensor(Tensor {
                data: vec![0.0],
                shape: vec![1],
                rows: 1,
                cols: 1,
                dtype: runmat_builtins::NumericDType::F64,
            }),
            Vec::new(),
        ));
        assert!(res.is_err());
    }

    #[test]
    fn meshc_type_is_string() {
        assert_eq!(
            string_type(
                &[Type::tensor(), Type::tensor(), Type::tensor()],
                &ResolveContext::new(Vec::new())
            ),
            Type::String
        );
    }
}
