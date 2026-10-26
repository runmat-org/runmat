//! `mesh(asset, ...)` display for geometry.Asset and `geometry.load` options.
use super::*;
#[cfg(feature = "plot-core")]
use runmat_builtins::GEOMETRY_MESH_METHOD_ERROR_INPUT as MESH_INPUT;
use runmat_builtins::GEOMETRY_MESH_METHOD_ERROR_OPERATION as MESH_OPERATION;
use runmat_geometry_core::LengthUnit;
use runmat_geometry_io::GeometryImportOptions;

pub(super) const GEOMETRY_MESH_METHOD_NAME: &str = "__runmat_geometry_mesh";

#[runtime_builtin(
    name = "__runmat_geometry_mesh",
    binding_variant = "default",
    builtin_path = "crate::builtins::geometry::display"
)]
pub async fn geometry_mesh_method_builtin(
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let args: Vec<Value> = args.into_iter().collect();
    #[cfg(feature = "plot-core")]
    {
        let (asset, options) = args
            .split_first()
            .ok_or_else(|| builtin_error("mesh", &MESH_INPUT, "mesh requires a geometry.Asset"))?;
        let asset = geometry_asset_from_value_with_error(asset, "mesh", &MESH_INPUT)?;
        let (show_edges, xray) = parse_mesh_options(options)?;
        let mut preview_options = crate::geometry::GeometryPreviewFigureOptions::cad_preview();
        preview_options.edge_overlay_triangle_limit = if show_edges { usize::MAX } else { 0 };
        preview_options.xray = xray;
        let figure = crate::geometry::geometry_preview_figure(&asset, "Geometry", preview_options)
            .map_err(|message| builtin_error("mesh", &MESH_OPERATION, message))?;
        let handle = crate::builtins::plotting::import_runtime_figure(figure);
        Ok(Value::Num(f64::from(handle)))
    }
    #[cfg(not(feature = "plot-core"))]
    {
        let _ = args;
        Err(builtin_error(
            "mesh",
            &MESH_OPERATION,
            "mesh display requires the plot-core runtime feature",
        ))
    }
}

fn option_text(
    value: &Value,
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<String> {
    geometry_path_from_value(value, builtin, error)
}

#[cfg(feature = "plot-core")]
fn parse_mesh_options(args: &[Value]) -> BuiltinResult<(bool, bool)> {
    if !args.len().is_multiple_of(2) {
        return Err(builtin_error(
            "mesh",
            &MESH_INPUT,
            "mesh options must be Name, Value pairs",
        ));
    }
    let mut show_edges = true;
    let mut xray = false;
    for pair in args.chunks(2) {
        let key = option_text(&pair[0], "mesh", &MESH_INPUT)?.to_ascii_lowercase();
        let Value::Bool(value) = pair[1] else {
            return Err(builtin_error(
                "mesh",
                &MESH_INPUT,
                format!("{key} must be logical"),
            ));
        };
        match key.as_str() {
            "edges" => show_edges = value,
            "xray" => xray = value,
            other => {
                return Err(builtin_error(
                    "mesh",
                    &MESH_INPUT,
                    format!("unsupported geometry mesh option `{other}`"),
                ))
            }
        }
    }
    Ok((show_edges, xray))
}

/// `geometry.load(path, 'Units', unit)`: the only option is the source unit
/// used when a file declares none.
pub(super) fn parse_load_options(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<GeometryImportOptions> {
    if !args.len().is_multiple_of(2) {
        return Err(builtin_error(
            GEOMETRY_LOAD_NAME,
            &GEOMETRY_LOAD_ERROR_INPUT,
            "geometry.load options must be Name, Value pairs",
        ));
    }
    let mut options = GeometryImportOptions::default();
    let mut units_seen = false;
    for pair in args.chunks_exact(2) {
        let key = option_text(&pair[0], GEOMETRY_LOAD_NAME, &GEOMETRY_LOAD_ERROR_INPUT)?
            .chars()
            .filter(|character| character.is_ascii_alphanumeric())
            .flat_map(char::to_lowercase)
            .collect::<String>();
        if key != "units" || units_seen {
            let detail = if units_seen && key == "units" {
                "duplicate Units option"
            } else {
                "unsupported geometry.load option"
            };
            return Err(builtin_error(
                GEOMETRY_LOAD_NAME,
                &GEOMETRY_LOAD_ERROR_INPUT,
                detail,
            ));
        }
        units_seen = true;
        options.units = match option_text(&pair[1], GEOMETRY_LOAD_NAME, &GEOMETRY_LOAD_ERROR_INPUT)?
            .to_ascii_lowercase()
            .as_str()
        {
            "m" | "meter" | "meters" | "metre" | "metres" => LengthUnit::Meter,
            "mm" | "millimeter" | "millimeters" | "millimetre" | "millimetres" => {
                LengthUnit::Millimeter
            }
            "in" | "inch" | "inches" => LengthUnit::Inch,
            other => {
                return Err(builtin_error(
                    GEOMETRY_LOAD_NAME,
                    &GEOMETRY_LOAD_ERROR_INPUT,
                    format!("unsupported geometry unit `{other}`"),
                ))
            }
        };
        options.source_units = Some(options.units);
    }
    Ok(options)
}
