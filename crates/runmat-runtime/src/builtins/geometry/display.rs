//! `mesh(asset, ...)` display for geometry.Asset and `geometry.load` options.
use super::*;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinIntegerAuditDescriptor,
    BuiltinIntegerAuditKind, BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor,
    BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_geometry_core::LengthUnit;
use runmat_geometry_io::GeometryImportOptions;

pub(super) const GEOMETRY_MESH_METHOD_NAME: &str = "__runmat_geometry_mesh";

pub const GEOMETRY_MESH_METHOD_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "mesh(asset, ...) accepts a geometry.Asset and logical display controls; it has no integer input role.",
    };

const FIGURE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "figure",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Figure handle.",
}];
const GEOMETRY_MESH_DISPLAY_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "asset",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Loaded geometry.Asset object.",
    },
    BuiltinParamDescriptor {
        name: "Name, Value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Geometry display options: Edges and XRay.",
    },
];
const GEOMETRY_MESH_DISPLAY_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "figure = mesh(asset, Name, Value, ...)",
        inputs: &GEOMETRY_MESH_DISPLAY_INPUTS,
        outputs: &FIGURE_OUTPUT,
    }];
pub const GEOMETRY_MESH_METHOD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GEOMETRY_MESH_DISPLAY_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: runmat_builtins::GEOMETRY_INSPECT_DESCRIPTOR.errors,
};

#[runtime_builtin(
    name = "__runmat_geometry_mesh",
    category = "geometry",
    summary = "Implement mesh display for geometry.Asset objects.",
    keywords = "geometry,mesh,display,cad",
    descriptor(crate::builtins::geometry::GEOMETRY_MESH_METHOD_DESCRIPTOR),
    integer_audit(crate::builtins::geometry::GEOMETRY_MESH_METHOD_INTEGER_AUDIT),
    builtin_path = "crate::builtins::geometry::display"
)]
pub async fn geometry_mesh_method_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    #[cfg(feature = "plot-core")]
    {
        let (asset, options) = args.split_first().ok_or_else(|| {
            builtin_error(
                "mesh",
                &GEOMETRY_INSPECT_ERROR_INTERNAL,
                "mesh requires a geometry.Asset",
            )
        })?;
        let asset = geometry_asset_from_value_with_builtin(asset, "mesh")?;
        let (show_edges, xray) = parse_mesh_options(options)?;
        let mut preview_options = crate::geometry::GeometryPreviewFigureOptions::cad_preview();
        preview_options.edge_overlay_triangle_limit = if show_edges { usize::MAX } else { 0 };
        preview_options.xray = xray;
        let figure = crate::geometry::geometry_preview_figure(&asset, "Geometry", preview_options)
            .map_err(|message| builtin_error("mesh", &GEOMETRY_INSPECT_ERROR_OPERATION, message))?;
        let handle = crate::builtins::plotting::import_runtime_figure(figure);
        Ok(Value::Num(f64::from(handle)))
    }
    #[cfg(not(feature = "plot-core"))]
    {
        let _ = args;
        Err(builtin_error(
            "mesh",
            &GEOMETRY_INSPECT_ERROR_OPERATION,
            "mesh display requires the plot-core runtime feature",
        ))
    }
}

fn option_text(value: &Value, builtin: &'static str) -> BuiltinResult<String> {
    geometry_path_from_value(value, builtin, &GEOMETRY_INSPECT_ERROR_INTERNAL)
}

#[cfg(feature = "plot-core")]
fn parse_mesh_options(args: &[Value]) -> BuiltinResult<(bool, bool)> {
    if !args.len().is_multiple_of(2) {
        return Err(builtin_error(
            "mesh",
            &GEOMETRY_INSPECT_ERROR_INTERNAL,
            "mesh options must be Name, Value pairs",
        ));
    }
    let mut show_edges = true;
    let mut xray = false;
    for pair in args.chunks(2) {
        let key = option_text(&pair[0], "mesh")?.to_ascii_lowercase();
        let Value::Bool(value) = pair[1] else {
            return Err(builtin_error(
                "mesh",
                &GEOMETRY_INSPECT_ERROR_INTERNAL,
                format!("{key} must be logical"),
            ));
        };
        match key.as_str() {
            "edges" => show_edges = value,
            "xray" => xray = value,
            other => {
                return Err(builtin_error(
                    "mesh",
                    &GEOMETRY_INSPECT_ERROR_INTERNAL,
                    format!("unsupported geometry mesh option `{other}`"),
                ))
            }
        }
    }
    Ok((show_edges, xray))
}

/// `geometry.load(path, 'Units', unit)`: the only option is the source unit
/// used when a file declares none.
pub(super) fn parse_load_options(args: &[Value]) -> BuiltinResult<GeometryImportOptions> {
    if !args.len().is_multiple_of(2) {
        return Err(builtin_error(
            GEOMETRY_LOAD_NAME,
            &GEOMETRY_LOAD_ERROR_INPUT,
            "geometry.load options must be Name, Value pairs",
        ));
    }
    let mut options = GeometryImportOptions::default();
    let mut units_seen = false;
    for pair in args.chunks(2) {
        let key = option_text(&pair[0], GEOMETRY_LOAD_NAME)?
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
        options.units = match option_text(&pair[1], GEOMETRY_LOAD_NAME)?
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
