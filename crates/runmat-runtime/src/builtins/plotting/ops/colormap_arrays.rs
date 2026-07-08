//! MATLAB-compatible colormap array generators.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    NumericDType, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::surface::ColorMap;

use super::state::current_colormap_length;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const PARULA: &str = "parula";
const COLORCUBE: &str = "colorcube";
const MAX_COLORMAP_LENGTH: usize = 1_000_000;

const OUTPUT_CMAP: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "c",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "m-by-3 RGB colormap array with values in [0, 1].",
}];

const INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];
const INPUTS_LENGTH: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "m",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("current colormap length or 256"),
    description: "Number of colors as a nonnegative integer.",
}];

const PARULA_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "c = parula()",
        inputs: &INPUTS_NONE,
        outputs: &OUTPUT_CMAP,
    },
    BuiltinSignatureDescriptor {
        label: "c = parula(m)",
        inputs: &INPUTS_LENGTH,
        outputs: &OUTPUT_CMAP,
    },
];

const COLORCUBE_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "c = colorcube()",
        inputs: &INPUTS_NONE,
        outputs: &OUTPUT_CMAP,
    },
    BuiltinSignatureDescriptor {
        label: "c = colorcube(m)",
        inputs: &INPUTS_LENGTH,
        outputs: &OUTPUT_CMAP,
    },
];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COLORMAP_ARRAY.INVALID_ARGUMENT",
    identifier: Some("RunMat:colormapArray:InvalidArgument"),
    when: "The requested colormap length or RGB colormap array is invalid.",
    message: "colormap array: invalid argument",
};

const ERRORS: [BuiltinErrorDescriptor; 1] = [ERROR_INVALID_ARGUMENT];

pub const PARULA_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PARULA_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const COLORCUBE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &COLORCUBE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

fn colormap_type(_args: &[Type], _ctx: &runmat_builtins::ResolveContext) -> Type {
    Type::Tensor { shape: None }
}

#[runtime_builtin(
    name = "parula",
    category = "plotting",
    summary = "Return the parula colormap as an RGB array.",
    keywords = "parula,colormap,plotting,rgb",
    type_resolver(colormap_type),
    descriptor(crate::builtins::plotting::colormap_arrays::PARULA_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::colormap_arrays"
)]
pub fn parula_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let len = parse_length_arg(&args, PARULA)?;
    Ok(Value::Tensor(colormap_tensor(ColorMap::Parula, len)))
}

#[runtime_builtin(
    name = "colorcube",
    category = "plotting",
    summary = "Return the colorcube colormap as an RGB array.",
    keywords = "colorcube,colormap,plotting,rgb",
    type_resolver(colormap_type),
    descriptor(crate::builtins::plotting::colormap_arrays::COLORCUBE_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::colormap_arrays"
)]
pub fn colorcube_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let len = parse_length_arg(&args, COLORCUBE)?;
    Ok(Value::Tensor(colormap_tensor(ColorMap::ColorCube, len)))
}

pub(crate) fn colormap_tensor(map: ColorMap, len: usize) -> Tensor {
    let mut colors = Vec::with_capacity(len);
    for idx in 0..len {
        colors.push(sample_colormap(&map, idx, len));
    }
    rgb_tensor_from_colors(&colors)
}

pub(crate) fn parse_rgb_colormap_tensor(
    tensor: &Tensor,
    builtin: &'static str,
) -> BuiltinResult<(ColorMap, usize)> {
    if tensor.cols != 3 || tensor.data.len() != tensor.rows.saturating_mul(3) {
        return Err(invalid(
            builtin,
            "RGB colormap arrays must be m-by-3 numeric matrices",
        ));
    }
    if tensor.rows == 0 {
        return Err(invalid(
            builtin,
            "RGB colormap arrays must contain at least one color",
        ));
    }
    if tensor.rows > MAX_COLORMAP_LENGTH {
        return Err(invalid(builtin, "RGB colormap array is too large"));
    }

    let mut colors = Vec::with_capacity(tensor.rows);
    for row in 0..tensor.rows {
        let color = [
            tensor.data[row],
            tensor.data[tensor.rows + row],
            tensor.data[2 * tensor.rows + row],
        ];
        if !color.iter().all(|value| value.is_finite()) {
            return Err(invalid(builtin, "RGB colormap entries must be finite"));
        }
        if !color.iter().all(|value| (0.0..=1.0).contains(value)) {
            return Err(invalid(builtin, "RGB colormap entries must be in [0, 1]"));
        }
        colors.push(color);
    }

    let listed = colors
        .iter()
        .map(|color| [color[0] as f32, color[1] as f32, color[2] as f32])
        .collect::<Vec<_>>();
    let map = ColorMap::from_rgb_rows(listed)
        .expect("validated RGB colormap array should produce a listed colormap");
    Ok((map, tensor.rows))
}

fn parse_length_arg(args: &[Value], builtin: &'static str) -> BuiltinResult<usize> {
    match args {
        [] => Ok(current_colormap_length()),
        [value] => colormap_length(value, builtin),
        _ => Err(invalid(builtin, "expected zero or one input argument")),
    }
}

fn colormap_length(value: &Value, builtin: &'static str) -> BuiltinResult<usize> {
    let raw = match value {
        Value::Num(value) => *value,
        Value::Int(value) => value.to_f64(),
        Value::Tensor(tensor) if tensor.data.len() == 1 => tensor.data[0],
        _ => return Err(invalid(builtin, "colormap length must be a numeric scalar")),
    };
    if !raw.is_finite() || raw.fract() != 0.0 || raw < 0.0 {
        return Err(invalid(
            builtin,
            "colormap length must be a nonnegative integer",
        ));
    }
    if raw > MAX_COLORMAP_LENGTH as f64 {
        return Err(invalid(builtin, "colormap length is too large"));
    }
    Ok(raw as usize)
}

fn rgb_tensor_from_colors(colors: &[[f64; 3]]) -> Tensor {
    let rows = colors.len();
    let mut data = Vec::with_capacity(rows.saturating_mul(3));
    for color in colors {
        data.push(color[0]);
    }
    for color in colors {
        data.push(color[1]);
    }
    for color in colors {
        data.push(color[2]);
    }
    Tensor {
        data,
        shape: vec![rows, 3],
        rows,
        cols: 3,
        dtype: NumericDType::F64,
    }
}

fn sample_colormap(map: &ColorMap, idx: usize, len: usize) -> [f64; 3] {
    let t = if len <= 1 {
        0.0
    } else {
        idx as f32 / (len - 1) as f32
    };
    let rgb = map.map_value(t);
    [rgb.x as f64, rgb.y as f64, rgb.z as f64]
}

fn invalid(builtin: &'static str, detail: impl AsRef<str>) -> RuntimeError {
    let mut builder = build_runtime_error(format!(
        "{}: {builtin}: {}",
        ERROR_INVALID_ARGUMENT.message,
        detail.as_ref()
    ))
    .with_builtin(builtin);
    if let Some(identifier) = ERROR_INVALID_ARGUMENT.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::figure::figure_builtin;
    use crate::builtins::plotting::state::set_colormap_with_length;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, reset_hold_state_for_run};

    fn setup() -> crate::builtins::plotting::state::PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    #[test]
    fn descriptors_cover_core_forms() {
        let parula_labels = PARULA_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect::<Vec<_>>();
        assert!(parula_labels.contains(&"c = parula()"));
        assert!(parula_labels.contains(&"c = parula(m)"));

        let colorcube_labels = COLORCUBE_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect::<Vec<_>>();
        assert!(colorcube_labels.contains(&"c = colorcube()"));
        assert!(colorcube_labels.contains(&"c = colorcube(m)"));
    }

    #[test]
    fn parula_and_colorcube_return_m_by_3_rgb_arrays() {
        let parula = parula_builtin(vec![Value::Num(4.0)]).expect("parula");
        let Value::Tensor(parula) = parula else {
            panic!("expected tensor");
        };
        assert_eq!((parula.rows, parula.cols, parula.data.len()), (4, 3, 12));
        assert!(parula.data.iter().all(|v| (0.0..=1.0).contains(v)));

        let colorcube = colorcube_builtin(vec![Value::Num(6.0)]).expect("colorcube");
        let Value::Tensor(colorcube) = colorcube else {
            panic!("expected tensor");
        };
        assert_eq!(
            (colorcube.rows, colorcube.cols, colorcube.data.len()),
            (6, 3, 18)
        );
        assert!(colorcube.data.iter().all(|v| (0.0..=1.0).contains(v)));
    }

    #[test]
    fn default_length_tracks_current_colormap_length() {
        let _guard = setup();
        let _ = figure_builtin(vec![]).expect("figure");
        set_colormap_with_length(ColorMap::ColorCube, 17);
        let Value::Tensor(cmap) = parula_builtin(vec![]).expect("parula default") else {
            panic!("expected tensor");
        };
        assert_eq!((cmap.rows, cmap.cols), (17, 3));
    }

    #[test]
    fn length_argument_validates_nonnegative_integer() {
        let err = parula_builtin(vec![Value::Num(1.5)]).expect_err("fractional length");
        assert!(err.message().contains("nonnegative integer"));
        let err = colorcube_builtin(vec![Value::Num(-1.0)]).expect_err("negative length");
        assert!(err.message().contains("nonnegative integer"));
    }

    #[test]
    fn parse_rgb_colormap_tensor_recognizes_generated_maps_and_custom_gradient() {
        let parula = colormap_tensor(ColorMap::Parula, 8);
        let (map, len) = parse_rgb_colormap_tensor(&parula, "colormap").expect("parse parula");
        assert!(matches!(map, ColorMap::Listed(_)));
        assert_eq!(len, 8);

        let custom = Tensor {
            data: vec![0.0, 1.0, 0.0, 0.5, 1.0, 0.0],
            shape: vec![2, 3],
            rows: 2,
            cols: 3,
            dtype: NumericDType::F64,
        };
        let (map, len) = parse_rgb_colormap_tensor(&custom, "colormap").expect("parse custom");
        assert_eq!(len, 2);
        assert!(matches!(map, ColorMap::Listed(_)));
    }
}
