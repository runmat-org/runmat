//! MATLAB-compatible colormap array generators.

use runmat_builtins::{MAX_PALETTE_LENGTH, PALETTE_INVALID_ARGUMENT};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::surface::ColorMap;
use runmat_value::{IntValue, Tensor, Value};

use super::state::current_colormap_length;
use crate::builtins::common::tensor as tensor_utils;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

macro_rules! define_colormap_builtin {
    ($function:ident, $name:literal, $palette:path) => {
        #[runtime_builtin(
                                    name = $name,
                                    binding_variant = "default",
                                    builtin_path = "crate::builtins::plotting::colormap_arrays"
                                )]
        pub fn $function(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
            let length = parse_length_arg(&args, $name)?;
            Ok(Value::Tensor(colormap_tensor($palette, length)))
        }
    };
}

define_colormap_builtin!(parula_builtin, "parula", ColorMap::Parula);
define_colormap_builtin!(colorcube_builtin, "colorcube", ColorMap::ColorCube);
define_colormap_builtin!(viridis_builtin, "viridis", ColorMap::Viridis);
define_colormap_builtin!(plasma_builtin, "plasma", ColorMap::Plasma);
define_colormap_builtin!(inferno_builtin, "inferno", ColorMap::Inferno);
define_colormap_builtin!(magma_builtin, "magma", ColorMap::Magma);
define_colormap_builtin!(turbo_builtin, "turbo", ColorMap::Turbo);
define_colormap_builtin!(jet_builtin, "jet", ColorMap::Jet);
define_colormap_builtin!(hot_builtin, "hot", ColorMap::Hot);
define_colormap_builtin!(cool_builtin, "cool", ColorMap::Cool);
define_colormap_builtin!(spring_builtin, "spring", ColorMap::Spring);
define_colormap_builtin!(summer_builtin, "summer", ColorMap::Summer);
define_colormap_builtin!(autumn_builtin, "autumn", ColorMap::Autumn);
define_colormap_builtin!(winter_builtin, "winter", ColorMap::Winter);
define_colormap_builtin!(gray_builtin, "gray", ColorMap::Gray);
define_colormap_builtin!(bone_builtin, "bone", ColorMap::Bone);
define_colormap_builtin!(copper_builtin, "copper", ColorMap::Copper);
define_colormap_builtin!(pink_builtin, "pink", ColorMap::Pink);
define_colormap_builtin!(lines_builtin, "lines", ColorMap::Lines);

pub(crate) fn colormap_tensor(map: ColorMap, len: usize) -> Tensor {
    let mut colors = Vec::with_capacity(len);
    for idx in 0..len {
        colors.push(sample_colormap(&map, idx, len));
    }
    rgb_tensor_from_colors(&colors)
}

fn colormap_matrix_dimensions(tensor: &Tensor) -> (usize, usize) {
    match tensor.shape() {
        [] => (0, 0),
        [columns] => (1, *columns),
        [rows, columns, ..] => (*rows, *columns),
    }
}

pub(crate) fn parse_rgb_colormap_tensor(
    tensor: &Tensor,
    builtin: &'static str,
) -> BuiltinResult<(ColorMap, usize)> {
    let (rows, cols) = colormap_matrix_dimensions(tensor);
    if cols != 3 || tensor_utils::tensor_element_len(tensor) != rows.saturating_mul(3) {
        return Err(invalid(
            builtin,
            "RGB colormap arrays must be m-by-3 numeric matrices",
        ));
    }
    if rows == 0 {
        return Err(invalid(
            builtin,
            "RGB colormap arrays must contain at least one color",
        ));
    }
    if rows > MAX_PALETTE_LENGTH {
        return Err(invalid(builtin, "RGB colormap array is too large"));
    }

    let mut colors = Vec::with_capacity(rows);
    let uint8_map = tensor.numeric_dtype() == runmat_value::NumericDType::U8;
    for row in 0..rows {
        let mut color = [
            tensor_utils::tensor_value_f64(tensor, row),
            tensor_utils::tensor_value_f64(tensor, rows + row),
            tensor_utils::tensor_value_f64(tensor, 2 * rows + row),
        ];
        if uint8_map {
            for component in &mut color {
                *component /= f64::from(u8::MAX);
            }
        }
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
    Ok((map, rows))
}

fn parse_length_arg(
    args: &crate::call::BuiltinArguments,
    builtin: &'static str,
) -> BuiltinResult<usize> {
    match args.len() {
        0 => Ok(current_colormap_length()),
        1 => colormap_length(&args[0], builtin),
        _ => Err(invalid(builtin, "expected zero or one input argument")),
    }
}

fn colormap_length(value: &Value, builtin: &'static str) -> BuiltinResult<usize> {
    if let Some(length) = exact_integer_scalar(value) {
        return colormap_integer_length(&length, builtin);
    }
    let raw = match value {
        Value::Num(value) => *value,
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(tensor) => {
            tensor_utils::tensor_value_f64(tensor, 0)
        }
        _ => return Err(invalid(builtin, "colormap length must be a numeric scalar")),
    };
    if !raw.is_finite() || raw.fract() != 0.0 || raw < 0.0 {
        return Err(invalid(
            builtin,
            "colormap length must be a nonnegative integer",
        ));
    }
    if raw > MAX_PALETTE_LENGTH as f64 {
        return Err(invalid(builtin, "colormap length is too large"));
    }
    Ok(raw as usize)
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

fn colormap_integer_length(value: &IntValue, builtin: &'static str) -> BuiltinResult<usize> {
    let Some(length) = value.try_to_usize() else {
        return Err(invalid(
            builtin,
            "colormap length must be a nonnegative integer",
        ));
    };
    if length > MAX_PALETTE_LENGTH {
        return Err(invalid(builtin, "colormap length is too large"));
    }
    Ok(length)
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
    Tensor::new(data, vec![rows, 3]).expect("RGB colormap matrix")
}

fn sample_colormap(map: &ColorMap, idx: usize, len: usize) -> [f64; 3] {
    let t = if len <= 1 {
        0.0
    } else {
        idx as f32 / (len - 1) as f32
    };
    let rgb = map.map_value(t);
    [
        rgb.x.clamp(0.0, 1.0) as f64,
        rgb.y.clamp(0.0, 1.0) as f64,
        rgb.z.clamp(0.0, 1.0) as f64,
    ]
}

fn invalid(builtin: &'static str, detail: impl AsRef<str>) -> RuntimeError {
    let mut builder = build_runtime_error(format!(
        "{}: {builtin}: {}",
        PALETTE_INVALID_ARGUMENT.message,
        detail.as_ref()
    ))
    .with_builtin(builtin);
    if let Some(identifier) = PALETTE_INVALID_ARGUMENT.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn rgb_colormap_preserves_dense_rank_boundaries() {
    let row = Tensor::new(vec![1.0, 0.0, 0.0], vec![3]).unwrap();
    let (_, rows) = parse_rgb_colormap_tensor(&row, "colormap").unwrap();
    assert_eq!(rows, 1);

    let scalar = Tensor::new(vec![1.0], Vec::new()).unwrap();
    let error = match parse_rgb_colormap_tensor(&scalar, "colormap") {
        Ok(_) => panic!("rank-zero tensor has no RGB columns"),
        Err(error) => error,
    };
    assert!(error.message().contains("m-by-3 numeric matrices"));
}

#[cfg(test)]
#[path = "colormap_arrays/tests.rs"]
mod tests;
