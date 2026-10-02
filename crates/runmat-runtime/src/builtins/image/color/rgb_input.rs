//! Domain admission shared by the truecolor/colormap conversion pair.

use runmat_builtins::RgbConversion;
use runmat_value::{NumericDType, Tensor, Value};

use super::common::{self, ColorLayout};
use crate::{BuiltinResult, RuntimeError};

fn supported(layout: ColorLayout, dtype: NumericDType) -> bool {
    match layout {
        ColorLayout::Truecolor { .. } => matches!(
            dtype,
            NumericDType::F32 | NumericDType::F64 | NumericDType::U8 | NumericDType::U16
        ),
        ColorLayout::Colormap { .. } => dtype == NumericDType::F64,
    }
}

#[derive(Clone, Copy)]
pub(crate) struct ResidentRgbInput {
    pub(crate) layout: ColorLayout,
    pub(crate) dtype: NumericDType,
}

pub(crate) enum RgbAdmission {
    PreparedHost,
    PreparedResident(ResidentRgbInput),
    Unprepared,
}

impl RgbAdmission {
    pub(crate) fn prepares(&self) -> bool {
        !matches!(self, Self::Unprepared)
    }
}

pub(crate) fn host_layout(
    tensor: &Tensor,
    name: &'static str,
    conversion: RgbConversion,
) -> BuiltinResult<ColorLayout> {
    let layout = common::color_layout(tensor, name)?;
    if !supported(layout, tensor.numeric_dtype()) {
        return Err(common::builtin_error(
            name,
            match conversion {
                RgbConversion::Gray => format!(
                    "{name}: unsupported {} class {}",
                    match layout {
                        ColorLayout::Truecolor { .. } => "truecolor",
                        ColorLayout::Colormap { .. } => "colormap",
                    },
                    tensor.numeric_dtype().class_name(),
                ),
                RgbConversion::Hsv => format!(
                    "{name}: {} input is not supported; expected single, double, uint8, or uint16",
                    tensor.numeric_dtype().class_name(),
                ),
            },
        ));
    }
    Ok(layout)
}

pub(crate) fn resident_layout(
    handle: &runmat_accelerate_api::GpuTensorHandle,
    name: &'static str,
) -> BuiltinResult<ResidentRgbInput> {
    let dtype = common::resident_numeric_dtype(handle, name)?;
    let shape = handle.shape();
    let layout = if shape.len() == 3 && shape[2] == 3 && shape[0] > 0 && shape[1] > 0 {
        Some(ColorLayout::Truecolor {
            rows: shape[0],
            cols: shape[1],
        })
    } else if shape.len() == 2 && shape[1] == 3 && shape[0] > 0 {
        Some(ColorLayout::Colormap { rows: shape[0] })
    } else {
        None
    };
    match layout {
        Some(layout) if supported(layout, dtype) => Ok(ResidentRgbInput { layout, dtype }),
        _ => Err(common::builtin_error(
            name,
            format!(
                "{name}: unsupported resident class {} or shape {:?}",
                dtype.class_name(),
                shape
            ),
        )),
    }
}

/// Only already validated image data crosses the shared materialization service.
/// Other values keep the owning public error path without service effects.
pub(crate) fn admitted(
    value: &Value,
    name: &'static str,
    conversion: RgbConversion,
    map: impl FnOnce(RuntimeError) -> RuntimeError,
) -> BuiltinResult<RgbAdmission> {
    match value {
        Value::Tensor(tensor) => host_layout(tensor, name, conversion)
            .map(|_| RgbAdmission::PreparedHost)
            .map_err(map),
        Value::GpuTensor(handle) => resident_layout(handle, name)
            .map(RgbAdmission::PreparedResident)
            .map_err(map),
        _ => Ok(RgbAdmission::Unprepared),
    }
}
