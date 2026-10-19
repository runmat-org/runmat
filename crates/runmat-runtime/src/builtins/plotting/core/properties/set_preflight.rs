//! Host-visible plot-child data admission before any source-buffer readback.
//! The prepared values are consumed by the owning guarded update; there is no
//! second conversion after suspension.

use super::*;

#[derive(Default)]
pub(super) struct LineDataUpdates {
    pub x: Option<NumericPlotData>,
    pub y: Option<NumericPlotData>,
    pub style_indices: Vec<usize>,
}
impl LineDataUpdates {
    pub fn is_empty(&self) -> bool {
        self.x.is_none() && self.y.is_none()
    }
}

#[derive(Default)]
pub(super) struct ScatterDataUpdates {
    pub x: Option<NumericPlotData>,
    pub y: Option<NumericPlotData>,
    pub theta: Option<Vec<f64>>,
    pub r: Option<Vec<f64>>,
    pub style_indices: Vec<usize>,
}
impl ScatterDataUpdates {
    pub fn is_empty(&self) -> bool {
        self.x.is_none() && self.y.is_none() && self.theta.is_none() && self.r.is_none()
    }
}

#[derive(Default)]
pub(super) struct Line3DataUpdates {
    pub x: Option<NumericPlotData>,
    pub y: Option<NumericPlotData>,
    pub z: Option<NumericPlotData>,
    pub style_indices: Vec<usize>,
}
impl Line3DataUpdates {
    pub fn is_empty(&self) -> bool {
        self.x.is_none() && self.y.is_none() && self.z.is_none()
    }
}

pub(super) fn prepare_line_data_updates(
    pairs: &[(String, &Value)],
    builtin: &'static str,
) -> BuiltinResult<LineDataUpdates> {
    let mut prepared = LineDataUpdates::default();
    for (index, (key, value)) in pairs.iter().enumerate() {
        match key.as_str() {
            "xdata" => prepared.x = Some(line_numeric_data(value, "XData", builtin)?),
            "ydata" => prepared.y = Some(line_numeric_data(value, "YData", builtin)?),
            "handlevisibility" => {
                parse_handle_visibility(&LineStyleParseOptions::generic(builtin), value)?;
            }
            _ => prepared.style_indices.push(index),
        }
    }
    Ok(prepared)
}

pub(super) fn prepare_scatter_data_updates(
    pairs: &[(String, &Value)],
    builtin: &'static str,
) -> BuiltinResult<ScatterDataUpdates> {
    let mut prepared = ScatterDataUpdates::default();
    for (index, (key, value)) in pairs.iter().enumerate() {
        match key.as_str() {
            "xdata" => prepared.x = Some(line_numeric_data(value, "XData", builtin)?),
            "ydata" => prepared.y = Some(line_numeric_data(value, "YData", builtin)?),
            "thetadata" => {
                prepared.theta =
                    Some(line_numeric_data(value, "ThetaData", builtin)?.materialize_f64());
            }
            "rdata" => {
                prepared.r = Some(line_numeric_data(value, "RData", builtin)?.materialize_f64());
            }
            _ => prepared.style_indices.push(index),
        }
    }
    Ok(prepared)
}

pub(super) fn prepare_line3_data_updates(
    pairs: &[(String, &Value)],
    builtin: &'static str,
) -> BuiltinResult<Line3DataUpdates> {
    let mut prepared = Line3DataUpdates::default();
    for (index, (key, value)) in pairs.iter().enumerate() {
        match key.as_str() {
            "xdata" => prepared.x = Some(line_numeric_data(value, "XData", builtin)?),
            "ydata" => prepared.y = Some(line_numeric_data(value, "YData", builtin)?),
            "zdata" => prepared.z = Some(line_numeric_data(value, "ZData", builtin)?),
            _ => prepared.style_indices.push(index),
        }
    }
    Ok(prepared)
}
