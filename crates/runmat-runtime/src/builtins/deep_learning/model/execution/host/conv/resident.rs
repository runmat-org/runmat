//! One projection from canonical host convolution geometry to physical GPU
//! coordinates. No alternative option parser or numerical kernel lives here.

use runmat_accelerate_api::{
    DeepLearningConv1DPaddingValue as PhysicalPadding, DeepLearningConv1DRequest,
};
use runmat_value::NumericDType;

use crate::builtins::deep_learning::model::plan::axes::AxisPlan;

use super::geometry::{ConvError, ConvOptions, Geometry, PaddingValue};

/// Lower one already-admitted host geometry to the provider's descriptor.
pub(in crate::builtins::deep_learning) fn resident_request(
    axes: &AxisPlan,
    weights_shape: &[usize],
    bias_shape: &[usize],
    dtype: NumericDType,
    options: ConvOptions,
) -> Result<DeepLearningConv1DRequest, ConvError> {
    let geometry = Geometry::from_shapes(axes, weights_shape, bias_shape, dtype, options)?;
    let padding_value = match options.padding_value {
        PaddingValue::Constant(value) => PhysicalPadding::Constant(value),
        PaddingValue::Replicate => PhysicalPadding::Replicate,
        PaddingValue::SymmetricIncludeEdge => PhysicalPadding::SymmetricIncludeEdge,
        PaddingValue::SymmetricExcludeEdge => PhysicalPadding::SymmetricExcludeEdge,
    };
    Ok(DeepLearningConv1DRequest {
        input_shape: axes.shape().to_vec(),
        output_shape: geometry.output_shape,
        weights_shape: [
            geometry.filter_size,
            geometry.input_channels,
            geometry.output_channels,
        ],
        bias_shape: [1, geometry.output_channels],
        span_axis: geometry.span_axis,
        channel_axis: geometry.channel_axis,
        batch_axis: geometry.batch_axis,
        time_axis: geometry.time_axis,
        stride: options.stride,
        dilation: options.dilation,
        padding_left: geometry.left,
        padding_right: geometry.right,
        padding_value,
    })
}
