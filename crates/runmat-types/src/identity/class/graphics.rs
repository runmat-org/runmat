//! Specific graphics identities; property support remains owned by the chart.
//!
//! Standard spellings follow the public MathWorks graphics-object reference:
//! https://www.mathworks.com/help/matlab/graphics-object-properties.html
//! An identity is not a claim of every MATLAB property or callback capability.

use super::StaticClassIdentity;

pub const GRAPHICS_COLOR_BAR: StaticClassIdentity =
    StaticClassIdentity::new("matlab.graphics.illustration.ColorBar");
pub const GRAPHICS_HISTOGRAM: StaticClassIdentity =
    StaticClassIdentity::new("matlab.graphics.chart.primitive.Histogram");
pub const GRAPHICS_HISTOGRAM2: StaticClassIdentity =
    StaticClassIdentity::new("matlab.graphics.chart.primitive.Histogram2");
pub const GRAPHICS_LINE: StaticClassIdentity =
    StaticClassIdentity::new("matlab.graphics.chart.primitive.Line");
pub const GRAPHICS_ANIMATED_LINE: StaticClassIdentity =
    StaticClassIdentity::new("matlab.graphics.animation.AnimatedLine");
pub const GRAPHICS_SCATTER: StaticClassIdentity =
    StaticClassIdentity::new("matlab.graphics.chart.primitive.Scatter");
pub const GRAPHICS_BAR: StaticClassIdentity =
    StaticClassIdentity::new("matlab.graphics.chart.primitive.Bar");
pub const GRAPHICS_STEM: StaticClassIdentity =
    StaticClassIdentity::new("matlab.graphics.chart.primitive.Stem");
pub const GRAPHICS_ERROR_BAR: StaticClassIdentity =
    StaticClassIdentity::new("matlab.graphics.chart.primitive.ErrorBar");
pub const GRAPHICS_STAIR: StaticClassIdentity =
    StaticClassIdentity::new("matlab.graphics.chart.primitive.Stair");
pub const GRAPHICS_QUIVER: StaticClassIdentity =
    StaticClassIdentity::new("matlab.graphics.chart.primitive.Quiver");
pub const GRAPHICS_IMAGE: StaticClassIdentity =
    StaticClassIdentity::new("matlab.graphics.primitive.Image");
pub const GRAPHICS_HEATMAP: StaticClassIdentity =
    StaticClassIdentity::new("matlab.graphics.chart.HeatmapChart");
pub const GRAPHICS_FUNCTION_SURFACE: StaticClassIdentity =
    StaticClassIdentity::new("matlab.graphics.function.FunctionSurface");
pub const GRAPHICS_PARAMETERIZED_FUNCTION_SURFACE: StaticClassIdentity =
    StaticClassIdentity::new("matlab.graphics.function.ParameterizedFunctionSurface");
pub const GRAPHICS_FUNCTION_CONTOUR: StaticClassIdentity =
    StaticClassIdentity::new("matlab.graphics.function.FunctionContour");
pub const GRAPHICS_AREA: StaticClassIdentity =
    StaticClassIdentity::new("matlab.graphics.chart.primitive.Area");
pub const GRAPHICS_SURFACE: StaticClassIdentity =
    StaticClassIdentity::new("matlab.graphics.primitive.Surface");
pub const GRAPHICS_PATCH: StaticClassIdentity =
    StaticClassIdentity::new("matlab.graphics.primitive.Patch");
pub const GRAPHICS_CONTOUR: StaticClassIdentity =
    StaticClassIdentity::new("matlab.graphics.chart.primitive.Contour");
pub const GRAPHICS_CONSTANT_LINE: StaticClassIdentity =
    StaticClassIdentity::new("matlab.graphics.chart.decoration.ConstantLine");
pub const GRAPHICS_TEXT: StaticClassIdentity =
    StaticClassIdentity::new("matlab.graphics.primitive.Text");
pub const GRAPHICS_WORD_CLOUD: StaticClassIdentity =
    StaticClassIdentity::new("matlab.graphics.chart.WordCloudChart");
pub const GRAPHICS_STACKED_LINE: StaticClassIdentity =
    StaticClassIdentity::new("matlab.graphics.chart.StackedLineChart");

/// RunMat's aggregated bin-scatter chart, not a generic MATLAB Scatter object.
pub const GRAPHICS_BINSCATTER: StaticClassIdentity =
    StaticClassIdentity::new("runmat.graphics.chart.BinScatter");
/// RunMat's single pie-chart owner, not MATLAB's legacy patch/text output array.
pub const GRAPHICS_PIE: StaticClassIdentity = StaticClassIdentity::new("runmat.graphics.chart.Pie");
/// RunMat's text/marker aggregate, not an invented generic chart fallback.
pub const GRAPHICS_TEXT_SCATTER: StaticClassIdentity =
    StaticClassIdentity::new("runmat.graphics.chart.TextScatter");
