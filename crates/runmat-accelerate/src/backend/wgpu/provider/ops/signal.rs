use super::elementwise::{IndexedFusionMultiRequest, IndexedFusionOperand};
use super::*;
use crate::backend::wgpu::params::{
    MovingWindowParamsF32, MovingWindowParamsF64, PackedF32, PackedU32,
};
use crate::backend::wgpu::shaders::signal::{
    analytic_signal_mask_shader, envelope_analytic_bounds_shader,
    envelope_analytic_fir_bounds_shader, envelope_analytic_mask_shader,
    envelope_center_real_to_complex_shader, envelope_rms_bounds_shader, spectral_frame_shader,
    spectral_power_shader, spectral_select_shader, SpectralFrameShaderConfig,
    SpectralFrameShaderMode, SpectralRangeShaderMode,
};
use runmat_accelerate_api::{
    ProviderEnvelopeMethod, ProviderEnvelopeRequest, ProviderEnvelopeResult,
    ProviderHilbertRequest, ProviderMovingWindowEndpoints, ProviderMovingWindowOp,
    ProviderMovingWindowRequest, ProviderNanMode, ProviderStdNormalization,
    ProviderTrapezoidSpacing,
};

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "signal/spectral_preflight_tests.rs"]
mod spectral_preflight_tests;

#[cfg(test)]
#[path = "signal/specialized_moving_window_validation_tests.rs"]
mod specialized_moving_window_validation_tests;
#[cfg(test)]
#[path = "signal/required_geometry_tests.rs"]
mod required_geometry_tests;

#[derive(Clone, Copy)]
enum GradientSpacingInput<'a> {
    Scalar(f64),
    Coordinates(&'a GpuTensorHandle),
}

enum IirCoefficientPlan<'a> {
    UnitDenominatorFir {
        numerator: &'a GpuTensorHandle,
    },
    PreparedResident {
        numerator: &'a GpuTensorHandle,
        denominator: &'a GpuTensorHandle,
    },
    Normalized {
        b: std::borrow::Cow<'a, [f64]>,
        a: std::borrow::Cow<'a, [f64]>,
    },
}

/// Owns the device-side finite mask and its scalar reduction until the
/// readback reaches a terminal state. Dropping a pending validation therefore
/// retires both private intermediates without touching the borrowed signal.
struct EnvelopeFiniteValidation {
    mask: GpuTensorHandle,
    all: GpuTensorHandle,
}

impl EnvelopeFiniteValidation {
    fn scalar(&self) -> &GpuTensorHandle {
        &self.all
    }

    fn complete(
        self,
        finite: anyhow::Result<HostTensorOwned>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<()> {
        use runmat_accelerate_api::{EnvelopeSignalDomainError, ProviderExecutionError};

        let Self { mask, all } = self;
        drop(all);
        drop(mask);
        let finite = finite.map_err(|error| ProviderExecutionError::from(error).after_effect())?;
        if finite.data.as_slice() != [1.0] {
            return Err(
                ProviderExecutionError::from_source(EnvelopeSignalDomainError::NonFinite)
                    .after_effect(),
            );
        }
        Ok(())
    }
}

fn moving_window_op_code(op: ProviderMovingWindowOp) -> u32 {
    match op {
        ProviderMovingWindowOp::Sum => 0,
        ProviderMovingWindowOp::Mean => 1,
        ProviderMovingWindowOp::Prod => 2,
        ProviderMovingWindowOp::Min => 3,
        ProviderMovingWindowOp::Max => 4,
        ProviderMovingWindowOp::Std => 5,
        ProviderMovingWindowOp::Var => 6,
        ProviderMovingWindowOp::Median => 7,
    }
}

fn moving_window_endpoint_code(endpoint: ProviderMovingWindowEndpoints) -> (u32, f64) {
    match endpoint {
        ProviderMovingWindowEndpoints::Shrink => (0, 0.0),
        ProviderMovingWindowEndpoints::Discard => (1, 0.0),
        ProviderMovingWindowEndpoints::Fill(value) => (2, value),
    }
}

fn specialized_moving_window_fill_helper(precision: NumericPrecision, value: f64) -> String {
    match precision {
        NumericPrecision::F32 => format!(
            "fn moving_window_fill_value() -> f32 {{ var bits: u32 = 0x{:08x}u; return bitcast<f32>(bits); }}",
            (value as f32).to_bits()
        ),
        NumericPrecision::F64 => {
            let bits = value.to_bits();
            format!(
                "fn moving_window_fill_value() -> f64 {{ var low: u32 = 0x{:08x}u; var high: u32 = 0x{:08x}u; return runmat_f64_from_words(low, high); }}",
                bits as u32,
                (bits >> 32) as u32
            )
        }
    }
}

fn finish_specialized_moving_window_shader(precision: NumericPrecision, source: String) -> String {
    match precision {
        NumericPrecision::F64 => crate::backend::wgpu::shaders::ieee_words::inject(&source),
        NumericPrecision::F32 => source,
    }
}

const WGPU_MOVING_MEDIAN_MAX_WINDOW: usize = 256;
const WGPU_MOVING_WINDOW_I32_LIMIT: usize = i32::MAX as usize;

struct MovingWindowShaderSpec {
    precision: NumericPrecision,
    output_len: usize,
    axis_len: usize,
    out_axis_len: usize,
    stride_before: usize,
    before: usize,
    after: usize,
    op: ProviderMovingWindowOp,
    endpoints: ProviderMovingWindowEndpoints,
    nan_mode: ProviderNanMode,
    normalization: ProviderStdNormalization,
    fill_value: f64,
}

fn build_specialized_moving_window_shader(spec: &MovingWindowShaderSpec) -> String {
    let ty = spec.precision.as_str();
    let nan_helpers = match spec.precision {
        NumericPrecision::F64 => {
            r#"
fn is_nan(value: f64) -> bool {
  let high = runmat_f64_high(value);
  let low = runmat_f64_low(value);
  return (high & 0x7ff00000u) == 0x7ff00000u &&
    ((high & 0x000fffffu) != 0u || low != 0u);
}
fn nan_value() -> f64 { return runmat_f64_from_words(0u, 0x7ff80000u); }
"#
        }
        NumericPrecision::F32 => {
            r#"
fn is_nan(value: f32) -> bool {
  let bits = bitcast<u32>(value);
  return (bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0u;
}
fn nan_value() -> f32 {
  let nan_bits: u32 = 0x7fc00000u;
  return bitcast<f32>(nan_bits);
}
"#
        }
    };
    let fill_helper = specialized_moving_window_fill_helper(spec.precision, spec.fill_value);
    let fill = "moving_window_fill_value()";
    let endpoint = moving_window_endpoint_code(spec.endpoints).0;
    let omit_nan = matches!(spec.nan_mode, ProviderNanMode::Omit);
    let workgroup = crate::backend::wgpu::config::WORKGROUP_SIZE;
    let common = format!(
        r#"
struct Tensor {{ data: array<{ty}>, }};
@group(0) @binding(0) var<storage, read> Input: Tensor;
@group(0) @binding(1) var<storage, read_write> Output: Tensor;
{nan_helpers}
{fill_helper}
"#
    );
    let main_prefix = format!(
        r#"
@compute @workgroup_size({workgroup})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
  let idx = gid.x;
  if (idx >= {output_len}u) {{ return; }}
  let before_idx = idx % {stride_before}u;
  let tmp = idx / {stride_before}u;
  let out_pos = tmp % {out_axis_len}u;
  let after_idx = tmp / {out_axis_len}u;
  let center = out_pos + {discard_offset}u;
  let start = i32(center) - {before};
  let end = i32(center) + {after};
  let in_start = max(start, 0);
  let in_end = min(end, {last_axis});
  var fill_count = 0u;
  if ({fill_enabled}) {{
    if (start < 0) {{ fill_count = fill_count + u32(-start); }}
    if (end > {last_axis}) {{ fill_count = fill_count + u32(end - {last_axis}); }}
  }}
"#,
        output_len = spec.output_len,
        stride_before = spec.stride_before,
        out_axis_len = spec.out_axis_len,
        discard_offset = if endpoint == 1 { spec.before } else { 0 },
        before = spec.before,
        after = spec.after,
        last_axis = spec.axis_len as i32 - 1,
        fill_enabled = endpoint == 2,
    );

    if matches!(spec.op, ProviderMovingWindowOp::Median) {
        let median_body = format!(
            r#"
  var values: array<{ty}, {max_window}>;
  var count = 0u;
  var saw_nan = false;
  if (in_start <= in_end) {{
    var pos = u32(in_start);
    loop {{
      if (pos > u32(in_end)) {{ break; }}
      let input_idx = before_idx + pos * {stride_before}u + after_idx * {stride_before}u * {axis_len}u;
      let value = Input.data[input_idx];
      if (is_nan(value)) {{
        if (!{omit_nan}) {{ saw_nan = true; break; }}
      }} else {{ values[count] = value; count = count + 1u; }}
      pos = pos + 1u;
    }}
  }}
  if (!saw_nan && fill_count > 0u) {{
    if (is_nan({fill})) {{
      if (!{omit_nan}) {{ saw_nan = true; }}
    }} else {{
      var fill_idx = 0u;
      loop {{
        if (fill_idx >= fill_count) {{ break; }}
        values[count] = {fill}; count = count + 1u; fill_idx = fill_idx + 1u;
      }}
    }}
  }}
  if (saw_nan || count == 0u) {{ Output.data[idx] = nan_value(); return; }}
  var i = 1u;
  loop {{
    if (i >= count) {{ break; }}
    let key = values[i];
    var j = i;
    loop {{
      if (j == 0u || values[j - 1u] <= key) {{ break; }}
      values[j] = values[j - 1u]; j = j - 1u;
    }}
    values[j] = key; i = i + 1u;
  }}
  let lower = values[(count - 1u) / 2u];
  let upper = values[count / 2u];
  Output.data[idx] = (lower + upper) * {ty}(0.5);
}}
"#,
            max_window = WGPU_MOVING_MEDIAN_MAX_WINDOW,
            stride_before = spec.stride_before,
            axis_len = spec.axis_len,
        );
        return finish_specialized_moving_window_shader(
            spec.precision,
            common + &main_prefix + &median_body,
        );
    }

    let empty_value = match spec.op {
        ProviderMovingWindowOp::Sum => format!("{ty}(0.0)"),
        ProviderMovingWindowOp::Prod => format!("{ty}(1.0)"),
        _ => "nan_value()".to_owned(),
    };
    let result = match spec.op {
        ProviderMovingWindowOp::Sum => "sum".to_owned(),
        ProviderMovingWindowOp::Mean => format!("sum / {ty}(count)"),
        ProviderMovingWindowOp::Prod => "product".to_owned(),
        ProviderMovingWindowOp::Min => "minimum".to_owned(),
        ProviderMovingWindowOp::Max => "maximum".to_owned(),
        ProviderMovingWindowOp::Std | ProviderMovingWindowOp::Var => {
            let denominator = match spec.normalization {
                ProviderStdNormalization::Sample => {
                    format!("{ty}(select(1u, count - 1u, count > 1u))")
                }
                ProviderStdNormalization::Population => format!("{ty}(count)"),
            };
            let variance = format!("m2 / {denominator}");
            if matches!(spec.op, ProviderMovingWindowOp::Std) {
                format!("sqrt({variance})")
            } else {
                variance
            }
        }
        ProviderMovingWindowOp::Median => unreachable!(),
    };
    let body = format!(
        r#"
  var count = 0u;
  var sum = {ty}(0.0);
  var product = {ty}(1.0);
  var minimum = {ty}(0.0);
  var maximum = {ty}(0.0);
  var mean = {ty}(0.0);
  var m2 = {ty}(0.0);
  var saw_nan = false;
  if (in_start <= in_end) {{
    var pos = u32(in_start);
    loop {{
      if (pos > u32(in_end)) {{ break; }}
      let input_idx = before_idx + pos * {stride_before}u + after_idx * {stride_before}u * {axis_len}u;
      let value = Input.data[input_idx];
      if (is_nan(value)) {{
        if (!{omit_nan}) {{ saw_nan = true; break; }}
      }} else {{
        count = count + 1u; sum = sum + value; product = product * value;
        if (count == 1u) {{ minimum = value; maximum = value; }}
        else {{ minimum = min(minimum, value); maximum = max(maximum, value); }}
        let delta = value - mean; mean = mean + delta / {ty}(count);
        m2 = m2 + delta * (value - mean);
      }}
      pos = pos + 1u;
    }}
  }}
  if (!saw_nan && fill_count > 0u) {{
    var fill_idx = 0u;
    loop {{
      if (fill_idx >= fill_count) {{ break; }}
      let value = {fill};
      if (is_nan(value)) {{ if (!{omit_nan}) {{ saw_nan = true; }} }}
      else {{
        count = count + 1u; sum = sum + value; product = product * value;
        if (count == 1u) {{ minimum = value; maximum = value; }}
        else {{ minimum = min(minimum, value); maximum = max(maximum, value); }}
        let delta = value - mean; mean = mean + delta / {ty}(count);
        m2 = m2 + delta * (value - mean);
      }}
      fill_idx = fill_idx + 1u;
    }}
  }}
  if (saw_nan) {{ Output.data[idx] = nan_value(); }}
  else if (count == 0u) {{ Output.data[idx] = {empty_value}; }}
  else {{ Output.data[idx] = {result}; }}
}}
"#,
        stride_before = spec.stride_before,
        axis_len = spec.axis_len,
    );
    finish_specialized_moving_window_shader(spec.precision, common + &main_prefix + &body)
}

impl WgpuProvider {
    pub(crate) async fn uniform_spectral_estimate_exec(
        &self,
        request: &ProviderSpectralRequest<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<ProviderSpectralResult> {
        runmat_accelerate_api::validate_uniform_spectral_request(request)?;
        // Authenticate the original allocation before the first window upload.
        let input = self.get_entry(request.input)?;
        let component_count = request
            .input_len
            .checked_mul(if request.input_complex { 2 } else { 1 })
            .ok_or_else(|| anyhow!("uniform_spectral_estimate: input too large"))?;
        if input.len < component_count {
            return Err(anyhow!("uniform_spectral_estimate: input length mismatch").into());
        }
        if runmat_accelerate_api::handle_is_transposed(request.input) {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "uniform_spectral_estimate: linear framing requires contiguous input",
            )
            .into());
        }

        let framed_len = request
            .nfft
            .checked_mul(request.frame_count)
            .and_then(|len| len.checked_mul(2))
            .ok_or_else(|| anyhow!("uniform_spectral_estimate: frame too large"))?;
        let mode_dimensions = match request.frame_mode {
            ProviderSpectralFrameMode::Sliding { hop } => [hop, 0, 0],
            ProviderSpectralFrameMode::ColumnSliding {
                hop,
                input_rows,
                frames_per_column,
            } => [hop, input_rows, frames_per_column],
            ProviderSpectralFrameMode::FoldedColumns { input_rows } => [input_rows, 0, 0],
        };
        if [framed_len, component_count, request.window.len()]
            .into_iter()
            .chain(mode_dimensions)
            .any(|n| n > u32::MAX as usize)
        {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "uniform_spectral_estimate: dimensions exceed GPU kernel limits",
            )
            .into());
        }

        let window_shape = [request.window.len(), 1usize];
        let window = self.upload_exec(&HostTensorView {
            data: request.window,
            shape: &window_shape,
        })?;
        let frame_shader = spectral_frame_shader(
            &SpectralFrameShaderConfig {
                input_complex: request.input_complex,
                mode: spectral_frame_shader_mode(request.frame_mode),
                window_len: request.window.len(),
                nfft: request.nfft,
                frame_count: request.frame_count,
                input_len: request.input_len,
            },
            self.precision,
        );
        let frame_inputs = [request.input.clone(), window.clone()];
        let framed = match self.fused_elementwise_with_telemetry_and_storage_exec(
            &frame_shader,
            &frame_inputs,
            &[request.nfft, request.frame_count],
            framed_len,
            GpuTensorStorage::ComplexInterleaved,
        ) {
            Ok(framed) => framed,
            Err(err) => {
                drop(window);
                return Err(err.into());
            }
        };
        drop(window);

        let spectrum = match self.fft_dim_exec(&framed, None, 0).await {
            Ok(spectrum) => spectrum,
            Err(err) => {
                drop(framed);
                return Err(err);
            }
        };
        drop(framed);

        let rows = spectral_selected_frequency_len(request.nfft, request.range);
        let range = spectral_range_shader_mode(request.range);
        let selected = if matches!(request.range, ProviderSpectralRange::Twosided) {
            spectrum
        } else {
            let selected_len = rows
                .checked_mul(request.frame_count)
                .and_then(|len| len.checked_mul(2))
                .ok_or_else(|| anyhow!("uniform_spectral_estimate: output too large"))?;
            let shader = spectral_select_shader(request.nfft, rows, range, self.precision);
            let handle = match self.fused_elementwise_with_telemetry_and_storage_exec(
                &shader,
                std::slice::from_ref(&spectrum),
                &[rows, request.frame_count],
                selected_len,
                GpuTensorStorage::ComplexInterleaved,
            ) {
                Ok(handle) => handle,
                Err(err) => {
                    drop(spectrum);
                    return Err(err.into());
                }
            };
            drop(spectrum);
            handle
        };

        let ps_len = rows
            .checked_mul(request.frame_count)
            .ok_or_else(|| anyhow!("uniform_spectral_estimate: power output too large"))?;
        let ps_shader = spectral_power_shader(
            rows,
            range,
            request.nfft.is_multiple_of(2),
            request.denominator,
            self.precision,
        );
        let ps = match self.fused_elementwise_with_telemetry_exec(
            &ps_shader,
            std::slice::from_ref(&selected),
            &[rows, request.frame_count],
            ps_len,
        ) {
            Ok(ps) => ps,
            Err(err) => {
                drop(selected);
                return Err(err.into());
            }
        };

        Ok(ProviderSpectralResult {
            s: selected,
            ps,
            rows,
            cols: request.frame_count,
        })
    }

    /// Side-effect-free request admission. This does not promise successful
    /// execution: failures after invocation remain terminal to the caller.
    pub(crate) fn signal_envelope_supported_exec(
        &self,
        request: &ProviderEnvelopeRequest<'_>,
    ) -> bool {
        let Ok(entry) = self.get_entry(request.input) else {
            return false;
        };
        self.signal_envelope_supported_entry(request, &entry)
    }

    fn signal_envelope_supported_entry(
        &self,
        request: &ProviderEnvelopeRequest<'_>,
        entry: &BufferEntry,
    ) -> bool {
        let Some(input_len) = request.channel_len.checked_mul(request.channel_count) else {
            return false;
        };
        let Some(output_len) = product_checked(request.output_shape) else {
            return false;
        };
        if request.channel_len == 0
            || request.channel_count == 0
            || input_len != output_len
            || input_len > u32::MAX as usize
            || entry.len != input_len
            || entry.precision != self.precision
            || entry.storage != GpuTensorStorage::Real
            || runmat_accelerate_api::handle_storage(request.input) != GpuTensorStorage::Real
            || entry.integer_type().is_some()
            || runmat_accelerate_api::handle_is_logical(request.input)
            || !envelope_input_shape_matches(
                &entry.shape,
                request.channel_len,
                request.channel_count,
            )
            || !envelope_input_shape_matches(
                request.input.shape(),
                request.channel_len,
                request.channel_count,
            )
            || !envelope_input_shape_matches(
                request.output_shape,
                request.channel_len,
                request.channel_count,
            )
        {
            return false;
        }
        match request.method {
            ProviderEnvelopeMethod::Analytic => input_len
                .checked_mul(2)
                .is_some_and(|n| n <= u32::MAX as usize),
            ProviderEnvelopeMethod::AnalyticFir { filter_len } => {
                filter_len > 0 && filter_len <= u32::MAX as usize
            }
            ProviderEnvelopeMethod::Rms { window_len } => {
                window_len > 0 && window_len <= u32::MAX as usize
            }
        }
    }

    pub(crate) async fn signal_envelope_exec(
        &self,
        request: &ProviderEnvelopeRequest<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<ProviderEnvelopeResult> {
        let output_len = request
            .output_shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| anyhow!("signal_envelope: invalid request"))?;
        let input_len = request
            .channel_len
            .checked_mul(request.channel_count)
            .ok_or_else(|| anyhow!("signal_envelope: invalid request"))?;
        if !(request.channel_len > 0 && request.channel_count > 0 && output_len == input_len) {
            return Err(anyhow!("signal_envelope: invalid request").into());
        }
        let entry = self.get_entry(request.input)?;
        if !self.signal_envelope_supported_entry(request, &entry) {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "signal_envelope: input view or geometry is not supported",
            )
            .into());
        }
        if entry.precision != self.precision {
            return Err(
                anyhow!("signal_envelope: mixed precision tensors are not supported").into(),
            );
        }
        if !(entry.storage == GpuTensorStorage::Real
            && runmat_accelerate_api::handle_storage(request.input) == GpuTensorStorage::Real)
        {
            return Err(anyhow!("signal_envelope: complex input tensors are not supported").into());
        }
        if entry.len != input_len {
            return Err(anyhow!("signal_envelope: input length mismatch").into());
        }
        if !(envelope_input_shape_matches(&entry.shape, request.channel_len, request.channel_count)
            && envelope_input_shape_matches(
                request.input.shape(),
                request.channel_len,
                request.channel_count,
            ))
        {
            return Err(anyhow!("signal_envelope: input shape mismatch").into());
        }

        // Invalid controls must not trigger a lazy-view materialization.
        if matches!(
            request.method,
            ProviderEnvelopeMethod::AnalyticFir { filter_len: 0 }
                | ProviderEnvelopeMethod::Rms { window_len: 0 }
        ) {
            return Err(anyhow!("signal_envelope: invalid request").into());
        }
        // All envelope kernels directly index logical column-major words,
        // including analytic centering and the final complex bounds. A lazy
        // transpose is an addressing view, not a reordered allocation.
        // Existing reshape materializes transpose through the bit-preserving
        // permutation owner, without mutating the borrowed source view.
        let materialized = if runmat_accelerate_api::handle_is_transposed(request.input) {
            Some(
                self.reshape_exec(request.input, request.input.shape())
                    .map_err(|error| {
                        runmat_accelerate_api::ProviderExecutionError::from(error)
                            .with_unproven_effects()
                    })?,
            )
        } else {
            None
        };
        let prepared = ProviderEnvelopeRequest {
            input: materialized.as_ref().unwrap_or(request.input),
            channel_len: request.channel_len,
            channel_count: request.channel_count,
            output_shape: request.output_shape,
            method: request.method,
        };
        let request = &prepared;
        self.validate_envelope_finite(request.input)
            .await
            .map_err(|error| {
                if materialized.is_some() {
                    error.after_effect()
                } else {
                    error
                }
            })?;
        let result: runmat_accelerate_api::ProviderExecutionResult<ProviderEnvelopeResult> =
            match request.method {
                ProviderEnvelopeMethod::Analytic => {
                    self.signal_envelope_analytic_exec(request).await
                }
                ProviderEnvelopeMethod::AnalyticFir { filter_len } => self
                    .signal_envelope_analytic_fir_exec(request, filter_len)
                    .map_err(Into::into),
                ProviderEnvelopeMethod::Rms { window_len } => self
                    .signal_envelope_rms_exec(request, window_len)
                    .map_err(Into::into),
            };
        result.map_err(runmat_accelerate_api::ProviderExecutionError::after_effect)
    }

    /// A resident input-domain check shared by every envelope method. The
    /// classification and reduction operate on the prepared logical view;
    /// only one scalar is read back. All temporary handles retire on every
    /// return path, and every rejection after classification is post-effect.
    async fn validate_envelope_finite(
        &self,
        input: &GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionResult<()> {
        let validation = self.prepare_envelope_finite_validation(input)?;
        let finite = self.download_exec(validation.scalar()).await;
        validation.complete(finite)
    }

    fn prepare_envelope_finite_validation(
        &self,
        input: &GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionResult<EnvelopeFiniteValidation> {
        use runmat_accelerate_api::ProviderExecutionError;

        let mask = self
            .logical_isfinite_exec(input)
            .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?;
        let all = match self.reduce_all_exec(&mask, false) {
            Ok(all) => all,
            Err(error) => {
                drop(mask);
                return Err(ProviderExecutionError::from(error).after_effect());
            }
        };
        Ok(EnvelopeFiniteValidation { mask, all })
    }

    /// Side-effect-free admission before FFT staging. A later hook failure is
    /// terminal even when it reports Unsupported.
    pub(crate) fn signal_hilbert_supported_exec(
        &self,
        request: &ProviderHilbertRequest<'_>,
    ) -> bool {
        let Ok(entry) = self.get_entry(request.input) else {
            return false;
        };
        self.signal_hilbert_supported_entry(request, &entry)
    }

    fn signal_hilbert_supported_entry(
        &self,
        request: &ProviderHilbertRequest<'_>,
        entry: &BufferEntry,
    ) -> bool {
        if entry.precision != self.precision
            || entry.storage != GpuTensorStorage::Real
            || runmat_accelerate_api::handle_storage(request.input) != GpuTensorStorage::Real
            || runmat_accelerate_api::handle_is_transposed(request.input)
            || entry.integer_type().is_some()
            || runmat_accelerate_api::handle_is_logical(request.input)
            || request.length == Some(0)
        {
            return false;
        }
        let shape = request.input.shape();
        let Some(&current_len) = shape.get(request.dim) else {
            return false;
        };
        let target_len = request.length.unwrap_or(current_len);
        if current_len == 0 || target_len == 0 || target_len > u32::MAX as usize {
            return false;
        }
        let Some(inner_stride) = product_checked(&shape[..request.dim]) else {
            return false;
        };
        let Some(outer_stride) = product_checked(&shape[request.dim + 1..]) else {
            return false;
        };
        inner_stride > 0
            && inner_stride <= u32::MAX as usize
            && outer_stride
                .checked_mul(inner_stride)
                .and_then(|slices| slices.checked_mul(target_len))
                .and_then(|values| values.checked_mul(2))
                .is_some_and(|complex_words| complex_words <= u32::MAX as usize)
    }

    pub(crate) async fn signal_hilbert_exec(
        &self,
        request: &ProviderHilbertRequest<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        // Authenticate the owning allocation first: an expired or foreign
        // handle keeps its actual owner/lease error instead of being recast
        // as an optional-hook Unsupported result.
        let entry = self.get_entry(request.input)?;
        if !self.signal_hilbert_supported_entry(request, &entry) {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "signal_hilbert: input view or geometry is not supported",
            )
            .into());
        }

        let spectrum = self
            .fft_dim_exec(request.input, request.length, request.dim)
            .await?;
        let filtered_result = (|| -> Result<GpuTensorHandle> {
            let spectrum_entry = self.get_entry(&spectrum)?;
            ensure!(
                spectrum_entry.storage == GpuTensorStorage::ComplexInterleaved,
                "signal_hilbert: fft output was not complex"
            );
            ensure!(
                spectrum_entry.len <= u32::MAX as usize,
                "signal_hilbert: output too large"
            );

            let mut shape = spectrum_entry.shape.clone();
            while shape.len() <= request.dim {
                shape.push(1);
            }
            let transform_len = shape[request.dim];
            ensure!(
                transform_len > 0,
                "signal_hilbert: invalid transform length"
            );
            let inner_stride = product_checked(&shape[..request.dim])
                .ok_or_else(|| anyhow!("signal_hilbert: shape overflow"))?;
            ensure!(
                inner_stride > 0
                    && inner_stride <= u32::MAX as usize
                    && transform_len <= u32::MAX as usize,
                "signal_hilbert: dimensions exceed GPU kernel limits"
            );

            ensure!(
                spectrum_entry.len % 2 == 0,
                "signal_hilbert: invalid complex output length"
            );
            let shader = analytic_signal_mask_shader(
                transform_len,
                inner_stride,
                spectrum_entry.len,
                self.precision,
            );
            self.fused_elementwise_with_telemetry_and_storage_exec(
                &shader,
                std::slice::from_ref(&spectrum),
                &spectrum_entry.shape,
                spectrum_entry.len,
                GpuTensorStorage::ComplexInterleaved,
            )
        })();
        let filtered = match filtered_result {
            Ok(filtered) => filtered,
            Err(err) => {
                drop(spectrum);
                return Err(err.into());
            }
        };
        drop(spectrum);

        let analytic = match self.ifft_dim_exec(&filtered, None, request.dim).await {
            Ok(analytic) => analytic,
            Err(err) => {
                drop(filtered);
                return Err(err);
            }
        };
        drop(filtered);
        Ok(analytic)
    }

    async fn signal_envelope_analytic_exec(
        &self,
        request: &ProviderEnvelopeRequest<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<ProviderEnvelopeResult> {
        let complex_len = request
            .channel_len
            .checked_mul(request.channel_count)
            .and_then(|len| len.checked_mul(2))
            .ok_or_else(|| anyhow!("signal_envelope: analytic output too large"))?;
        let channel_shape = [request.channel_len, request.channel_count];
        let center_shader = envelope_center_real_to_complex_shader(
            request.channel_len,
            request.channel_count,
            self.precision,
        );
        let centered = self.fused_elementwise_with_telemetry_and_storage_exec(
            &center_shader,
            std::slice::from_ref(request.input),
            &channel_shape,
            complex_len,
            GpuTensorStorage::ComplexInterleaved,
        )?;

        let spectrum = match self.fft_dim_exec(&centered, None, 0).await {
            Ok(spectrum) => spectrum,
            Err(err) => {
                drop(centered);
                return Err(err);
            }
        };
        drop(centered);

        let mask_shader = envelope_analytic_mask_shader(
            request.channel_len,
            request.channel_count,
            self.precision,
        );
        let filtered = match self.fused_elementwise_with_telemetry_and_storage_exec(
            &mask_shader,
            std::slice::from_ref(&spectrum),
            &channel_shape,
            complex_len,
            GpuTensorStorage::ComplexInterleaved,
        ) {
            Ok(filtered) => filtered,
            Err(err) => {
                drop(spectrum);
                return Err(err.into());
            }
        };
        drop(spectrum);

        let analytic = match self.ifft_dim_exec(&filtered, None, 0).await {
            Ok(analytic) => analytic,
            Err(err) => {
                drop(filtered);
                return Err(err);
            }
        };
        drop(filtered);

        let real_len = request
            .channel_len
            .checked_mul(request.channel_count)
            .ok_or_else(|| anyhow!("signal_envelope: output too large"))?;
        let bounds_shader = envelope_analytic_bounds_shader(
            request.channel_len,
            request.channel_count,
            self.precision,
        );
        let inputs = [request.input.clone(), analytic.clone()];
        let operands = [
            IndexedFusionOperand {
                shape: request.input.shape(),
                lanes: real_len,
                storage: GpuTensorStorage::Real,
            },
            IndexedFusionOperand {
                shape: &channel_shape,
                lanes: complex_len,
                storage: GpuTensorStorage::ComplexInterleaved,
            },
        ];
        let mut outputs =
            match self.fused_indexed_multi_with_telemetry_exec(IndexedFusionMultiRequest {
                shader: &bounds_shader,
                inputs: &inputs,
                output_shape: request.output_shape,
                len: real_len,
                output_classes: &[self.provider_precision_exec().into(); 2],
                operands: &operands,
            }) {
                Ok(outputs) => outputs,
                Err(err) => {
                    drop(analytic);
                    return Err(err.into());
                }
            };
        drop(analytic);
        if outputs.len() != 2 {
            return Err(anyhow!("signal_envelope: missing outputs").into());
        }
        let lower = outputs.pop().expect("lower output");
        let upper = outputs.pop().expect("upper output");
        Ok(ProviderEnvelopeResult { upper, lower })
    }

    fn signal_envelope_analytic_fir_exec(
        &self,
        request: &ProviderEnvelopeRequest<'_>,
        filter_len: usize,
    ) -> Result<ProviderEnvelopeResult> {
        ensure!(filter_len > 0, "signal_envelope: invalid filter length");
        let real_len = request
            .channel_len
            .checked_mul(request.channel_count)
            .ok_or_else(|| anyhow!("signal_envelope: output too large"))?;
        let kernel_data = envelope_hilbert_fir_kernel(filter_len);
        let kernel_shape = [filter_len, 1usize];
        let kernel = self.upload_exec(&HostTensorView {
            data: &kernel_data,
            shape: &kernel_shape,
        })?;
        let shader = envelope_analytic_fir_bounds_shader(
            request.channel_len,
            request.channel_count,
            filter_len,
            self.precision,
        );
        let inputs = [request.input.clone(), kernel.clone()];
        let operands = [
            IndexedFusionOperand {
                shape: request.input.shape(),
                lanes: real_len,
                storage: GpuTensorStorage::Real,
            },
            IndexedFusionOperand {
                shape: &kernel_shape,
                lanes: filter_len,
                storage: GpuTensorStorage::Real,
            },
        ];
        let mut outputs =
            match self.fused_indexed_multi_with_telemetry_exec(IndexedFusionMultiRequest {
                shader: &shader,
                inputs: &inputs,
                output_shape: request.output_shape,
                len: real_len,
                output_classes: &[self.provider_precision_exec().into(); 2],
                operands: &operands,
            }) {
                Ok(outputs) => outputs,
                Err(err) => {
                    drop(kernel);
                    return Err(err);
                }
            };
        drop(kernel);
        ensure!(outputs.len() == 2, "signal_envelope: missing outputs");
        let lower = outputs.pop().expect("lower output");
        let upper = outputs.pop().expect("upper output");
        Ok(ProviderEnvelopeResult { upper, lower })
    }

    fn signal_envelope_rms_exec(
        &self,
        request: &ProviderEnvelopeRequest<'_>,
        window_len: usize,
    ) -> Result<ProviderEnvelopeResult> {
        ensure!(window_len > 0, "signal_envelope: invalid window length");
        let real_len = request
            .channel_len
            .checked_mul(request.channel_count)
            .ok_or_else(|| anyhow!("signal_envelope: output too large"))?;
        let shader = envelope_rms_bounds_shader(
            request.channel_len,
            request.channel_count,
            window_len,
            self.precision,
        );
        let operands = [IndexedFusionOperand {
            shape: request.input.shape(),
            lanes: real_len,
            storage: GpuTensorStorage::Real,
        }];
        let mut outputs =
            self.fused_indexed_multi_with_telemetry_exec(IndexedFusionMultiRequest {
                shader: &shader,
                inputs: std::slice::from_ref(request.input),
                output_shape: request.output_shape,
                len: real_len,
                output_classes: &[self.provider_precision_exec().into(); 2],
                operands: &operands,
            })?;
        ensure!(outputs.len() == 2, "signal_envelope: missing outputs");
        let lower = outputs.pop().expect("lower output");
        let upper = outputs.pop().expect("upper output");
        Ok(ProviderEnvelopeResult { upper, lower })
    }

    pub(crate) fn conv2d_exec(
        &self,
        _signal: &GpuTensorHandle,
        _kernel: &GpuTensorHandle,
        _mode: ProviderConvMode,
    ) -> Result<GpuTensorHandle> {
        // The capability snapshot and feasibility query both reject this
        // absent hook before any device work. Preserve the same typed
        // Unsupported result for direct legacy invocations so an unselected
        // runtime call may gather and use its existing host implementation.
        Err(runmat_accelerate_api::unsupported_provider_operation(
            "conv2d not implemented for the WGPU provider yet",
        ))
    }

    pub(crate) fn conv1d_exec(
        &self,
        signal: &GpuTensorHandle,
        kernel: &GpuTensorHandle,
        options: ProviderConv1dOptions,
    ) -> Result<GpuTensorHandle> {
        let operation = runmat_accelerate_api::ProviderOperationIdentity::new("conv");
        let observed = runmat_accelerate_api::observation::current_provider_observation()
            .filter(|recorder| recorder.expects_operation(&operation));
        let checkpoint = observed.as_ref().map(
            runmat_accelerate_api::observation::ProviderOperationRecorder::submission_checkpoint,
        );
        let entry_signal = self.get_entry(signal)?;
        let entry_kernel = self.get_entry(kernel)?;

        ensure!(
            entry_signal.precision == self.precision && entry_kernel.precision == self.precision,
            "conv1d: mixed precision tensors are not supported"
        );

        let signal_len = entry_signal.len;
        let kernel_len = entry_kernel.len;

        let (output_len, start_offset, _) = conv1d_window(signal_len, kernel_len, options.mode)?;

        if output_len == 0 {
            let out_shape = conv1d_output_shape(0, options.orientation);
            let out_buffer = self.with_emission(|emission, _| {
                self.create_storage_buffer(emission, 0, "runmat-conv1d-empty")
            })??;
            return self.register_existing_buffer(out_buffer, out_shape, 0);
        }

        ensure!(
            signal_len <= u32::MAX as usize
                && kernel_len <= u32::MAX as usize
                && output_len <= u32::MAX as usize
                && start_offset <= u32::MAX as usize,
            "conv1d: tensor exceeds GPU kernel limits"
        );

        let out_shape = conv1d_output_shape(output_len, options.orientation);
        let out_buffer = self.with_emission(|emission, _| {
            self.create_storage_buffer_checked(emission, output_len, "runmat-conv1d-out")
        })??;

        let params = Conv1dParams {
            signal_len: signal_len as u32,
            kernel_len: kernel_len as u32,
            output_len: output_len as u32,
            start_offset: start_offset as u32,
        };
        self.with_emission(|emission, pending| -> Result<()> {
            let params_buffer = self.uniform_buffer(emission, &params, "runmat-conv1d-params")?;

            let bind_group = emission
                .device()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-conv1d-bind"),
                    layout: &self.pipelines.conv1d.get(emission)?.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: entry_signal.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: entry_kernel.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: out_buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: params_buffer.as_entire_binding(),
                        },
                    ],
                });

            let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                output_len as u32,
                crate::backend::wgpu::config::WORKGROUP_SIZE,
            );
            crate::backend::wgpu::dispatch::conv::run(
                emission,
                pending,
                &self.pipelines.conv1d.get(emission)?.pipeline,
                &bind_group,
                workgroups,
            )?;
            Ok(())
        })??;

        let handle = self.register_existing_buffer(out_buffer, out_shape, output_len)?;
        if let (Some(recorder), Some(checkpoint)) = (observed, checkpoint) {
            recorder.confirm_submitted_provider_execution_after(
                self.device_id(),
                &operation,
                checkpoint,
            );
        }
        Ok(handle)
    }
    pub(crate) async fn iir_filter_exec<'a>(
        &self,
        coefficients: runmat_accelerate_api::ProviderIirCoefficientSource<'a>,
        x: &GpuTensorHandle,
        options: ProviderIirFilterOptions,
    ) -> Result<ProviderIirFilterResult> {
        let ProviderIirFilterOptions { dim, zi } = options;

        let entry_x = self.get_entry(x)?;
        ensure!(
            runmat_accelerate_api::handle_integer_type(x).is_none(),
            "iir_filter: integer buffers must be converted before floating provider dispatch"
        );
        ensure!(
            !runmat_accelerate_api::handle_is_logical(x),
            "iir_filter: logical buffers must be converted before floating provider dispatch"
        );
        let effective_storage = |handle: &GpuTensorHandle, entry_storage: GpuTensorStorage| match (
            runmat_accelerate_api::handle_storage(handle),
            entry_storage,
        ) {
            (GpuTensorStorage::ComplexInterleaved, _)
            | (_, GpuTensorStorage::ComplexInterleaved) => GpuTensorStorage::ComplexInterleaved,
            _ => GpuTensorStorage::Real,
        };
        let x_storage = effective_storage(x, entry_x.storage);
        ensure!(
            entry_x.precision == self.precision,
            "iir_filter: mixed precision tensors are not supported"
        );

        let (order, coefficient_plan) = match coefficients {
            runmat_accelerate_api::ProviderIirCoefficientSource::UnitDenominatorFir {
                numerator,
            } => {
                let entry_b = self.get_entry(numerator)?;
                ensure!(
                    entry_b.len > 0
                        && entry_b.precision == self.precision
                        && effective_storage(numerator, entry_b.storage) == GpuTensorStorage::Real
                        && runmat_accelerate_api::handle_integer_type(numerator).is_none()
                        && !runmat_accelerate_api::handle_is_logical(numerator),
                    "iir_filter: invalid resident FIR numerator"
                );
                (
                    entry_b.len,
                    IirCoefficientPlan::UnitDenominatorFir { numerator },
                )
            }
            runmat_accelerate_api::ProviderIirCoefficientSource::Prepared(prepared) => {
                let numerator = prepared.numerator();
                let denominator = prepared.denominator();
                let entry_b = self.get_entry(numerator)?;
                let entry_a = self.get_entry(denominator)?;
                let order = entry_b.len;
                ensure!(
                    order > 0
                        && entry_a.len == order
                        && entry_b.precision == self.precision
                        && entry_a.precision == self.precision
                        && effective_storage(numerator, entry_b.storage) == GpuTensorStorage::Real
                        && effective_storage(denominator, entry_a.storage)
                            == GpuTensorStorage::Real
                        && runmat_accelerate_api::handle_integer_type(numerator).is_none()
                        && runmat_accelerate_api::handle_integer_type(denominator).is_none()
                        && !runmat_accelerate_api::handle_is_logical(numerator)
                        && !runmat_accelerate_api::handle_is_logical(denominator),
                    "iir_filter: invalid prepared resident coefficients"
                );
                ensure!(
                    !numerator.shares_allocation_with(x) && !denominator.shares_allocation_with(x),
                    "iir_filter: prepared coefficients alias the signal"
                );
                (
                    order,
                    IirCoefficientPlan::PreparedResident {
                        numerator,
                        denominator,
                    },
                )
            }
            runmat_accelerate_api::ProviderIirCoefficientSource::Resident {
                numerator,
                denominator,
            } => {
                let entry_b = self.get_entry(numerator)?;
                let entry_a = self.get_entry(denominator)?;
                ensure!(
                    entry_b.len > 0
                        && entry_a.len > 0
                        && entry_b.precision == self.precision
                        && entry_a.precision == self.precision
                        && effective_storage(numerator, entry_b.storage) == GpuTensorStorage::Real
                        && effective_storage(denominator, entry_a.storage)
                            == GpuTensorStorage::Real
                        && runmat_accelerate_api::handle_integer_type(numerator).is_none()
                        && runmat_accelerate_api::handle_integer_type(denominator).is_none()
                        && !runmat_accelerate_api::handle_is_logical(numerator)
                        && !runmat_accelerate_api::handle_is_logical(denominator),
                    "iir_filter: invalid resident coefficients"
                );
                let nb = entry_b.len;
                let na = entry_a.len;
                let order = nb.max(na);
                let b_host = self.download_exec(numerator).await?;
                let a_host = self.download_exec(denominator).await?;
                let a0 = *a_host.data.first().ok_or_else(|| {
                    anyhow!("iir_filter: denominator coefficients cannot be empty")
                })?;
                ensure!(
                    a0 != 0.0,
                    "iir_filter: denominator coefficient a(1) must be non-zero"
                );

                let mut b_norm = vec![0.0f64; order];
                let mut a_norm = vec![0.0f64; order];
                for i in 0..order {
                    let b_coeff = if i < nb { b_host.data[i] } else { 0.0 };
                    b_norm[i] = b_coeff / a0;
                    if i == 0 {
                        a_norm[0] = 1.0;
                    } else {
                        let a_coeff = if i < na { a_host.data[i] } else { 0.0 };
                        a_norm[i] = a_coeff / a0;
                    }
                }
                (
                    order,
                    IirCoefficientPlan::Normalized {
                        b: b_norm.into(),
                        a: a_norm.into(),
                    },
                )
            }
        };
        ensure!(
            order <= u32::MAX as usize,
            "iir_filter: filter order exceeds GPU limits"
        );

        let state_len = order.saturating_sub(1);

        let mut shape_ext = entry_x.shape.clone();
        if dim >= shape_ext.len() {
            shape_ext.extend(std::iter::repeat_n(1, dim + 1 - shape_ext.len()));
        }
        ensure!(
            dim < shape_ext.len(),
            "iir_filter: dimension argument exceeds tensor rank"
        );
        let dim_idx = dim;
        let dim_len = shape_ext[dim_idx];
        let x_lane_factor = match x_storage {
            GpuTensorStorage::Real => 1usize,
            GpuTensorStorage::ComplexInterleaved => 2usize,
        };
        let signal_logical_len = product_checked(&shape_ext)
            .ok_or_else(|| anyhow!("iir_filter: tensor exceeds GPU limits"))?;
        let signal_raw_len = signal_logical_len
            .checked_mul(x_lane_factor)
            .ok_or_else(|| anyhow!("iir_filter: signal length overflow"))?;
        ensure!(
            entry_x.len == signal_raw_len,
            "iir_filter: signal raw length mismatch (shape implies {} logical elements, buffer has {}, lane factor {})",
            signal_logical_len,
            entry_x.len,
            x_lane_factor
        );

        let leading = if dim_idx == 0 {
            1usize
        } else {
            product_checked(&shape_ext[..dim_idx])
                .ok_or_else(|| anyhow!("iir_filter: tensor exceeds GPU limits"))?
        };
        let trailing = if dim_idx + 1 >= shape_ext.len() {
            1usize
        } else {
            product_checked(&shape_ext[dim_idx + 1..])
                .ok_or_else(|| anyhow!("iir_filter: tensor exceeds GPU limits"))?
        };
        let channel_count = leading
            .checked_mul(trailing)
            .ok_or_else(|| anyhow!("iir_filter: tensor exceeds GPU limits"))?;

        ensure!(
            shape_ext.len() <= crate::backend::wgpu::params::FILTER_MAX_RANK,
            "iir_filter: tensors exceed supported rank for GPU kernel"
        );

        let state_shape =
            runmat_accelerate_api::provider_iir_state_shape(&shape_ext, dim_idx, state_len)?;
        ensure!(
            state_shape.len() <= crate::backend::wgpu::params::FILTER_MAX_RANK,
            "iir_filter: filter state rank exceeds GPU limits"
        );

        let state_total = if state_len == 0 {
            0usize
        } else {
            product_checked(&state_shape)
                .ok_or_else(|| anyhow!("iir_filter: filter state exceeds GPU limits"))?
        };

        if let Some(ref zi_handle) = zi {
            let zi_entry = self.get_entry(zi_handle)?;
            ensure!(
                runmat_accelerate_api::handle_integer_type(zi_handle).is_none()
                    && !runmat_accelerate_api::handle_is_logical(zi_handle),
                "iir_filter: nonfloating initial conditions must be converted before provider dispatch"
            );
            ensure!(
                zi_entry.precision == self.precision,
                "iir_filter: initial conditions use incompatible precision"
            );
            let zi_storage = effective_storage(zi_handle, zi_entry.storage);
            ensure!(
                zi_storage == x_storage,
                "iir_filter: initial conditions storage must match the signal"
            );
            ensure!(
                shapes_compatible(&state_shape, &zi_entry.shape),
                "iir_filter: initial conditions are not compatible with the requested dimension"
            );
            let zi_dim = zi_entry.shape.first().copied().unwrap_or(1);
            ensure!(
                zi_dim == state_len,
                "iir_filter: initial conditions must have {} states along the leading state dimension",
                state_len
            );
            if state_total == 0 {
                ensure!(
                    zi_entry.len == 0,
                    "iir_filter: initial conditions have {} elements but zero were expected",
                    zi_entry.len
                );
            } else {
                let expected_zi_len = state_total
                    .checked_mul(x_lane_factor)
                    .ok_or_else(|| anyhow!("iir_filter: initial state length overflow"))?;
                ensure!(
                    zi_entry.len == expected_zi_len,
                    "iir_filter: initial state vector raw length mismatch (expected {}, found {})",
                    expected_zi_len,
                    zi_entry.len
                );
            }
        }

        ensure!(
            signal_logical_len <= u32::MAX as usize,
            "iir_filter: signal length exceeds GPU limits"
        );
        ensure!(
            entry_x.len <= u32::MAX as usize,
            "iir_filter: signal raw buffer length exceeds GPU limits"
        );
        ensure!(
            leading <= u32::MAX as usize
                && trailing <= u32::MAX as usize
                && channel_count <= u32::MAX as usize,
            "iir_filter: tensor exceeds GPU kernel limits"
        );
        ensure!(
            dim_len <= u32::MAX as usize,
            "iir_filter: dimension length exceeds GPU limits"
        );
        ensure!(
            state_len <= u32::MAX as usize,
            "iir_filter: filter order exceeds GPU limits"
        );
        ensure!(
            state_total <= u32::MAX as usize,
            "iir_filter: filter state size exceeds GPU limits"
        );

        let state_buffer_len = if state_len == 0 {
            0usize
        } else {
            state_len
                .checked_mul(channel_count)
                .and_then(|value| value.checked_mul(x_lane_factor))
                .ok_or_else(|| anyhow!("iir_filter: state buffer length overflow"))?
        };
        let state_total_raw = state_total
            .checked_mul(x_lane_factor)
            .ok_or_else(|| anyhow!("iir_filter: final state length overflow"))?;
        ensure!(
            state_buffer_len <= u32::MAX as usize,
            "iir_filter: state buffer length exceeds GPU limits"
        );
        ensure!(
            state_total_raw <= u32::MAX as usize,
            "iir_filter: filter state size exceeds GPU limits"
        );

        let result = (|| -> Result<ProviderIirFilterResult> {
            let (b_norm_handle, a_norm_handle) = match &coefficient_plan {
                IirCoefficientPlan::UnitDenominatorFir { numerator } => {
                    let a_norm_handle = self.zeros_exec(&[order, 1usize])?;
                    ((*numerator).clone(), a_norm_handle)
                }
                IirCoefficientPlan::PreparedResident {
                    numerator,
                    denominator,
                } => ((*numerator).clone(), (*denominator).clone()),
                IirCoefficientPlan::Normalized { b, a } => {
                    let b_shape = [order, 1usize];
                    let b_view = HostTensorView {
                        data: b,
                        shape: &b_shape,
                    };
                    let b_norm_handle = self.upload_exec(&b_view)?;

                    let a_shape = [order, 1usize];
                    let a_view = HostTensorView {
                        data: a,
                        shape: &a_shape,
                    };
                    let a_norm_handle = self.upload_exec(&a_view)?;
                    (b_norm_handle, a_norm_handle)
                }
            };

            let b_norm_entry = self.get_entry(&b_norm_handle)?;
            let a_norm_entry = self.get_entry(&a_norm_handle)?;

            let (out_buffer, states_buffer, final_state_buffer) =
                self.with_emission(|emission, _| -> Result<_> {
                    Ok((
                        self.create_storage_buffer(emission, entry_x.len, "runmat-iir-filter-out")?,
                        self.create_storage_buffer(
                            emission,
                            state_buffer_len,
                            "runmat-iir-filter-state",
                        )?,
                        self.create_storage_buffer(
                            emission,
                            state_total_raw,
                            "runmat-iir-filter-final",
                        )?,
                    ))
                })??;

            let (zi_buffer, zi_present_flag) = if let Some(ref zi_handle) = zi {
                let zi_entry = self.get_entry(zi_handle)?;
                (zi_entry.buffer, 1u32)
            } else {
                (
                    self.with_emission(|emission, _| {
                        self.create_storage_buffer(
                            emission,
                            state_total_raw,
                            "runmat-iir-filter-zi",
                        )
                    })??,
                    0u32,
                )
            };

            let mut signal_shape_arr = [crate::backend::wgpu::params::AlignedU32::new(0);
                crate::backend::wgpu::params::FILTER_MAX_RANK];
            for (idx, dim_len) in shape_ext.iter().enumerate() {
                signal_shape_arr[idx] =
                    crate::backend::wgpu::params::AlignedU32::new(*dim_len as u32);
            }
            let mut state_shape_arr = [crate::backend::wgpu::params::AlignedU32::new(0);
                crate::backend::wgpu::params::FILTER_MAX_RANK];
            for (idx, dim_len) in state_shape.iter().enumerate() {
                state_shape_arr[idx] =
                    crate::backend::wgpu::params::AlignedU32::new(*dim_len as u32);
            }

            let params = FilterParams {
                dim_len: dim_len as u32,
                leading: leading as u32,
                trailing: trailing as u32,
                order: order as u32,
                state_len: state_len as u32,
                signal_len: signal_logical_len as u32,
                channel_count: channel_count as u32,
                zi_present: zi_present_flag,
                dim_idx: dim_idx as u32,
                rank: shape_ext.len() as u32,
                state_rank: state_shape.len() as u32,
                lane_factor: x_lane_factor as u32,
                signal_shape: signal_shape_arr,
                state_shape: state_shape_arr,
            };

            self.with_emission(|emission, pending| -> Result<()> {
                let params_buffer =
                    self.uniform_buffer(emission, &params, "runmat-iir-filter-params")?;

                let bind_group = emission
                    .device()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-iir-filter-bind"),
                        layout: &self.pipelines.filter.get(emission)?.layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: entry_x.buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: b_norm_entry.buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: a_norm_entry.buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: zi_buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: out_buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: states_buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 6,
                                resource: final_state_buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 7,
                                resource: params_buffer.as_entire_binding(),
                            },
                        ],
                    });

                let workgroups = self
                    .pipelines
                    .filter
                    .linear_dispatch_groups(channel_count)?;
                crate::backend::wgpu::dispatch::filter::run(
                    emission,
                    pending,
                    &self.pipelines.filter.get(emission)?.pipeline,
                    &bind_group,
                    workgroups,
                )?;
                Ok(())
            })??;

            let output_handle = self.register_existing_buffer_with_storage(
                out_buffer,
                entry_x.shape.clone(),
                entry_x.len,
                x_storage,
            )?;
            let final_state_handle = self.register_existing_buffer_with_storage(
                final_state_buffer,
                state_shape.clone(),
                state_total_raw,
                x_storage,
            )?;

            Ok(ProviderIirFilterResult {
                output: output_handle,
                final_state: Some(final_state_handle),
            })
        })();

        result
    }

    pub(crate) fn moving_window_exec(
        &self,
        request: &ProviderMovingWindowRequest<'_>,
    ) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(request.input)?;
        let input_storage = match (
            runmat_accelerate_api::handle_storage(request.input),
            entry.storage,
        ) {
            (GpuTensorStorage::ComplexInterleaved, _)
            | (_, GpuTensorStorage::ComplexInterleaved) => GpuTensorStorage::ComplexInterleaved,
            _ => GpuTensorStorage::Real,
        };
        ensure!(
            input_storage == GpuTensorStorage::Real,
            "moving_window: complex tensors are not supported"
        );
        ensure!(
            entry.precision == self.precision,
            "moving_window: mixed precision tensors are not supported"
        );
        let mut input_shape = entry.shape.clone();
        if request.dim >= input_shape.len() {
            input_shape.extend(std::iter::repeat_n(1, request.dim + 1 - input_shape.len()));
        }
        ensure!(
            request.dim < input_shape.len() && request.dim < request.output_shape.len(),
            "moving_window: dimension exceeds tensor rank"
        );
        let input_len = product_checked(&input_shape)
            .ok_or_else(|| anyhow!("moving_window: input tensor exceeds GPU limits"))?;
        ensure!(
            input_len == entry.len,
            "moving_window: input shape mismatch (expected {} elements, got {})",
            input_len,
            entry.len
        );
        let output_len = product_checked(request.output_shape)
            .ok_or_else(|| anyhow!("moving_window: output tensor exceeds GPU limits"))?;
        let axis_len = input_shape[request.dim];
        let out_axis_len = request.output_shape[request.dim];
        let stride_before = if request.dim == 0 {
            1usize
        } else {
            product_checked(&input_shape[..request.dim])
                .ok_or_else(|| anyhow!("moving_window: stride computation overflow"))?
                .max(1)
        };
        let stride_after = if request.dim + 1 >= input_shape.len() {
            1usize
        } else {
            product_checked(&input_shape[request.dim + 1..])
                .ok_or_else(|| anyhow!("moving_window: stride computation overflow"))?
                .max(1)
        };

        ensure!(
            output_len <= u32::MAX as usize
                && axis_len <= u32::MAX as usize
                && out_axis_len <= u32::MAX as usize
                && stride_before <= u32::MAX as usize
                && stride_after <= u32::MAX as usize
                && request.before <= u32::MAX as usize
                && request.after <= u32::MAX as usize,
            "moving_window: tensor exceeds GPU kernel limits"
        );
        let max_center = if matches!(request.endpoints, ProviderMovingWindowEndpoints::Discard) {
            out_axis_len
                .saturating_sub(1)
                .checked_add(request.before)
                .ok_or_else(|| anyhow!("moving_window: center index exceeds GPU kernel limits"))?
        } else {
            out_axis_len.saturating_sub(1)
        };
        let max_end = max_center
            .checked_add(request.after)
            .ok_or_else(|| anyhow!("moving_window: endpoint index exceeds GPU kernel limits"))?;
        ensure!(
            axis_len <= WGPU_MOVING_WINDOW_I32_LIMIT
                && request.before <= WGPU_MOVING_WINDOW_I32_LIMIT
                && request.after <= WGPU_MOVING_WINDOW_I32_LIMIT
                && max_center <= WGPU_MOVING_WINDOW_I32_LIMIT
                && max_end <= WGPU_MOVING_WINDOW_I32_LIMIT,
            "moving_window: tensor exceeds signed GPU kernel limits"
        );
        if matches!(request.op, ProviderMovingWindowOp::Median) {
            let window_len = request
                .before
                .checked_add(request.after)
                .and_then(|sum| sum.checked_add(1))
                .ok_or_else(|| anyhow!("moving_window: median window exceeds GPU kernel limits"))?;
            ensure!(
                window_len <= WGPU_MOVING_MEDIAN_MAX_WINDOW,
                "moving_window: median windows above {WGPU_MOVING_MEDIAN_MAX_WINDOW} elements use host fallback"
            );
        }

        if output_len != 0 {
            self.require_ieee_words(entry.precision)?;
        }
        let out_shape = request.output_shape.to_vec();
        if output_len == 0 {
            let out_buffer = self.with_emission(|emission, _| {
                self.create_storage_buffer_checked(emission, output_len, "runmat-moving-window-out")
            })??;
            return self.register_existing_buffer(out_buffer, out_shape, 0);
        }

        let op = moving_window_op_code(request.op);
        let (endpoint, fill_value) = moving_window_endpoint_code(request.endpoints);
        let nan_mode = match request.nan_mode {
            ProviderNanMode::Include => 0,
            ProviderNanMode::Omit => 1,
        };
        let normalization = match request.normalization {
            ProviderStdNormalization::Sample => 0,
            ProviderStdNormalization::Population => 1,
        };
        let shader = build_specialized_moving_window_shader(&MovingWindowShaderSpec {
            precision: self.precision,
            output_len,
            axis_len,
            out_axis_len,
            stride_before,
            before: request.before,
            after: request.after,
            op: request.op,
            endpoints: request.endpoints,
            nan_mode: request.nan_mode,
            normalization: request.normalization,
            fill_value,
        });
        let prepared_shader = crate::backend::wgpu::shader_compilation::prepare(
            &shader,
            self.access.enabled_features(),
            None,
        )?;
        let out_buffer = self.with_emission(|emission, _| {
            self.create_storage_buffer_checked(emission, output_len, "runmat-moving-window-out")
        })??;
        self.with_emission(|emission, pending| -> Result<()> {
            let shader_module = crate::backend::wgpu::shader_compilation::create_prepared(
                emission,
                "runmat-moving-window-specialized-shader",
                &prepared_shader,
            )?;
            let pipeline_layout =
                emission
                    .device()
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("runmat-moving-window-specialized-layout"),
                        bind_group_layouts: &[&self.pipelines.moving_window.get(emission)?.layout],
                        push_constant_ranges: &[],
                    });
            let pipeline = emission.device().create_compute_pipeline(
                &crate::backend::wgpu::compat::wgpu_compute_pipeline_descriptor! {
                    label: Some("runmat-moving-window-specialized-pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader_module,
                    entry_point: "main",
                },
            );

            let bind_group = match self.precision {
                NumericPrecision::F64 => {
                    let params = MovingWindowParamsF64 {
                        meta0: PackedU32([
                            output_len as u32,
                            axis_len as u32,
                            out_axis_len as u32,
                            stride_before as u32,
                        ]),
                        meta1: PackedU32([
                            endpoint,
                            nan_mode,
                            request.before as u32,
                            request.after as u32,
                        ]),
                        meta2: PackedU32([op, normalization, stride_after as u32, 0]),
                        fill_value,
                        _pad1: 0.0,
                    };
                    let params_buffer =
                        self.uniform_buffer(emission, &params, "runmat-moving-window-params")?;
                    emission
                        .device()
                        .create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("runmat-moving-window-bind"),
                            layout: &self.pipelines.moving_window.get(emission)?.layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: entry.buffer.as_ref().as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: out_buffer.as_ref().as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: params_buffer.as_entire_binding(),
                                },
                            ],
                        })
                }
                NumericPrecision::F32 => {
                    let params = MovingWindowParamsF32 {
                        meta0: PackedU32([
                            output_len as u32,
                            axis_len as u32,
                            out_axis_len as u32,
                            stride_before as u32,
                        ]),
                        meta1: PackedU32([
                            endpoint,
                            nan_mode,
                            request.before as u32,
                            request.after as u32,
                        ]),
                        meta2: PackedU32([op, normalization, stride_after as u32, 0]),
                        meta3: PackedF32([fill_value as f32, 0.0, 0.0, 0.0]),
                    };
                    let params_buffer =
                        self.uniform_buffer(emission, &params, "runmat-moving-window-params")?;
                    emission
                        .device()
                        .create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("runmat-moving-window-bind"),
                            layout: &self.pipelines.moving_window.get(emission)?.layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: entry.buffer.as_ref().as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: out_buffer.as_ref().as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: params_buffer.as_entire_binding(),
                                },
                            ],
                        })
                }
            };

            let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                output_len as u32,
                crate::backend::wgpu::config::WORKGROUP_SIZE,
            );
            crate::backend::wgpu::dispatch::diff::run(
                emission,
                pending,
                &pipeline,
                &bind_group,
                workgroups,
            )?;
            Ok(())
        })??;

        self.register_existing_buffer(out_buffer, out_shape, output_len)
    }

    pub(crate) fn diff_once_exec(
        &self,
        handle: &GpuTensorHandle,
        dim: usize,
    ) -> Result<GpuTensorHandle> {
        ensure!(
            runmat_accelerate_api::handle_integer_type(handle).is_none(),
            "diff: typed-integer buffers require the exact typed fallback"
        );
        let entry = self.get_entry(handle)?;

        let mut ext_shape = if entry.shape.is_empty() {
            vec![if entry.len == 0 { 1 } else { entry.len }]
        } else {
            entry.shape.clone()
        };
        while ext_shape.len() <= dim {
            ext_shape.push(1);
        }

        let len_dim = ext_shape[dim];

        let mut out_shape = entry.shape.clone();
        while out_shape.len() <= dim {
            out_shape.push(1);
        }

        if len_dim <= 1 || entry.len == 0 {
            out_shape[dim] = out_shape[dim].saturating_sub(1);
            let out_len = product_checked(&out_shape).unwrap_or(0);
            let out_buffer = self.with_emission(|emission, _| {
                self.create_storage_buffer(emission, out_len, "runmat-diff-empty")
            })??;
            return self.register_existing_buffer(out_buffer, out_shape, out_len);
        }

        let stride_before = if dim == 0 {
            1usize
        } else {
            product_checked(&ext_shape[..dim])
                .ok_or_else(|| anyhow!("diff: stride computation overflow"))?
                .max(1)
        };
        let stride_after = if dim + 1 >= ext_shape.len() {
            1usize
        } else {
            product_checked(&ext_shape[dim + 1..])
                .ok_or_else(|| anyhow!("diff: stride computation overflow"))?
                .max(1)
        };

        let expected_len = stride_before
            .checked_mul(len_dim)
            .and_then(|v| v.checked_mul(stride_after))
            .ok_or_else(|| anyhow!("diff: tensor size exceeds GPU limits"))?;
        ensure!(
            expected_len == entry.len,
            "diff: tensor shape mismatch (expected {} elements, got {})",
            expected_len,
            entry.len
        );

        let segment_out = len_dim - 1;
        let segments = stride_before
            .checked_mul(stride_after)
            .ok_or_else(|| anyhow!("diff: segment count exceeds GPU limits"))?;
        let out_len = segments
            .checked_mul(segment_out)
            .ok_or_else(|| anyhow!("diff: output size exceeds GPU limits"))?;
        let block = stride_before
            .checked_mul(len_dim)
            .ok_or_else(|| anyhow!("diff: block size exceeds GPU limits"))?;

        ensure!(
            len_dim <= u32::MAX as usize
                && stride_before <= u32::MAX as usize
                && stride_after <= u32::MAX as usize
                && segments <= u32::MAX as usize
                && block <= u32::MAX as usize
                && out_len <= u32::MAX as usize
                && entry.len <= u32::MAX as usize,
            "diff: tensor exceeds GPU kernel limits"
        );

        let out_buffer = self.with_emission(|emission, _| {
            self.create_storage_buffer(emission, out_len, "runmat-diff-out")
        })??;
        out_shape[dim] = len_dim - 1;

        let params = DiffParams {
            stride_before: stride_before as u32,
            segments: segments as u32,
            segment_len: len_dim as u32,
            segment_out: segment_out as u32,
            block: block as u32,
            total_out: out_len as u32,
            total_in: entry.len as u32,
            _pad: 0,
        };
        self.with_emission(|emission, pending| -> Result<()> {
            let params_buffer = self.uniform_buffer(emission, &params, "runmat-diff-params")?;

            let bind_group = emission
                .device()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-diff-bind"),
                    layout: &self.pipelines.diff.get(emission)?.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: entry.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: out_buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: params_buffer.as_entire_binding(),
                        },
                    ],
                });

            let workgroups = self.pipelines.diff.linear_dispatch_groups(out_len)?;
            crate::backend::wgpu::dispatch::diff::run(
                emission,
                pending,
                &self.pipelines.diff.get(emission)?.pipeline,
                &bind_group,
                workgroups,
            )?;
            Ok(())
        })??;

        self.register_existing_buffer(out_buffer, out_shape, out_len)
    }

    pub(crate) fn diff_exec(
        &self,
        handle: &GpuTensorHandle,
        dim: usize,
        order: usize,
    ) -> Result<GpuTensorHandle> {
        if order == 0 {
            return Ok(handle.clone());
        }

        let mut current = handle.clone();
        let mut owns_current = false;
        for step in 0..order {
            let next = self.diff_once_exec(&current, dim)?;
            if owns_current {
                drop(current);
            }
            current = next;
            owns_current = true;

            let entry = self.get_entry(&current)?;
            if entry.len == 0 {
                // Other empty axes can exhaust storage before this axis has
                // exhausted its geometry. Finish the remaining orders with
                // the canonical zero-copy view owner, not more dispatches.
                let mut shape = current.shape().to_vec();
                shape[dim] = shape[dim].saturating_sub(order - step - 1);
                return self.reshape_exec(&current, &shape);
            }
        }
        Ok(current)
    }

    pub(crate) fn gradient_exec(
        &self,
        handle: &GpuTensorHandle,
        dim: usize,
        spacing: f64,
    ) -> Result<GpuTensorHandle> {
        self.gradient_exec_with_spacing(handle, dim, GradientSpacingInput::Scalar(spacing))
    }

    pub(crate) fn gradient_exec_with_coordinates(
        &self,
        handle: &GpuTensorHandle,
        dim: usize,
        coordinates: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        self.gradient_exec_with_spacing(handle, dim, GradientSpacingInput::Coordinates(coordinates))
    }

    fn gradient_exec_with_spacing(
        &self,
        handle: &GpuTensorHandle,
        dim: usize,
        spacing: GradientSpacingInput<'_>,
    ) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(handle)?;
        let complex_storage = entry.storage == GpuTensorStorage::ComplexInterleaved
            || runmat_accelerate_api::handle_storage(handle)
                == GpuTensorStorage::ComplexInterleaved;
        ensure!(
            !complex_storage || entry.len % 2 == 0,
            "gradient: complex-interleaved buffers must have even length"
        );
        let logical_len = if complex_storage {
            entry.len / 2
        } else {
            entry.len
        };

        let mut ext_shape = normalize_gradient_shape(&entry.shape, logical_len);
        if ext_shape.is_empty() {
            ext_shape = vec![0, 0];
        }
        while ext_shape.len() <= dim {
            ext_shape.push(1);
        }

        let len_dim = ext_shape[dim];
        let coordinate_entry = match spacing {
            GradientSpacingInput::Coordinates(coordinates) => {
                let coordinate_entry = self.get_entry(coordinates)?;
                ensure!(
                    coordinate_entry.storage == GpuTensorStorage::Real
                        && runmat_accelerate_api::handle_storage(coordinates)
                            == GpuTensorStorage::Real,
                    "gradient: coordinate-vector spacing must be real"
                );
                ensure!(
                    coordinate_entry.len == len_dim,
                    "gradient: coordinate-vector spacing length {} does not match dimension length {}",
                    coordinate_entry.len,
                    len_dim
                );
                Some(coordinate_entry)
            }
            GradientSpacingInput::Scalar(_) => None,
        };
        let mut out_shape = normalize_gradient_shape(&entry.shape, logical_len);
        if out_shape.is_empty() {
            out_shape = vec![0, 0];
        }
        while out_shape.len() <= dim {
            out_shape.push(1);
        }

        let out_buffer = self.with_emission(|emission, _| {
            self.create_storage_buffer(emission, entry.len, "runmat-gradient-out")
        })??;
        if entry.len == 0 {
            return self.register_existing_buffer_with_storage(
                out_buffer,
                out_shape,
                0,
                if complex_storage {
                    GpuTensorStorage::ComplexInterleaved
                } else {
                    GpuTensorStorage::Real
                },
            );
        }

        let stride_before = if dim == 0 {
            1usize
        } else {
            product_checked(&ext_shape[..dim])
                .ok_or_else(|| anyhow!("gradient: stride computation overflow"))?
                .max(1)
        };
        let stride_after = if dim + 1 >= ext_shape.len() {
            1usize
        } else {
            product_checked(&ext_shape[dim + 1..])
                .ok_or_else(|| anyhow!("gradient: stride computation overflow"))?
                .max(1)
        };

        let expected_len = stride_before
            .checked_mul(len_dim.max(1))
            .and_then(|v| v.checked_mul(stride_after))
            .ok_or_else(|| anyhow!("gradient: tensor size exceeds GPU limits"))?;
        ensure!(
            expected_len == logical_len,
            "gradient: tensor shape mismatch (expected {} elements, got {})",
            expected_len,
            logical_len
        );

        let block = stride_before
            .checked_mul(len_dim.max(1))
            .ok_or_else(|| anyhow!("gradient: block size exceeds GPU limits"))?;
        let lane_multiplier = if complex_storage { 2usize } else { 1usize };
        let kernel_stride_before = stride_before
            .checked_mul(lane_multiplier)
            .ok_or_else(|| anyhow!("gradient: stride computation overflow"))?;
        let kernel_block = block
            .checked_mul(lane_multiplier)
            .ok_or_else(|| anyhow!("gradient: block size exceeds GPU limits"))?;
        ensure!(
            len_dim <= u32::MAX as usize
                && kernel_stride_before <= u32::MAX as usize
                && kernel_block <= u32::MAX as usize
                && entry.len <= u32::MAX as usize,
            "gradient: tensor exceeds GPU kernel limits"
        );

        self.with_emission(|emission, pending| -> Result<()> {
            let params_buffer = match self.precision {
                NumericPrecision::F64 => self.uniform_buffer(
                    emission,
                    &GradientParamsF64 {
                        stride_before: kernel_stride_before as u32,
                        segment_len: len_dim as u32,
                        block: kernel_block as u32,
                        total: entry.len as u32,
                        spacing: match spacing {
                            GradientSpacingInput::Scalar(spacing) => spacing,
                            GradientSpacingInput::Coordinates(_) => 0.0,
                        },
                        _pad0: 0.0,
                        _pad1: 0.0,
                        _pad2: 0.0,
                    },
                    "runmat-gradient-params",
                )?,
                NumericPrecision::F32 => self.uniform_buffer(
                    emission,
                    &GradientParamsF32 {
                        meta0: crate::backend::wgpu::params::PackedU32([
                            kernel_stride_before as u32,
                            len_dim as u32,
                            kernel_block as u32,
                            entry.len as u32,
                        ]),
                        meta1: crate::backend::wgpu::params::PackedF32([
                            match spacing {
                                GradientSpacingInput::Scalar(spacing) => spacing as f32,
                                GradientSpacingInput::Coordinates(_) => 0.0,
                            },
                            0.0,
                            0.0,
                            0.0,
                        ]),
                    },
                    "runmat-gradient-params",
                )?,
            };

            let (bind_group, pipeline) = if let Some(coordinate_entry) = coordinate_entry.as_ref() {
                (
                    emission
                        .device()
                        .create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("runmat-gradient-coordinates-bind"),
                            layout: &self.pipelines.gradient_coordinates.get(emission)?.layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: entry.buffer.as_ref().as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: out_buffer.as_ref().as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: params_buffer.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 3,
                                    resource: coordinate_entry.buffer.as_ref().as_entire_binding(),
                                },
                            ],
                        }),
                    &self.pipelines.gradient_coordinates.get(emission)?.pipeline,
                )
            } else {
                (
                    emission
                        .device()
                        .create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("runmat-gradient-bind"),
                            layout: &self.pipelines.gradient.get(emission)?.layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: entry.buffer.as_ref().as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: out_buffer.as_ref().as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: params_buffer.as_entire_binding(),
                                },
                            ],
                        }),
                    &self.pipelines.gradient.get(emission)?.pipeline,
                )
            };

            let workgroups = if coordinate_entry.is_some() {
                self.pipelines
                    .gradient_coordinates
                    .linear_dispatch_groups(entry.len)?
            } else {
                self.pipelines.gradient.linear_dispatch_groups(entry.len)?
            };
            crate::backend::wgpu::dispatch::gradient::run(
                emission,
                pending,
                pipeline,
                &bind_group,
                workgroups,
            )?;
            Ok(())
        })??;

        self.register_existing_buffer_with_storage(
            out_buffer,
            out_shape,
            entry.len,
            if complex_storage {
                GpuTensorStorage::ComplexInterleaved
            } else {
                GpuTensorStorage::Real
            },
        )
    }

    pub(crate) fn cumsum_exec(
        &self,
        handle: &GpuTensorHandle,
        dim: usize,
        direction: ProviderScanDirection,
        nan_mode: ProviderNanMode,
    ) -> Result<GpuTensorHandle> {
        let selected = self.get_entry(handle)?;
        if selected.len != 0 {
            self.require_ieee_words(selected.precision)?;
        }
        // For reverse scans, compute as flip → forward-scan → flip to preserve exact semantics
        if matches!(direction, ProviderScanDirection::Reverse) {
            let flipped_in = self.flip_exec(handle, &[dim])?;
            let forward =
                self.cumsum_exec(&flipped_in, dim, ProviderScanDirection::Forward, nan_mode)?;
            drop(flipped_in);
            let flipped_out = self.flip_exec(&forward, &[dim])?;
            drop(forward);
            return Ok(flipped_out);
        }
        let entry = self.get_entry(handle)?;
        if entry.len == 0 {
            return Ok(handle.clone());
        }

        let work_shape = if entry.shape.is_empty() {
            vec![entry.len]
        } else {
            entry.shape.clone()
        };

        if dim >= work_shape.len() {
            return Err(anyhow!(
                "cumsum: dimension {} exceeds tensor rank {}",
                dim + 1,
                work_shape.len()
            ));
        }

        let segment_len = work_shape[dim];
        if segment_len == 0 {
            let out_buffer = self.with_emission(|emission, _| {
                self.create_storage_buffer(emission, 0, "runmat-cumsum-empty")
            })??;
            return self.register_existing_buffer(out_buffer, entry.shape, 0);
        }

        let stride_before = if dim == 0 {
            1usize
        } else {
            work_shape[..dim].iter().copied().product::<usize>().max(1)
        };
        let stride_after = if dim + 1 >= work_shape.len() {
            1usize
        } else {
            work_shape[dim + 1..]
                .iter()
                .copied()
                .product::<usize>()
                .max(1)
        };

        let segments = stride_before
            .checked_mul(stride_after)
            .ok_or_else(|| anyhow!("cumsum: segment count exceeds GPU limits"))?;
        let block = stride_before
            .checked_mul(segment_len)
            .ok_or_else(|| anyhow!("cumsum: segment stride exceeds GPU limits"))?;

        ensure!(
            segment_len <= u32::MAX as usize,
            "cumsum: dimension length exceeds GPU kernel limits"
        );
        ensure!(
            stride_before <= u32::MAX as usize,
            "cumsum: stride_before exceeds GPU kernel limits"
        );
        ensure!(
            segments <= u32::MAX as usize,
            "cumsum: segment count exceeds GPU limits"
        );
        ensure!(
            block <= u32::MAX as usize,
            "cumsum: block size exceeds GPU limits"
        );
        ensure!(
            entry.len <= u32::MAX as usize,
            "cumsum: tensor too large for GPU kernel"
        );

        let out_buffer = self.with_emission(|emission, _| {
            self.create_storage_buffer(emission, entry.len, "runmat-cumsum-out")
        })??;
        if entry.len == 0 {
            return self.register_existing_buffer(out_buffer, entry.shape, 0);
        }

        let mut flags = 0u32;
        if matches!(nan_mode, ProviderNanMode::Omit) {
            flags |= 2;
        }

        let params = CumsumParams {
            segment_len: segment_len as u32,
            segments: segments as u32,
            stride_before: stride_before as u32,
            block: block as u32,
            flags,
            total_len: entry.len as u32,
            _pad0: 0,
            _pad1: 0,
        };
        self.with_emission(|emission, pending| -> Result<()> {
            let params_buffer = self.uniform_buffer(emission, &params, "runmat-cumsum-params")?;
            let bind_group = emission
                .device()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-cumsum-bind"),
                    layout: &self.pipelines.cumsum.get(emission)?.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: entry.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: out_buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: params_buffer.as_entire_binding(),
                        },
                    ],
                });
            let groups = self.pipelines.cumsum.linear_dispatch_groups(segments)?;
            crate::backend::wgpu::dispatch::scan::run(
                emission,
                pending,
                &self.pipelines.cumsum.get(emission)?.pipeline,
                &bind_group,
                groups,
                "runmat-cumsum-encoder",
                "runmat-cumsum-pass",
            )?;
            Ok(())
        })??;
        self.register_existing_buffer(out_buffer, entry.shape, entry.len)
    }

    pub(crate) fn trapz_exec(
        &self,
        handle: &GpuTensorHandle,
        dim: usize,
        spacing: ProviderTrapezoidSpacing<'_>,
    ) -> Result<GpuTensorHandle> {
        self.trapezoid_exec(handle, dim, spacing, false)
    }

    pub(crate) fn cumtrapz_exec(
        &self,
        handle: &GpuTensorHandle,
        dim: usize,
        spacing: ProviderTrapezoidSpacing<'_>,
    ) -> Result<GpuTensorHandle> {
        self.trapezoid_exec(handle, dim, spacing, true)
    }

    fn trapezoid_exec(
        &self,
        handle: &GpuTensorHandle,
        dim: usize,
        spacing: ProviderTrapezoidSpacing<'_>,
        cumulative: bool,
    ) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(handle)?;
        ensure!(
            entry.precision == self.precision,
            "trapezoid: mixed precision tensors are not supported"
        );
        let handle_storage = runmat_accelerate_api::handle_storage(handle);
        ensure!(
            entry.storage == handle_storage,
            "trapezoid: input storage metadata mismatch"
        );
        let lane_factor = match entry.storage {
            GpuTensorStorage::Real => 1usize,
            GpuTensorStorage::ComplexInterleaved => 2usize,
        };

        let mut work_shape = if entry.shape.is_empty() {
            ensure!(
                entry.len % lane_factor == 0,
                "trapezoid: input raw length is not divisible by lane factor"
            );
            vec![entry.len / lane_factor]
        } else {
            entry.shape.clone()
        };
        let stored_rank = work_shape.len();
        let virtual_axis = dim >= stored_rank;
        let dim = dim.min(stored_rank);
        if virtual_axis {
            work_shape.push(1);
        }
        let logical_len = work_shape
            .iter()
            .try_fold(1usize, |acc, &value| acc.checked_mul(value))
            .ok_or_else(|| anyhow!("trapezoid: input too large"))?;
        let expected_raw_len = logical_len
            .checked_mul(lane_factor)
            .ok_or_else(|| anyhow!("trapezoid: input length overflow"))?;
        ensure!(
            entry.len == expected_raw_len,
            "trapezoid: input raw length mismatch (shape implies {} logical elements, buffer has {}, lane factor {})",
            logical_len,
            entry.len,
            lane_factor
        );

        let segment_len = work_shape[dim];
        let stride_before = if dim == 0 {
            1usize
        } else {
            work_shape[..dim].iter().copied().product::<usize>().max(1)
        };
        let stride_after = if dim + 1 >= work_shape.len() {
            1usize
        } else {
            work_shape[dim + 1..]
                .iter()
                .copied()
                .product::<usize>()
                .max(1)
        };
        let segments = stride_before
            .checked_mul(stride_after)
            .ok_or_else(|| anyhow!("trapezoid: segment count exceeds GPU limits"))?;
        let block = stride_before
            .checked_mul(segment_len)
            .ok_or_else(|| anyhow!("trapezoid: segment stride exceeds GPU limits"))?;
        let mut output_shape = work_shape.clone();
        if !cumulative {
            output_shape[dim] = 1;
        }
        if virtual_axis {
            output_shape.truncate(stored_rank);
        }
        let output_len = output_shape
            .iter()
            .try_fold(1usize, |acc, &value| acc.checked_mul(value))
            .ok_or_else(|| anyhow!("trapezoid: output too large"))?;
        let output_raw_len = output_len
            .checked_mul(lane_factor)
            .ok_or_else(|| anyhow!("trapezoid: output length overflow"))?;

        ensure!(
            logical_len <= u32::MAX as usize
                && entry.len <= u32::MAX as usize
                && segment_len <= u32::MAX as usize
                && stride_before <= u32::MAX as usize
                && segments <= u32::MAX as usize
                && block <= u32::MAX as usize
                && output_len <= u32::MAX as usize
                && output_raw_len <= u32::MAX as usize
                && lane_factor <= u32::MAX as usize,
            "trapezoid: tensor too large for GPU kernel"
        );

        let (spacing_kind, scalar, spacing_buffer) = match spacing {
            ProviderTrapezoidSpacing::Unit => (0u32, 1.0, entry.buffer.clone()),
            ProviderTrapezoidSpacing::Scalar(value) => (1u32, value, entry.buffer.clone()),
            ProviderTrapezoidSpacing::ScalarHandle(spacing_handle) => {
                let spacing_entry = self.get_entry(spacing_handle)?;
                ensure!(
                    spacing_entry.storage == GpuTensorStorage::Real
                        && runmat_accelerate_api::handle_storage(spacing_handle)
                            == GpuTensorStorage::Real,
                    "trapezoid: spacing tensor must be real"
                );
                ensure!(spacing_entry.len >= 1, "trapezoid: scalar spacing is empty");
                (2u32, 0.0, spacing_entry.buffer)
            }
            ProviderTrapezoidSpacing::Vector(spacing_handle) => {
                let spacing_entry = self.get_entry(spacing_handle)?;
                ensure!(
                    spacing_entry.storage == GpuTensorStorage::Real
                        && runmat_accelerate_api::handle_storage(spacing_handle)
                            == GpuTensorStorage::Real,
                    "trapezoid: spacing vector must be real"
                );
                ensure!(
                    spacing_entry.len >= segment_len,
                    "trapezoid: spacing vector is shorter than integration dimension"
                );
                (3u32, 0.0, spacing_entry.buffer)
            }
            ProviderTrapezoidSpacing::Tensor(spacing_handle) => {
                let spacing_entry = self.get_entry(spacing_handle)?;
                ensure!(
                    spacing_entry.storage == GpuTensorStorage::Real
                        && runmat_accelerate_api::handle_storage(spacing_handle)
                            == GpuTensorStorage::Real,
                    "trapezoid: spacing tensor must be real"
                );
                ensure!(
                    spacing_entry.len >= logical_len,
                    "trapezoid: spacing tensor is smaller than input"
                );
                (4u32, 0.0, spacing_entry.buffer)
            }
        };

        let out_buffer = self.with_emission(|emission, _| {
            self.create_storage_buffer(emission, output_raw_len, "runmat-trapezoid-out")
        })??;
        self.with_emission(|emission, pending| -> Result<()> {
            let params_buffer = match self.precision {
                NumericPrecision::F64 => {
                    let params = TrapezoidParamsF64 {
                        segment_len: segment_len as u32,
                        segments: segments as u32,
                        stride_before: stride_before as u32,
                        block: block as u32,
                        total_len: logical_len as u32,
                        output_len: output_len as u32,
                        spacing_kind,
                        mode: if cumulative { 1 } else { 0 },
                        lane_factor: lane_factor as u32,
                        _pad_u32_0: 0,
                        _pad_u32_1: 0,
                        _pad_u32_2: 0,
                        spacing_scalar: scalar,
                        _pad0: 0.0,
                        _pad1: 0.0,
                        _pad2: 0.0,
                    };
                    self.uniform_buffer(emission, &params, "runmat-trapezoid-params")?
                }
                NumericPrecision::F32 => {
                    let params = TrapezoidParamsF32 {
                        meta0: PackedU32([
                            segment_len as u32,
                            segments as u32,
                            stride_before as u32,
                            block as u32,
                        ]),
                        meta1: PackedU32([
                            logical_len as u32,
                            output_len as u32,
                            spacing_kind,
                            if cumulative { 1 } else { 0 },
                        ]),
                        meta2: PackedU32([lane_factor as u32, 0, 0, 0]),
                        scalar: PackedF32([scalar as f32, 0.0, 0.0, 0.0]),
                    };
                    self.uniform_buffer(emission, &params, "runmat-trapezoid-params")?
                }
            };
            let bind_group = emission
                .device()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-trapezoid-bind"),
                    layout: &self.pipelines.trapezoid.get(emission)?.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: entry.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: out_buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: params_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: spacing_buffer.as_ref().as_entire_binding(),
                        },
                    ],
                });
            let groups = self.pipelines.trapezoid.linear_dispatch_groups(segments)?;
            crate::backend::wgpu::dispatch::scan::run(
                emission,
                pending,
                &self.pipelines.trapezoid.get(emission)?.pipeline,
                &bind_group,
                groups,
                "runmat-trapezoid-encoder",
                "runmat-trapezoid-pass",
            )?;
            Ok(())
        })??;
        self.register_existing_buffer_with_storage(
            out_buffer,
            output_shape,
            output_raw_len,
            entry.storage,
        )
    }
    pub(crate) fn cumprod_exec(
        &self,
        handle: &GpuTensorHandle,
        dim: usize,
        direction: ProviderScanDirection,
        nan_mode: ProviderNanMode,
    ) -> Result<GpuTensorHandle> {
        let selected = self.get_entry(handle)?;
        if selected.len != 0 {
            self.require_ieee_words(selected.precision)?;
        }
        // For reverse scans, compute as flip → forward-scan → flip to preserve exact semantics
        if matches!(direction, ProviderScanDirection::Reverse) {
            let flipped_in = self.flip_exec(handle, &[dim])?;
            let forward =
                self.cumprod_exec(&flipped_in, dim, ProviderScanDirection::Forward, nan_mode)?;
            drop(flipped_in);
            let flipped_out = self.flip_exec(&forward, &[dim])?;
            drop(forward);
            return Ok(flipped_out);
        }
        let entry = self.get_entry(handle)?;
        if entry.len == 0 {
            return Ok(handle.clone());
        }

        let work_shape = if entry.shape.is_empty() {
            vec![entry.len]
        } else {
            entry.shape.clone()
        };

        if dim >= work_shape.len() {
            return Err(anyhow!(
                "cumprod: dimension {} exceeds tensor rank {}",
                dim + 1,
                work_shape.len()
            ));
        }

        let segment_len = work_shape[dim];
        if segment_len == 0 {
            let out_buffer = self.with_emission(|emission, _| {
                self.create_storage_buffer(emission, 0, "runmat-cumprod-empty")
            })??;
            return self.register_existing_buffer(out_buffer, entry.shape, 0);
        }

        let stride_before = if dim == 0 {
            1usize
        } else {
            work_shape[..dim].iter().copied().product::<usize>().max(1)
        };
        let stride_after = if dim + 1 >= work_shape.len() {
            1usize
        } else {
            work_shape[dim + 1..]
                .iter()
                .copied()
                .product::<usize>()
                .max(1)
        };

        let segments = stride_before
            .checked_mul(stride_after)
            .ok_or_else(|| anyhow!("cumprod: segment count exceeds GPU limits"))?;
        let block = stride_before
            .checked_mul(segment_len)
            .ok_or_else(|| anyhow!("cumprod: segment stride exceeds GPU limits"))?;

        ensure!(
            segment_len <= u32::MAX as usize,
            "cumprod: dimension length exceeds GPU kernel limits"
        );
        ensure!(
            stride_before <= u32::MAX as usize,
            "cumprod: stride_before exceeds GPU kernel limits"
        );
        ensure!(
            segments <= u32::MAX as usize,
            "cumprod: segment count exceeds GPU limits"
        );
        ensure!(
            block <= u32::MAX as usize,
            "cumprod: block size exceeds GPU limits"
        );
        ensure!(
            entry.len <= u32::MAX as usize,
            "cumprod: tensor too large for GPU kernel"
        );

        let out_buffer = self.with_emission(|emission, _| {
            self.create_storage_buffer(emission, entry.len, "runmat-cumprod-out")
        })??;
        if entry.len == 0 {
            return self.register_existing_buffer(out_buffer, entry.shape, 0);
        }

        let mut flags = 0u32;
        if matches!(nan_mode, ProviderNanMode::Omit) {
            flags |= 2;
        }

        let params = CumprodParams {
            segment_len: segment_len as u32,
            segments: segments as u32,
            stride_before: stride_before as u32,
            block: block as u32,
            flags,
            total_len: entry.len as u32,
            _pad0: 0,
            _pad1: 0,
        };
        self.with_emission(|emission, pending| -> Result<()> {
            let params_buffer = self.uniform_buffer(emission, &params, "runmat-cumprod-params")?;
            let bind_group = emission
                .device()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-cumprod-bind"),
                    layout: &self.pipelines.cumprod.get(emission)?.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: entry.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: out_buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: params_buffer.as_entire_binding(),
                        },
                    ],
                });
            let groups = self.pipelines.cumprod.linear_dispatch_groups(segments)?;
            crate::backend::wgpu::dispatch::scan::run(
                emission,
                pending,
                &self.pipelines.cumprod.get(emission)?.pipeline,
                &bind_group,
                groups,
                "runmat-cumprod-encoder",
                "runmat-cumprod-pass",
            )?;
            Ok(())
        })??;
        self.register_existing_buffer(out_buffer, entry.shape, entry.len)
    }
    pub(crate) fn cummin_exec(
        &self,
        handle: &GpuTensorHandle,
        dim: usize,
        direction: ProviderScanDirection,
        nan_mode: ProviderNanMode,
    ) -> Result<ProviderCumminResult> {
        let entry = self.get_entry(handle)?;
        if entry.len == 0 {
            let (values_buffer, indices_buffer) =
                self.with_emission(|emission, _| -> Result<_> {
                    Ok((
                        self.create_storage_buffer(emission, 0, "runmat-cummin-values-empty")?,
                        self.create_storage_buffer(emission, 0, "runmat-cummin-indices-empty")?,
                    ))
                })??;
            let values =
                self.register_existing_buffer(values_buffer, entry.shape.clone(), entry.len)?;
            let indices =
                self.register_existing_buffer(indices_buffer, entry.shape.clone(), entry.len)?;
            return Ok(ProviderCumminResult { values, indices });
        }

        self.require_ieee_words(entry.precision)?;

        let work_shape = if entry.shape.is_empty() {
            vec![entry.len]
        } else {
            entry.shape.clone()
        };

        if dim >= work_shape.len() {
            return Err(anyhow!(
                "cummin: dimension {} exceeds tensor rank {}",
                dim + 1,
                work_shape.len()
            ));
        }

        let segment_len = work_shape[dim];
        if segment_len == 0 {
            let (values_buffer, indices_buffer) =
                self.with_emission(|emission, _| -> Result<_> {
                    Ok((
                        self.create_storage_buffer(emission, 0, "runmat-cummin-values-empty")?,
                        self.create_storage_buffer(emission, 0, "runmat-cummin-indices-empty")?,
                    ))
                })??;
            let values =
                self.register_existing_buffer(values_buffer, entry.shape.clone(), entry.len)?;
            let indices =
                self.register_existing_buffer(indices_buffer, entry.shape.clone(), entry.len)?;
            return Ok(ProviderCumminResult { values, indices });
        }

        let stride_before = if dim == 0 {
            1usize
        } else {
            work_shape[..dim].iter().copied().product::<usize>().max(1)
        };
        let stride_after = if dim + 1 >= work_shape.len() {
            1usize
        } else {
            work_shape[dim + 1..]
                .iter()
                .copied()
                .product::<usize>()
                .max(1)
        };

        let segments = stride_before
            .checked_mul(stride_after)
            .ok_or_else(|| anyhow!("cummin: segment count exceeds GPU limits"))?;
        let block = stride_before
            .checked_mul(segment_len)
            .ok_or_else(|| anyhow!("cummin: segment stride exceeds GPU limits"))?;

        ensure!(
            segment_len <= u32::MAX as usize,
            "cummin: dimension length exceeds GPU kernel limits"
        );
        ensure!(
            stride_before <= u32::MAX as usize,
            "cummin: stride_before exceeds GPU kernel limits"
        );
        ensure!(
            segments <= u32::MAX as usize,
            "cummin: segment count exceeds GPU limits"
        );
        ensure!(
            block <= u32::MAX as usize,
            "cummin: block size exceeds GPU limits"
        );
        ensure!(
            entry.len <= u32::MAX as usize,
            "cummin: tensor too large for GPU kernel"
        );

        let (values_buffer, indices_buffer) =
            self.with_emission(|emission, _| -> Result<_> {
                Ok((
                    self.create_storage_buffer(emission, entry.len, "runmat-cummin-values")?,
                    self.create_storage_buffer(emission, entry.len, "runmat-cummin-indices")?,
                ))
            })??;

        let mut flags = 0u32;
        if matches!(direction, ProviderScanDirection::Reverse) {
            flags |= 1;
        }
        if matches!(nan_mode, ProviderNanMode::Omit) {
            flags |= 2;
        }

        let params = CumminParams {
            segment_len: segment_len as u32,
            segments: segments as u32,
            stride_before: stride_before as u32,
            block: block as u32,
            flags,
            total_len: entry.len as u32,
            _pad0: 0,
            _pad1: 0,
        };
        self.with_emission(|emission, pending| -> Result<()> {
            let params_buffer = self.uniform_buffer(emission, &params, "runmat-cummin-params")?;

            let bind_group = emission
                .device()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-cummin-bind"),
                    layout: &self.pipelines.cummin.get(emission)?.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: entry.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: values_buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: indices_buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: params_buffer.as_entire_binding(),
                        },
                    ],
                });

            let groups = self.pipelines.cummin.linear_dispatch_groups(segments)?;
            crate::backend::wgpu::dispatch::scan::run(
                emission,
                pending,
                &self.pipelines.cummin.get(emission)?.pipeline,
                &bind_group,
                groups,
                "runmat-cummin-encoder",
                "runmat-cummin-pass",
            )?;
            Ok(())
        })??;

        let values =
            self.register_existing_buffer(values_buffer, entry.shape.clone(), entry.len)?;
        let indices =
            self.register_existing_buffer(indices_buffer, entry.shape.clone(), entry.len)?;
        Ok(ProviderCumminResult { values, indices })
    }
    pub(crate) fn cummax_exec(
        &self,
        handle: &GpuTensorHandle,
        dim: usize,
        direction: ProviderScanDirection,
        nan_mode: ProviderNanMode,
    ) -> Result<ProviderCummaxResult> {
        let entry = self.get_entry(handle)?;
        if entry.len == 0 {
            let (values_buffer, indices_buffer) =
                self.with_emission(|emission, _| -> Result<_> {
                    Ok((
                        self.create_storage_buffer(emission, 0, "runmat-cummax-values-empty")?,
                        self.create_storage_buffer(emission, 0, "runmat-cummax-indices-empty")?,
                    ))
                })??;
            let values =
                self.register_existing_buffer(values_buffer, entry.shape.clone(), entry.len)?;
            let indices =
                self.register_existing_buffer(indices_buffer, entry.shape.clone(), entry.len)?;
            return Ok(ProviderCummaxResult { values, indices });
        }

        self.require_ieee_words(entry.precision)?;

        let work_shape = if entry.shape.is_empty() {
            vec![entry.len]
        } else {
            entry.shape.clone()
        };

        if dim >= work_shape.len() {
            return Err(anyhow!(
                "cummax: dimension {} exceeds tensor rank {}",
                dim + 1,
                work_shape.len()
            ));
        }

        let segment_len = work_shape[dim];
        if segment_len == 0 {
            let (values_buffer, indices_buffer) =
                self.with_emission(|emission, _| -> Result<_> {
                    Ok((
                        self.create_storage_buffer(emission, 0, "runmat-cummax-values-empty")?,
                        self.create_storage_buffer(emission, 0, "runmat-cummax-indices-empty")?,
                    ))
                })??;
            let values =
                self.register_existing_buffer(values_buffer, entry.shape.clone(), entry.len)?;
            let indices =
                self.register_existing_buffer(indices_buffer, entry.shape.clone(), entry.len)?;
            return Ok(ProviderCummaxResult { values, indices });
        }

        let stride_before = if dim == 0 {
            1usize
        } else {
            work_shape[..dim].iter().copied().product::<usize>().max(1)
        };
        let stride_after = if dim + 1 >= work_shape.len() {
            1usize
        } else {
            work_shape[dim + 1..]
                .iter()
                .copied()
                .product::<usize>()
                .max(1)
        };

        let segments = stride_before
            .checked_mul(stride_after)
            .ok_or_else(|| anyhow!("cummax: segment count exceeds GPU limits"))?;
        let block = stride_before
            .checked_mul(segment_len)
            .ok_or_else(|| anyhow!("cummax: segment stride exceeds GPU limits"))?;

        ensure!(
            segment_len <= u32::MAX as usize,
            "cummax: dimension length exceeds GPU kernel limits"
        );
        ensure!(
            stride_before <= u32::MAX as usize,
            "cummax: stride_before exceeds GPU kernel limits"
        );
        ensure!(
            segments <= u32::MAX as usize,
            "cummax: segment count exceeds GPU limits"
        );
        ensure!(
            block <= u32::MAX as usize,
            "cummax: block size exceeds GPU limits"
        );
        ensure!(
            entry.len <= u32::MAX as usize,
            "cummax: tensor too large for GPU kernel"
        );

        let (values_buffer, indices_buffer) =
            self.with_emission(|emission, _| -> Result<_> {
                Ok((
                    self.create_storage_buffer(emission, entry.len, "runmat-cummax-values")?,
                    self.create_storage_buffer(emission, entry.len, "runmat-cummax-indices")?,
                ))
            })??;

        let mut flags = 0u32;
        if matches!(direction, ProviderScanDirection::Reverse) {
            flags |= 1;
        }
        if matches!(nan_mode, ProviderNanMode::Omit) {
            flags |= 2;
        }

        let params = CummaxParams {
            segment_len: segment_len as u32,
            segments: segments as u32,
            stride_before: stride_before as u32,
            block: block as u32,
            flags,
            total_len: entry.len as u32,
            _pad0: 0,
            _pad1: 0,
        };
        self.with_emission(|emission, pending| -> Result<()> {
            let params_buffer = self.uniform_buffer(emission, &params, "runmat-cummax-params")?;

            let bind_group = emission
                .device()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-cummax-bind"),
                    layout: &self.pipelines.cummax.get(emission)?.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: entry.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: values_buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: indices_buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: params_buffer.as_entire_binding(),
                        },
                    ],
                });

            let groups = self.pipelines.cummax.linear_dispatch_groups(segments)?;
            crate::backend::wgpu::dispatch::scan::run(
                emission,
                pending,
                &self.pipelines.cummax.get(emission)?.pipeline,
                &bind_group,
                groups,
                "runmat-cummax-encoder",
                "runmat-cummax-pass",
            )?;
            Ok(())
        })??;

        let values =
            self.register_existing_buffer(values_buffer, entry.shape.clone(), entry.len)?;
        let indices =
            self.register_existing_buffer(indices_buffer, entry.shape.clone(), entry.len)?;
        Ok(ProviderCummaxResult { values, indices })
    }
}

fn spectral_selected_frequency_len(nfft: usize, range: ProviderSpectralRange) -> usize {
    match range {
        ProviderSpectralRange::Onesided => nfft / 2 + 1,
        ProviderSpectralRange::Twosided | ProviderSpectralRange::Centered => nfft,
    }
}

fn envelope_hilbert_fir_kernel(filter_len: usize) -> Vec<f64> {
    if filter_len == 0 {
        return Vec::new();
    }
    let center = (filter_len as f64 - 1.0) / 2.0;
    let denominator = envelope_modified_bessel_i0(8.0);
    (0..filter_len)
        .map(|idx| {
            let k = idx as f64 - center;
            let ideal = if k.abs() <= f64::EPSILON {
                0.0
            } else {
                let rounded = k.round();
                if (rounded - k).abs() <= f64::EPSILON && (rounded as i64).rem_euclid(2) == 0 {
                    0.0
                } else {
                    2.0 / (std::f64::consts::PI * k)
                }
            };
            ideal * envelope_kaiser_window(idx, filter_len, denominator)
        })
        .collect()
}

fn envelope_input_shape_matches(shape: &[usize], channel_len: usize, channel_count: usize) -> bool {
    if channel_count == 1 {
        return match shape {
            [len] => *len == channel_len,
            [rows, cols] => {
                (*rows == channel_len && *cols == 1) || (*rows == 1 && *cols == channel_len)
            }
            _ => false,
        };
    }

    matches!(shape, [rows, cols] if *rows == channel_len && *cols == channel_count)
}

fn envelope_kaiser_window(idx: usize, len: usize, denominator: f64) -> f64 {
    if len <= 1 {
        return 1.0;
    }
    let ratio = 2.0 * idx as f64 / (len - 1) as f64 - 1.0;
    let argument = 8.0 * (1.0 - ratio * ratio).max(0.0).sqrt();
    envelope_modified_bessel_i0(argument) / denominator
}

fn envelope_modified_bessel_i0(x: f64) -> f64 {
    let y = x * x / 4.0;
    let mut term = 1.0;
    let mut sum = 1.0;
    for k in 1..=32 {
        term *= y / ((k * k) as f64);
        sum += term;
        if term.abs() <= sum.abs() * 1.0e-15 {
            break;
        }
    }
    sum
}

fn spectral_frame_shader_mode(mode: ProviderSpectralFrameMode) -> SpectralFrameShaderMode {
    match mode {
        ProviderSpectralFrameMode::Sliding { hop } => SpectralFrameShaderMode::Sliding { hop },
        ProviderSpectralFrameMode::ColumnSliding {
            hop,
            input_rows,
            frames_per_column,
        } => SpectralFrameShaderMode::ColumnSliding {
            hop,
            input_rows,
            frames_per_column,
        },
        ProviderSpectralFrameMode::FoldedColumns { input_rows } => {
            SpectralFrameShaderMode::FoldedColumns { input_rows }
        }
    }
}

fn spectral_range_shader_mode(range: ProviderSpectralRange) -> SpectralRangeShaderMode {
    match range {
        ProviderSpectralRange::Onesided => SpectralRangeShaderMode::Onesided,
        ProviderSpectralRange::Twosided => SpectralRangeShaderMode::Twosided,
        ProviderSpectralRange::Centered => SpectralRangeShaderMode::Centered,
    }
}

#[cfg(test)]
mod tests {
    use crate::backend::wgpu::provider::{
        register_retained_test_wgpu_provider, register_test_wgpu_provider, WgpuProvider,
        WgpuProviderOptions,
    };
    use num_complex::Complex;
    use runmat_accelerate_api::{
        AccelProvider, GpuTensorStorage, HostNumericDataView, HostNumericTensorView,
        HostTensorView, ProviderEnvelopeMethod, ProviderEnvelopeRequest, ProviderHilbertRequest,
        ProviderIirFilterOptions, ProviderMovingWindowEndpoints, ProviderMovingWindowOp,
        ProviderMovingWindowRequest, ProviderNanMode, ProviderPreparedIirCoefficients,
        ProviderStdNormalization, ProviderTrapezoidSpacing,
    };
    use runmat_runtime::builtins::math::reduction::{
        gradient_complex_tensor_host, gradient_real_tensor_host_with_coordinates,
    };
    use runmat_value::{ComplexTensor, Tensor};
    use rustfft::FftPlanner;

    fn with_wgpu_provider<F, R>(f: F) -> Option<R>
    where
        F: FnOnce(&dyn AccelProvider) -> R,
    {
        let Ok(provider) = register_test_wgpu_provider(WgpuProviderOptions::default()) else {
            return None;
        };
        Some(f(provider.provider()))
    }

    #[test]
    fn diff_provider_preserves_higher_order_empty_geometry_and_native_precision() {
        use crate::backend::wgpu::provider::WgpuPrecisionPolicy;
        for precision in [
            WgpuPrecisionPolicy::RequireF32,
            WgpuPrecisionPolicy::RequireF64,
        ] {
            let Ok(registered) = register_test_wgpu_provider(WgpuProviderOptions {
                precision,
                ..Default::default()
            }) else {
                continue; // This physical precision is unavailable on this adapter.
            };
            let provider = registered.provider();
            for (shape, order, dim, expected) in [
                (vec![0, 5], 3, 1, vec![0, 2]),
                (vec![2, 0, 7], 4, 2, vec![2, 0, 3]),
                (vec![0, 5], usize::MAX, 1, vec![0, 0]),
            ] {
                let data = match precision {
                    WgpuPrecisionPolicy::RequireF32 => HostNumericDataView::F32(&[]),
                    WgpuPrecisionPolicy::RequireF64 => HostNumericDataView::F64(&[]),
                    _ => unreachable!(),
                };
                let input = provider
                    .upload_numeric(&HostNumericTensorView {
                        data,
                        shape: &shape,
                        storage: GpuTensorStorage::Real,
                    })
                    .expect("native empty input");
                let output = provider
                    .diff_dim(&input, order, dim)
                    .expect("empty differences")
                    .expect("admitted native precision");
                assert_eq!(output.shape(), expected);
                assert_eq!(
                    output.physical().element_type(),
                    input.physical().element_type()
                );
                assert_eq!(output.physical().storage(), input.physical().storage());
                assert_eq!(output.device_id(), input.device_id());
                assert!(!output.shares_allocation_with(&input));
                assert!(pollster::block_on(provider.download(&output))
                    .unwrap()
                    .data
                    .is_empty());
                output.release().unwrap();
                input.allocation_lease().validate_live().unwrap();
                input.release().unwrap();
            }
        }
    }

    fn run_envelope(method: ProviderEnvelopeMethod, values: &[f64], shape: &[usize]) -> Vec<f64> {
        with_wgpu_provider(|provider| run_envelope_on(provider, method, values, shape))
            .unwrap_or_default()
    }

    fn with_retained_wgpu_provider<F, R>(f: F) -> Option<R>
    where
        F: FnOnce(&'static dyn AccelProvider) -> R,
    {
        let Ok(session) = register_retained_test_wgpu_provider(WgpuProviderOptions::default())
        else {
            return None;
        };
        Some(f(session.provider()))
    }

    fn run_envelope_on(
        provider: &dyn AccelProvider,
        method: ProviderEnvelopeMethod,
        values: &[f64],
        shape: &[usize],
    ) -> Vec<f64> {
        let input = provider
            .upload(&HostTensorView {
                data: values,
                shape,
            })
            .expect("upload");
        let result = pollster::block_on(provider.signal_envelope(&ProviderEnvelopeRequest {
            input: &input,
            channel_len: values.len(),
            channel_count: 1,
            output_shape: shape,
            method,
        }))
        .expect("envelope");
        let upper = pollster::block_on(provider.download(&result.upper)).expect("download");
        input.release().unwrap();
        result.upper.release().unwrap();
        result.lower.release().unwrap();
        upper.data
    }

    #[test]
    fn moving_window_provider_runs_count_window_reductions_on_wgpu() {
        let Some(()) = with_wgpu_provider(|provider| {
            let input = provider
                .upload(&HostTensorView {
                    data: &[1.0, 2.0, f64::NAN, 4.0],
                    shape: &[1, 4],
                })
                .expect("upload moving input");
            let sum = pollster::block_on(provider.moving_window(&ProviderMovingWindowRequest {
                input: &input,
                output_shape: &[1, 4],
                dim: 1,
                before: 1,
                after: 1,
                op: ProviderMovingWindowOp::Sum,
                endpoints: ProviderMovingWindowEndpoints::Shrink,
                nan_mode: ProviderNanMode::Omit,
                normalization: ProviderStdNormalization::Sample,
            }))
            .expect("moving window");
            let host = pollster::block_on(provider.download(&sum)).expect("download moving");
            assert_eq!(host.shape, vec![1, 4]);
            assert_eq!(host.data, vec![3.0, 3.0, 6.0, 4.0]);

            let var_input = provider
                .upload(&HostTensorView {
                    data: &[1.0, 2.0],
                    shape: &[1, 2],
                })
                .expect("upload var input");
            let var = pollster::block_on(provider.moving_window(&ProviderMovingWindowRequest {
                input: &var_input,
                output_shape: &[1, 2],
                dim: 1,
                before: 1,
                after: 1,
                op: ProviderMovingWindowOp::Var,
                endpoints: ProviderMovingWindowEndpoints::Fill(0.0),
                nan_mode: ProviderNanMode::Include,
                normalization: ProviderStdNormalization::Sample,
            }))
            .expect("moving var");
            let host = pollster::block_on(provider.download(&var)).expect("download var");
            assert_eq!(host.shape, vec![1, 2]);
            assert!((host.data[0] - 1.0).abs() < 1e-12);
            assert!((host.data[1] - 1.0).abs() < 1e-12);

            let median_input = provider
                .upload(&HostTensorView {
                    data: &[1.0, f64::NAN, 3.0, 9.0],
                    shape: &[1, 4],
                })
                .expect("upload median input");
            let median = pollster::block_on(provider.moving_window(&ProviderMovingWindowRequest {
                input: &median_input,
                output_shape: &[1, 4],
                dim: 1,
                before: 1,
                after: 1,
                op: ProviderMovingWindowOp::Median,
                endpoints: ProviderMovingWindowEndpoints::Shrink,
                nan_mode: ProviderNanMode::Omit,
                normalization: ProviderStdNormalization::Sample,
            }))
            .expect("moving median");
            let host = pollster::block_on(provider.download(&median)).expect("download median");
            assert_eq!(host.shape, vec![1, 4]);
            for (actual, expected) in host.data.iter().zip([1.0, 2.0, 6.0, 6.0]) {
                assert!((*actual - expected).abs() < 1e-12);
            }

            let oversized =
                pollster::block_on(provider.moving_window(&ProviderMovingWindowRequest {
                    input: &median_input,
                    output_shape: &[1, 4],
                    dim: 1,
                    before: super::WGPU_MOVING_MEDIAN_MAX_WINDOW,
                    after: 0,
                    op: ProviderMovingWindowOp::Median,
                    endpoints: ProviderMovingWindowEndpoints::Shrink,
                    nan_mode: ProviderNanMode::Omit,
                    normalization: ProviderStdNormalization::Sample,
                }));
            assert!(oversized.is_err());

            input.release().unwrap();
            sum.release().unwrap();
            var_input.release().unwrap();
            var.release().unwrap();
            median_input.release().unwrap();
            median.release().unwrap();
        }) else {
            return;
        };
    }

    fn run_hilbert(
        values: &[f64],
        shape: &[usize],
        length: Option<usize>,
        dim: usize,
    ) -> Vec<(f64, f64)> {
        with_wgpu_provider(|provider| {
            let input = provider
                .upload(&HostTensorView {
                    data: values,
                    shape,
                })
                .expect("upload");
            let output = pollster::block_on(provider.signal_hilbert(&ProviderHilbertRequest {
                input: &input,
                length,
                dim,
            }))
            .expect("hilbert");
            assert_eq!(
                output.descriptor().storage,
                Some(GpuTensorStorage::ComplexInterleaved)
            );
            let host = pollster::block_on(provider.download(&output)).expect("download");
            assert_eq!(host.storage, GpuTensorStorage::ComplexInterleaved);
            assert_eq!(host.data.len() % 2, 0);
            input.release().unwrap();
            output.release().unwrap();
            host.data
                .chunks_exact(2)
                .map(|pair| (pair[0], pair[1]))
                .collect()
        })
        .unwrap_or_default()
    }

    fn run_complex_gradient(
        values: &[(f64, f64)],
        shape: &[usize],
        dim: usize,
        spacing: f64,
    ) -> Vec<(f64, f64)> {
        with_wgpu_provider(|provider| {
            let mut interleaved = Vec::with_capacity(values.len() * 2);
            for &(re, im) in values {
                interleaved.push(re as f32);
                interleaved.push(im as f32);
            }
            let input = provider
                .upload_numeric(&HostNumericTensorView {
                    data: HostNumericDataView::F32(&interleaved),
                    shape,
                    storage: GpuTensorStorage::ComplexInterleaved,
                })
                .expect("upload");
            let output = provider
                .gradient_dim(&input, dim, spacing)
                .expect("gradient")
                .expect("provider accepted numerical operation");
            let host = pollster::block_on(provider.download(&output)).expect("download");
            assert_eq!(host.storage, GpuTensorStorage::ComplexInterleaved);
            assert_eq!(host.data.len() % 2, 0);
            input.release().unwrap();
            output.release().unwrap();
            host.data
                .chunks_exact(2)
                .map(|pair| (pair[0], pair[1]))
                .collect()
        })
        .unwrap_or_default()
    }

    fn host_hilbert(
        values: &[f64],
        shape: &[usize],
        length: Option<usize>,
        dim: usize,
    ) -> Vec<(f64, f64)> {
        let mut ext_shape = shape.to_vec();
        while ext_shape.len() <= dim {
            ext_shape.push(1);
        }
        let current_len = ext_shape[dim];
        let target_len = length.unwrap_or(current_len);
        let inner_stride = ext_shape[..dim].iter().copied().product::<usize>();
        let outer_stride = ext_shape[dim + 1..].iter().copied().product::<usize>();
        let mut out = vec![Complex::new(0.0, 0.0); target_len * inner_stride * outer_stride];
        let input = values
            .iter()
            .copied()
            .map(|re| Complex::new(re, 0.0))
            .collect::<Vec<_>>();
        let mut planner = FftPlanner::<f64>::new();
        let forward = (target_len > 1).then(|| planner.plan_fft_forward(target_len));
        let inverse = (target_len > 1).then(|| planner.plan_fft_inverse(target_len));
        let mut buffer = vec![Complex::new(0.0, 0.0); target_len];
        let copy_len = current_len.min(target_len);

        for outer in 0..outer_stride {
            let base_in = outer * current_len * inner_stride;
            let base_out = outer * target_len * inner_stride;
            for inner in 0..inner_stride {
                buffer.fill(Complex::new(0.0, 0.0));
                for (freq, slot) in buffer.iter_mut().enumerate().take(copy_len) {
                    *slot = input[base_in + inner + freq * inner_stride];
                }
                if let Some(plan) = &forward {
                    plan.process(&mut buffer);
                }
                for (freq, slot) in buffer.iter_mut().enumerate() {
                    *slot *= analytic_test_multiplier(freq, target_len);
                }
                if let Some(plan) = &inverse {
                    plan.process(&mut buffer);
                }
                let scale = if target_len > 0 {
                    1.0 / target_len as f64
                } else {
                    1.0
                };
                for (freq, value) in buffer.iter().enumerate() {
                    out[base_out + inner + freq * inner_stride] = *value * scale;
                }
            }
        }

        out.into_iter().map(|value| (value.re, value.im)).collect()
    }

    fn analytic_test_multiplier(freq: usize, len: usize) -> f64 {
        if freq == 0 {
            1.0
        } else if len.is_multiple_of(2) {
            if freq < len / 2 {
                2.0
            } else if freq == len / 2 {
                1.0
            } else {
                0.0
            }
        } else if freq <= len / 2 {
            2.0
        } else {
            0.0
        }
    }

    fn assert_complex_slices_close(actual: &[(f64, f64)], expected: &[(f64, f64)], tol: f64) {
        assert_eq!(actual.len(), expected.len());
        for (idx, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual.0 - expected.0).abs() <= tol,
                "real mismatch at {idx}: actual={} expected={}",
                actual.0,
                expected.0
            );
            assert!(
                (actual.1 - expected.1).abs() <= tol,
                "imag mismatch at {idx}: actual={} expected={}",
                actual.1,
                expected.1
            );
        }
    }

    #[test]
    fn signal_envelope_rms_provider_matches_centered_window() {
        let values = [0.0, 3.0, 4.0, 0.0];
        let upper = run_envelope(
            ProviderEnvelopeMethod::Rms { window_len: 3 },
            &values,
            &[4, 1],
        );
        if upper.is_empty() {
            return;
        }
        assert!((upper[1] - (25.0f64 / 3.0).sqrt()).abs() < 1.0e-5);
    }

    #[test]
    fn signal_envelope_provider_rejects_layout_mismatch_with_same_len() {
        let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let Some(()) = with_wgpu_provider(|provider| {
            let input = provider
                .upload(&HostTensorView {
                    data: &values,
                    shape: &[3, 2],
                })
                .expect("upload");
            let output_shape = [2, 3];
            let request = ProviderEnvelopeRequest {
                input: &input,
                channel_len: 2,
                channel_count: 3,
                output_shape: &output_shape,
                method: ProviderEnvelopeMethod::Rms { window_len: 3 },
            };
            assert!(!provider.supports_signal_envelope(&request));
            let err = pollster::block_on(provider.signal_envelope(&request))
                .expect_err("layout mismatch should be rejected");
            assert!(
                err.is_unsupported(),
                "expected pre-effect unsupported: {err}"
            );
            input.allocation_lease().validate_live().unwrap();
            input.release().unwrap();
        }) else {
            return;
        };
    }

    #[test]
    fn signal_envelope_analytic_provider_keeps_constant_bounds() {
        let values = [2.5; 8];
        let upper = run_envelope(ProviderEnvelopeMethod::Analytic, &values, &[8, 1]);
        if upper.is_empty() {
            return;
        }
        for value in upper {
            assert!((value - 2.5).abs() < 1.0e-10);
        }
    }

    #[test]
    fn signal_envelope_analytic_fir_provider_keeps_constant_bounds() {
        let values = [1.25; 8];
        let upper = run_envelope(
            ProviderEnvelopeMethod::AnalyticFir { filter_len: 5 },
            &values,
            &[8, 1],
        );
        if upper.is_empty() {
            return;
        }
        for value in upper {
            assert!((value - 1.25).abs() < 1.0e-10);
        }
    }

    #[test]
    #[ignore = "requires actual WGPU device; run explicitly for indexed envelope regression"]
    fn signal_envelope_indexed_operands_preserve_row_vector_output_shape() {
        let values = [1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0];
        with_wgpu_provider(|provider| {
            for method in [
                ProviderEnvelopeMethod::Analytic,
                ProviderEnvelopeMethod::AnalyticFir { filter_len: 5 },
                ProviderEnvelopeMethod::Rms { window_len: 3 },
            ] {
                let column = run_envelope_on(provider, method, &values, &[8, 1]);
                let row = run_envelope_on(provider, method, &values, &[1, 8]);
                assert_eq!(row.len(), column.len());
                for (actual, expected) in row.iter().zip(column.iter()) {
                    assert!((actual - expected).abs() < 1.0e-5);
                }
            }
        })
        .expect("actual WGPU device required");
    }

    #[test]
    #[ignore = "requires actual WGPU device; run explicitly for indexed envelope regression"]
    fn signal_envelope_direct_indexing_materializes_lazy_transpose() {
        with_wgpu_provider(|provider| {
            let base = provider
                .upload(&HostTensorView {
                    data: &[1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0],
                    shape: &[2, 4],
                })
                .expect("base upload");
            let lazy = provider.transpose(&base).expect("lazy transpose");
            let contiguous = provider
                .upload(&HostTensorView {
                    data: &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                    shape: &[4, 2],
                })
                .expect("contiguous upload");
            for method in [
                ProviderEnvelopeMethod::Analytic,
                ProviderEnvelopeMethod::AnalyticFir { filter_len: 5 },
                ProviderEnvelopeMethod::Rms { window_len: 3 },
            ] {
                let invoke = |input| {
                    pollster::block_on(provider.signal_envelope(&ProviderEnvelopeRequest {
                        input,
                        channel_len: 4,
                        channel_count: 2,
                        output_shape: &[4, 2],
                        method,
                    }))
                    .expect("indexed envelope")
                };
                let observed = invoke(&lazy);
                let expected = invoke(&contiguous);
                for (actual, reference) in [
                    (&observed.upper, &expected.upper),
                    (&observed.lower, &expected.lower),
                ] {
                    let actual =
                        pollster::block_on(provider.download(actual)).expect("download lazy");
                    let reference = pollster::block_on(provider.download(reference))
                        .expect("download contiguous");
                    assert_eq!(actual.shape, reference.shape);
                    for (actual, reference) in actual.data.iter().zip(reference.data.iter()) {
                        assert!((actual - reference).abs() < 1.0e-5);
                    }
                }
                observed.upper.release().unwrap();
                observed.lower.release().unwrap();
                expected.upper.release().unwrap();
                expected.lower.release().unwrap();
            }
            assert!(lazy.allocation_lease().validate_live().is_ok());
            drop(lazy);
            base.release().unwrap();
            contiguous.release().unwrap();
        })
        .expect("actual WGPU device required");
    }

    #[test]
    fn signal_hilbert_provider_matches_row_cosine() {
        let values = [1.0, 0.0, -1.0, 0.0];
        let out = run_hilbert(&values, &[1, 4], None, 1);
        if out.is_empty() {
            return;
        }
        let expected = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)];
        for (idx, (actual, expected)) in out.iter().copied().zip(expected).enumerate() {
            assert!(
                (actual.0 - expected.0).abs() < 1.0e-5,
                "real mismatch at {idx}: actual={} expected={}",
                actual.0,
                expected.0
            );
            assert!(
                (actual.1 - expected.1).abs() < 1.0e-5,
                "imag mismatch at {idx}: actual={} expected={}",
                actual.1,
                expected.1
            );
        }
    }

    #[test]
    fn signal_hilbert_provider_operates_down_columns() {
        let values = [1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, -1.0];
        let out = run_hilbert(&values, &[4, 2], None, 0);
        if out.is_empty() {
            return;
        }
        let expected = [
            (1.0, 0.0),
            (0.0, 1.0),
            (-1.0, 0.0),
            (0.0, -1.0),
            (0.0, -1.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (-1.0, 0.0),
        ];
        for (idx, (actual, expected)) in out.iter().copied().zip(expected).enumerate() {
            assert!(
                (actual.0 - expected.0).abs() < 1.0e-5,
                "real mismatch at {idx}: actual={} expected={}",
                actual.0,
                expected.0
            );
            assert!(
                (actual.1 - expected.1).abs() < 1.0e-5,
                "imag mismatch at {idx}: actual={} expected={}",
                actual.1,
                expected.1
            );
        }
    }

    #[test]
    fn signal_hilbert_provider_matches_host_for_padded_inner_stride() {
        let values = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let out = run_hilbert(&values, &[2, 3], Some(5), 1);
        if out.is_empty() {
            return;
        }
        let expected = host_hilbert(&values, &[2, 3], Some(5), 1);
        assert_complex_slices_close(&out, &expected, 1.0e-5);
    }

    #[test]
    fn signal_hilbert_provider_matches_host_for_truncated_odd_signal() {
        let values = [1.0, 2.0, 0.5, -1.0, 0.25];
        let out = run_hilbert(&values, &[1, 5], Some(3), 1);
        if out.is_empty() {
            return;
        }
        let expected = host_hilbert(&values, &[1, 5], Some(3), 1);
        assert_complex_slices_close(&out, &expected, 1.0e-5);
    }

    #[test]
    fn gradient_provider_complex_matches_host_and_stays_interleaved() {
        let values = [
            (1.0, 1.0),
            (2.0, -1.0),
            (4.0, 3.0),
            (6.0, 2.0),
            (9.0, 6.0),
            (12.0, 4.0),
        ];
        let out = run_complex_gradient(&values, &[2, 3], 1, 2.0);
        if out.is_empty() {
            return;
        }
        let expected = gradient_complex_tensor_host(
            ComplexTensor::new(values.to_vec(), vec![2, 3]).expect("complex tensor"),
            2,
            2.0,
        )
        .expect("host gradient");
        assert_complex_slices_close(&out, &expected.materialize_f64(), 1.0e-5);
    }

    #[test]
    fn gradient_provider_coordinate_spacing_matches_host_and_stays_resident() {
        with_wgpu_provider(|provider| {
            let input = provider
                .upload(&HostTensorView {
                    data: &[1.0, 4.0, 9.0],
                    shape: &[1, 3],
                })
                .expect("upload input");
            let coordinates = provider
                .upload(&HostTensorView {
                    data: &[0.0, 1.0, 3.0],
                    shape: &[1, 3],
                })
                .expect("upload coordinates");
            let output = provider
                .gradient_dim_with_coordinates(&input, 1, &coordinates)
                .expect("gradient");
            let gathered = pollster::block_on(provider.download(&output)).expect("download");
            let expected = gradient_real_tensor_host_with_coordinates(
                Tensor::new(vec![1.0, 4.0, 9.0], vec![1, 3]).unwrap(),
                2,
                vec![0.0, 1.0, 3.0],
            )
            .expect("host gradient");
            input.release().unwrap();
            coordinates.release().unwrap();
            output.release().unwrap();
            assert_eq!(gathered.shape.as_slice(), expected.shape());
            let tol = match provider.precision() {
                runmat_accelerate_api::ProviderPrecision::F64 => 1.0e-10,
                runmat_accelerate_api::ProviderPrecision::F32 => 2.0e-6,
            };
            for (idx, (actual, expected)) in gathered
                .data
                .iter()
                .zip(expected.materialize_f64())
                .enumerate()
            {
                assert!(
                    (*actual - expected).abs() < tol * expected.abs().max(1.0),
                    "gradient mismatch at {idx}: actual={actual} expected={expected}"
                );
            }
        });
    }

    #[test]
    fn iir_filter_provider_preserves_complex_storage_and_filters_lanes() {
        with_wgpu_provider(|provider| {
            let b = provider
                .upload(&HostTensorView {
                    data: &[1.0, 1.0],
                    shape: &[1, 2],
                })
                .expect("upload numerator");
            let a = provider
                .upload(&HostTensorView {
                    data: &[1.0],
                    shape: &[1, 1],
                })
                .expect("upload denominator");
            let data = [1.0_f32, 10.0, 2.0, 20.0, 3.0, 30.0];
            let x = provider
                .upload_numeric(&HostNumericTensorView {
                    data: HostNumericDataView::F32(&data),
                    shape: &[1, 3],
                    storage: GpuTensorStorage::ComplexInterleaved,
                })
                .expect("upload signal");

            let result = pollster::block_on(provider.iir_filter(
                runmat_accelerate_api::ProviderIirCoefficientSource::Resident {
                    numerator: &b,
                    denominator: &a,
                },
                &x,
                ProviderIirFilterOptions { dim: 1, zi: None },
            ))
            .expect("iir filter");
            let output = pollster::block_on(provider.download(&result.output)).expect("download");
            let final_state =
                pollster::block_on(provider.download(result.final_state.as_ref().unwrap()))
                    .expect("download state");
            b.release().unwrap();
            a.release().unwrap();
            x.release().unwrap();
            result.output.release().unwrap();
            result.final_state.unwrap().release().unwrap();

            assert_eq!(output.storage, GpuTensorStorage::ComplexInterleaved);
            assert_eq!(output.shape, vec![1, 3]);
            assert_eq!(output.data, vec![1.0, 10.0, 3.0, 30.0, 5.0, 50.0]);
            assert_eq!(final_state.storage, GpuTensorStorage::ComplexInterleaved);
            assert_eq!(final_state.shape, vec![1, 1]);
            assert_eq!(final_state.data, vec![3.0, 30.0]);
        });
    }

    #[test]
    fn iir_filter_nonleading_axis_uses_order_first_state_geometry() {
        let Some(()) = with_retained_wgpu_provider(|provider| {
            let x = provider
                .upload(&HostTensorView {
                    data: &[
                        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                    ],
                    shape: &[3, 4],
                })
                .expect("matrix signal");
            let zi = provider
                .upload(&HostTensorView {
                    data: &[10.0, 20.0, 30.0],
                    shape: &[1, 3],
                })
                .expect("order-first initial state");
            let prepared = ProviderPreparedIirCoefficients::upload(
                provider,
                &[1.0, 0.0],
                &[1.0, 0.0],
                provider.precision(),
                &[&x, &zi],
            )
            .expect("prepare normalized coefficients before observation");
            provider.reset_telemetry();
            let result = pollster::block_on(provider.iir_filter(
                runmat_accelerate_api::ProviderIirCoefficientSource::Prepared(&prepared),
                &x,
                ProviderIirFilterOptions {
                    dim: 1,
                    zi: Some(zi.clone()),
                },
            ))
            .expect("prepared nonleading filter");
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
            let final_state = result.final_state.expect("final state");
            assert_eq!(final_state.shape(), &[1, 3]);
            let actual = pollster::block_on(provider.download(&result.output)).unwrap();
            let state = pollster::block_on(provider.download(&final_state)).unwrap();
            assert_eq!(actual.shape, vec![3, 4]);
            assert_eq!(actual.data.len(), 12);
            for (got, want) in actual.data.iter().zip([
                11.0, 22.0, 33.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ]) {
                assert!((got - want).abs() < 1.0e-5);
            }
            assert_eq!(state.shape, vec![1, 3]);
            assert_eq!(state.data.len(), 3);
            assert!(state.data.iter().all(|value| value.abs() < 1.0e-5));
            assert!(x.allocation_lease().validate_live().is_ok());
            result.output.release().unwrap();
            final_state.release().unwrap();
            zi.release().unwrap();
            x.release().unwrap();
        }) else {
            eprintln!("WGPU adapter unavailable: nonleading IIR hardware witness not executed");
            return;
        };
    }

    #[test]
    fn iir_filter_third_axis_uses_order_first_state_geometry() {
        let Some(()) = with_retained_wgpu_provider(|provider| {
            let x = provider
                .upload(&HostTensorView {
                    data: &[
                        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                        15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
                    ],
                    shape: &[2, 3, 4],
                })
                .expect("three-dimensional signal");
            let zi = provider
                .upload(&HostTensorView {
                    data: &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
                    shape: &[1, 2, 3],
                })
                .expect("third-axis initial state");
            let prepared = ProviderPreparedIirCoefficients::upload(
                provider,
                &[1.0, 0.0],
                &[1.0, 0.0],
                provider.precision(),
                &[&x, &zi],
            )
            .expect("prepare normalized coefficients before observation");
            provider.reset_telemetry();
            let result = pollster::block_on(provider.iir_filter(
                runmat_accelerate_api::ProviderIirCoefficientSource::Prepared(&prepared),
                &x,
                ProviderIirFilterOptions {
                    dim: 2,
                    zi: Some(zi.clone()),
                },
            ))
            .expect("prepared third-axis filter");
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
            let final_state = result.final_state.expect("final state");
            assert_eq!(final_state.shape(), &[1, 2, 3]);
            let actual = pollster::block_on(provider.download(&result.output)).unwrap();
            let state = pollster::block_on(provider.download(&final_state)).unwrap();
            let expected = [
                11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
            ];
            assert_eq!(actual.shape, vec![2, 3, 4]);
            assert!(actual
                .data
                .iter()
                .zip(expected)
                .all(|(got, want)| (got - want).abs() < 1.0e-5));
            assert_eq!(state.shape, vec![1, 2, 3]);
            assert_eq!(state.data.len(), 6);
            assert!(state.data.iter().all(|value| value.abs() < 1.0e-5));
            assert!(x.allocation_lease().validate_live().is_ok());
            result.output.release().unwrap();
            final_state.release().unwrap();
            zi.release().unwrap();
            x.release().unwrap();
        }) else {
            eprintln!("WGPU adapter unavailable: third-axis IIR hardware witness not executed");
            return;
        };
    }

    #[test]
    fn iir_filter_resident_coefficients_read_back_and_normalize_nonunit_denominator() {
        let Some(()) = with_wgpu_provider(|provider| {
            let b = provider
                .upload(&HostTensorView {
                    data: &[2.0, 0.0],
                    shape: &[1, 2],
                })
                .expect("resident numerator");
            let a = provider
                .upload(&HostTensorView {
                    data: &[2.0, 1.0],
                    shape: &[1, 2],
                })
                .expect("nonunit resident denominator");
            let x = provider
                .upload(&HostTensorView {
                    data: &[1.0, 2.0, 3.0],
                    shape: &[1, 3],
                })
                .expect("resident signal");
            provider.reset_telemetry();
            let result = pollster::block_on(provider.iir_filter(
                runmat_accelerate_api::ProviderIirCoefficientSource::Resident {
                    numerator: &b,
                    denominator: &a,
                },
                &x,
                ProviderIirFilterOptions { dim: 1, zi: None },
            ))
            .expect("resident coefficients normalize by a0");
            let expected_readback = match provider.precision() {
                runmat_accelerate_api::ProviderPrecision::F32 => 16,
                runmat_accelerate_api::ProviderPrecision::F64 => 32,
            };
            assert_eq!(
                provider.telemetry_snapshot().download_bytes,
                expected_readback
            );
            let output = pollster::block_on(provider.download(&result.output)).expect("output");
            assert_eq!(output.shape, vec![1, 3]);
            for (actual, expected) in output.data.iter().zip([1.0, 1.5, 2.25]) {
                assert!((*actual - expected).abs() < 1.0e-5);
            }
            result.output.release().unwrap();
            result.final_state.expect("final state").release().unwrap();
            x.release().unwrap();
            a.release().unwrap();
            b.release().unwrap();
        }) else {
            eprintln!("WGPU adapter unavailable: resident IIR normalization not executed");
            return;
        };
    }

    #[test]
    fn iir_filter_resident_zero_leading_denominator_rejects_after_accounted_readback() {
        let Some(()) = with_wgpu_provider(|provider| {
            let b = provider
                .upload(&HostTensorView {
                    data: &[2.0, 0.0],
                    shape: &[1, 2],
                })
                .expect("resident numerator");
            let a = provider
                .upload(&HostTensorView {
                    data: &[0.0, 1.0],
                    shape: &[1, 2],
                })
                .expect("invalid resident denominator");
            let x = provider
                .upload(&HostTensorView {
                    data: &[1.0, 2.0, 3.0],
                    shape: &[1, 3],
                })
                .expect("resident signal");
            provider.reset_telemetry();
            let error = pollster::block_on(provider.iir_filter(
                runmat_accelerate_api::ProviderIirCoefficientSource::Resident {
                    numerator: &b,
                    denominator: &a,
                },
                &x,
                ProviderIirFilterOptions { dim: 1, zi: None },
            ))
            .expect_err("zero leading denominator must reject");
            assert!(error.to_string().contains("a(1) must be non-zero"));
            let expected_readback = match provider.precision() {
                runmat_accelerate_api::ProviderPrecision::F32 => 16,
                runmat_accelerate_api::ProviderPrecision::F64 => 32,
            };
            assert_eq!(
                provider.telemetry_snapshot().download_bytes,
                expected_readback
            );
            assert!(x.allocation_lease().validate_live().is_ok());
            x.release().unwrap();
            a.release().unwrap();
            b.release().unwrap();
        }) else {
            eprintln!("WGPU adapter unavailable: resident IIR rejection not executed");
            return;
        };
    }

    #[test]
    fn iir_filter_unit_denominator_uses_resident_numerator_without_download() {
        let Ok(provider) = WgpuProvider::new(WgpuProviderOptions::default()) else {
            return;
        };
        {
            let provider: &dyn AccelProvider = &provider;
            let b = provider
                .upload(&HostTensorView {
                    data: &[1.0, 1.0],
                    shape: &[1, 2],
                })
                .expect("upload numerator");
            let a = provider
                .upload(&HostTensorView {
                    data: &[1.0],
                    shape: &[1, 1],
                })
                .expect("upload denominator");
            let x = provider
                .upload(&HostTensorView {
                    data: &[1.0, 2.0, 3.0],
                    shape: &[1, 3],
                })
                .expect("upload signal");
            provider.reset_telemetry();

            let result = pollster::block_on(provider.iir_filter(
                runmat_accelerate_api::ProviderIirCoefficientSource::UnitDenominatorFir {
                    numerator: &b,
                },
                &x,
                ProviderIirFilterOptions { dim: 1, zi: None },
            ))
            .expect("iir filter");
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);

            let output = pollster::block_on(provider.download(&result.output)).expect("download");
            b.release().unwrap();
            a.release().unwrap();
            x.release().unwrap();
            result.output.release().unwrap();
            result.final_state.unwrap().release().unwrap();

            assert_eq!(output.shape, vec![1, 3]);
            assert_eq!(output.data, vec![1.0, 3.0, 5.0]);
        }
    }

    #[test]
    #[ignore = "run explicitly on actual WGPU hardware"]
    fn prepared_iir_coefficients_are_uploaded_once_and_reused_without_download() {
        let session = register_retained_test_wgpu_provider(WgpuProviderOptions::default())
            .expect("actual WGPU provider required");
        let provider = session.provider();
        let prepared = ProviderPreparedIirCoefficients::upload(
            provider,
            &[0.5, 0.5],
            &[1.0, 0.0],
            provider.precision(),
            &[],
        )
        .expect("prepare coefficients once");
        let input = provider
            .upload(&HostTensorView {
                data: &[1.0, 3.0, 5.0],
                shape: &[1, 3],
            })
            .expect("upload signal");
        provider.reset_telemetry();
        let first = pollster::block_on(provider.iir_filter(
            runmat_accelerate_api::ProviderIirCoefficientSource::Prepared(&prepared),
            &input,
            ProviderIirFilterOptions { dim: 1, zi: None },
        ))
        .expect("first pass");
        let second = pollster::block_on(provider.iir_filter(
            runmat_accelerate_api::ProviderIirCoefficientSource::Prepared(&prepared),
            &first.output,
            ProviderIirFilterOptions { dim: 1, zi: None },
        ))
        .expect("second pass");
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        let output =
            pollster::block_on(provider.download(&second.output)).expect("result readback");
        let tolerance = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-6,
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
        };
        for (actual, expected) in output.data.iter().zip([0.25, 1.25, 3.0]) {
            assert!((*actual - expected).abs() < tolerance);
        }
    }

    #[test]
    fn trapezoid_provider_trapz_vector_spacing_matches_host() {
        with_wgpu_provider(|provider| {
            let input = provider
                .upload(&HostTensorView {
                    data: &[0.0, 1.0, 2.0],
                    shape: &[1, 3],
                })
                .expect("upload input");
            let spacing = provider
                .upload(&HostTensorView {
                    data: &[0.0, 1.0, 3.0],
                    shape: &[1, 3],
                })
                .expect("upload spacing");
            let output = provider
                .trapz_dim(&input, 1, ProviderTrapezoidSpacing::Vector(&spacing))
                .expect("trapz")
                .expect("provider accepted numerical operation");
            let gathered = pollster::block_on(provider.download(&output)).expect("download");
            input.release().unwrap();
            spacing.release().unwrap();
            output.release().unwrap();
            assert_eq!(gathered.shape, vec![1, 1]);
            assert!((gathered.data[0] - 3.5).abs() < 1.0e-5);
        });
    }

    #[test]
    fn trapezoid_provider_cumtrapz_tensor_spacing_matches_host() {
        with_wgpu_provider(|provider| {
            let input = provider
                .upload(&HostTensorView {
                    data: &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
                    shape: &[2, 3],
                })
                .expect("upload input");
            let spacing = provider
                .upload(&HostTensorView {
                    data: &[0.0, 0.0, 1.0, 1.0, 3.0, 3.0],
                    shape: &[2, 3],
                })
                .expect("upload spacing");
            let output = provider
                .cumtrapz_dim(&input, 1, ProviderTrapezoidSpacing::Tensor(&spacing))
                .expect("cumtrapz")
                .expect("provider accepted numerical operation");
            let gathered = pollster::block_on(provider.download(&output)).expect("download");
            input.release().unwrap();
            spacing.release().unwrap();
            output.release().unwrap();
            assert_eq!(gathered.shape, vec![2, 3]);
            assert_eq!(gathered.data.len(), 6);
            let expected = [0.0, 0.0, 1.5, 4.5, 6.5, 15.5];
            for (idx, (actual, expected)) in gathered.data.iter().zip(expected).enumerate() {
                assert!(
                    (*actual - expected).abs() < 1.0e-5,
                    "cumtrapz mismatch at {idx}: actual={actual} expected={expected}"
                );
            }
        });
    }

    #[test]
    fn trapezoid_provider_preserves_complex_storage_and_integrates_lanes() {
        with_wgpu_provider(|provider| {
            let data = [1.0_f32, 1.0, 2.0, 2.0, 3.0, 3.0];
            let input = provider
                .upload_numeric(&HostNumericTensorView {
                    data: HostNumericDataView::F32(&data),
                    shape: &[1, 3],
                    storage: GpuTensorStorage::ComplexInterleaved,
                })
                .expect("upload input");

            let trapz = provider
                .trapz_dim(&input, 1, ProviderTrapezoidSpacing::Unit)
                .expect("trapz")
                .expect("provider accepted numerical operation");
            let cumtrapz = provider
                .cumtrapz_dim(&input, 1, ProviderTrapezoidSpacing::Unit)
                .expect("cumtrapz")
                .expect("provider accepted numerical operation");

            let trapz_host = pollster::block_on(provider.download(&trapz)).expect("download trapz");
            let cumtrapz_host =
                pollster::block_on(provider.download(&cumtrapz)).expect("download cumtrapz");

            for dimension in [2, usize::MAX] {
                for output in [
                    provider.trapz_dim(&input, dimension, ProviderTrapezoidSpacing::Unit),
                    provider.cumtrapz_dim(&input, dimension, ProviderTrapezoidSpacing::Unit),
                ] {
                    let output = output.unwrap().expect("virtual-axis provider route");
                    assert_eq!(output.shape(), input.shape());
                    assert_eq!(
                        runmat_accelerate_api::handle_precision(&output),
                        runmat_accelerate_api::handle_precision(&input)
                    );
                    let host = pollster::block_on(provider.download(&output)).unwrap();
                    assert_eq!(host.storage, GpuTensorStorage::ComplexInterleaved);
                    assert_eq!(host.data, vec![0.; 6]);
                    output.release().unwrap();
                }
            }
            input.release().unwrap();
            trapz.release().unwrap();
            cumtrapz.release().unwrap();

            assert_eq!(trapz_host.storage, GpuTensorStorage::ComplexInterleaved);
            assert_eq!(trapz_host.shape, vec![1, 1]);
            assert_eq!(trapz_host.data, vec![4.0, 4.0]);
            assert_eq!(cumtrapz_host.storage, GpuTensorStorage::ComplexInterleaved);
            assert_eq!(cumtrapz_host.shape, vec![1, 3]);
            assert_eq!(cumtrapz_host.data, vec![0.0, 0.0, 1.5, 1.5, 4.0, 4.0]);
        });
    }
}
