//! Result publication and optional graphics presentation.
use super::*;

pub struct ImhistEvaluation {
    pub(super) counts: Tensor,
    pub(super) locations: Tensor,
    resident_source: Option<runmat_accelerate_api::GpuTensorHandle>,
}

impl ImhistEvaluation {
    pub(super) fn from_counts_locations(
        counts: Vec<f64>,
        locations: Vec<f64>,
    ) -> BuiltinResult<Self> {
        if counts.len() != locations.len() {
            return Err(internal("counts and bin locations length mismatch"));
        }
        let rows = counts.len();
        let counts = Tensor::new(counts, vec![rows, 1])
            .map_err(|err| internal(format!("counts tensor: {err}")))?;
        let locations = Tensor::new(locations, vec![rows, 1])
            .map_err(|err| internal(format!("bin location tensor: {err}")))?;
        Ok(Self {
            counts,
            locations,
            resident_source: None,
        })
    }

    pub(super) fn has_resident_source(&self) -> bool {
        self.resident_source.is_some()
    }

    pub(super) fn with_resident_source(
        mut self,
        source: Option<runmat_accelerate_api::GpuTensorHandle>,
    ) -> Self {
        self.resident_source = source;
        self
    }

    pub(super) fn into_counts_value(self) -> BuiltinResult<Value> {
        let mut outputs = self.publish_values(vec![Value::Tensor(self.counts.clone())])?;
        Ok(outputs.remove(0))
    }

    pub(super) fn into_host_counts_value(self) -> Value {
        Value::Tensor(self.counts)
    }

    pub(super) fn into_outputs(self) -> BuiltinResult<Vec<Value>> {
        self.publish_values(vec![
            Value::Tensor(self.counts.clone()),
            Value::Tensor(self.locations.clone()),
        ])
    }

    fn publish_values(&self, host_values: Vec<Value>) -> BuiltinResult<Vec<Value>> {
        let Some(source) = self.resident_source.as_ref() else {
            return Ok(host_values);
        };
        let mut restored = Vec::with_capacity(host_values.len());
        for host_value in host_values.iter().cloned() {
            let protected = std::iter::once(source.clone())
                .chain(restored.iter().filter_map(|value| match value {
                    Value::GpuTensor(handle) => Some(handle.clone()),
                    _ => None,
                }))
                .collect::<Vec<_>>();
            // The shared boundary authenticates the owner and permits an
            // automatic host result only for typed Unsupported. Operational
            // errors remain terminal, including failure of a later output.
            let output =
                crate::builtins::common::gpu_helpers::restore_class_preserving_value_protecting(
                    source,
                    host_value,
                    NAME,
                    &protected.iter().collect::<Vec<_>>(),
                )?;
            let Value::GpuTensor(handle) = &output else {
                if !runmat_accelerate_api::handle_is_explicit(source) {
                    return Ok(host_values);
                }
                return Err(internal(
                    "provider returned a host value during result publication",
                ));
            };
            if protected
                .iter()
                .any(|protected| same_gpu_handle(handle, protected))
            {
                return Err(internal(
                    "provider aliased a protected handle during result publication",
                ));
            }
            restored.push(output);
        }
        Ok(restored)
    }

    pub(super) fn render_plot(&self) -> BuiltinResult<()> {
        render_imhist_plot(&self.counts, &self.locations)
    }
}

pub(super) fn same_gpu_handle(
    left: &runmat_accelerate_api::GpuTensorHandle,
    right: &runmat_accelerate_api::GpuTensorHandle,
) -> bool {
    left.shares_allocation_with(right)
}

#[cfg(feature = "plot-core")]
fn render_imhist_plot(counts: &Tensor, locations: &Tensor) -> BuiltinResult<()> {
    let plot_data = plot_display_bins(counts, locations)?;
    let mut chart = BarChart::new(plot_data.labels, plot_data.counts).map_err(|cause| {
        let mapped = plot_failed(format!("chart construction failed: {cause}"));
        crate::builtins::plotting::state::with_render_cause(
            mapped,
            build_runtime_error(cause).with_builtin(NAME).build(),
        )
    })?;
    chart.set_bar_width(0.95);
    chart.set_color(glam::Vec4::new(0.1, 0.1, 0.1, 0.95));
    let mut chart = Some(chart);
    let render_result = crate::builtins::plotting::state::render_active_plot(
        NAME,
        crate::builtins::plotting::state::PlotRenderOptions {
            // MATLAB's imhist writes no title or axis labels (DEF-31).
            ..Default::default()
        },
        move |figure, axes| {
            figure.add_bar_chart_on_axes(
                chart.take().expect("imhist chart consumed exactly once"),
                axes,
            );
            Ok(())
        },
    );
    if let Err(err) = render_result {
        if crate::builtins::plotting::state::presentation_unavailable(&err) {
            return Ok(());
        }
        return Err(crate::builtins::plotting::state::with_render_cause(
            plot_failed(err.message()),
            err,
        ));
    }
    Ok(())
}

#[cfg(not(feature = "plot-core"))]
fn render_imhist_plot(_counts: &Tensor, _locations: &Tensor) -> BuiltinResult<()> {
    Ok(())
}

#[cfg(feature = "plot-core")]
fn format_bin_label(value: f64) -> String {
    if (value.round() - value).abs() <= INTEGER_TOL {
        format!("{:.0}", value)
    } else {
        format!("{:.3}", value)
    }
}

#[cfg(feature = "plot-core")]
pub(super) struct PlotDisplayBins {
    pub(super) labels: Vec<String>,
    pub(super) counts: Vec<f64>,
}

#[cfg(feature = "plot-core")]
pub(super) fn plot_display_bins(
    counts: &Tensor,
    locations: &Tensor,
) -> BuiltinResult<PlotDisplayBins> {
    let counts = tensor_utils::tensor_values_f64_cow(counts);
    let locations = tensor_utils::tensor_values_f64_cow(locations);
    if counts.len() != locations.len() {
        return Err(internal("counts and bin locations length mismatch"));
    }
    if counts.is_empty() {
        return Err(internal("histogram has no bins to plot"));
    }
    if counts.len() <= MAX_PLOT_BINS {
        return Ok(PlotDisplayBins {
            labels: locations
                .iter()
                .map(|value| format_bin_label(*value))
                .collect(),
            counts: counts.to_vec(),
        });
    }

    let stride = counts.len().div_ceil(MAX_PLOT_BINS);
    let mut labels = Vec::with_capacity(counts.len().div_ceil(stride));
    let mut display_counts = Vec::with_capacity(labels.capacity());
    for start in (0..counts.len()).step_by(stride) {
        let end = (start + stride).min(counts.len());
        let total = counts[start..end].iter().sum::<f64>();
        let location = 0.5 * (locations[start] + locations[end - 1]);
        labels.push(format_bin_label(location));
        display_counts.push(total);
    }

    Ok(PlotDisplayBins {
        labels,
        counts: display_counts,
    })
}
