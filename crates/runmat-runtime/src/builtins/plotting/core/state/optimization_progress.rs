//! Optimization charts retain data in ordinary scene payloads, not solver state.
use super::*;
use crate::builtins::math::optim::common::output_callback::OutputState;
use crate::context::RuntimeContext;
use runmat_builtins::OptimPlotKind;
use runmat_builtins::{OptimGeneticPlotKind, OptimGeneticPlotSeries};
use runmat_plot::plots::{BarChart, LinePlot};
#[path = "optimization_progress/genetic.rs"]
mod genetic;
pub(crate) use genetic::update_genetic_plot;
#[path = "optimization_progress/payload.rs"]
mod payload;
pub(crate) use payload::OptimizationPlotData as GeneticPlotData;
use payload::OptimizationPlotData;
#[path = "optimization_progress/metrics.rs"]
mod metrics;
pub(crate) use metrics::{update_constrained_metric, ConstrainedPlotSample};
#[path = "optimization_progress/surrogate.rs"]
mod surrogate;
pub(crate) use surrogate::{update_surrogate_plot, SurrogatePlotSample, SurrogatePlotStep};
#[derive(Clone, Copy)]
struct ChartAnnotation<'a> {
    title: &'a str,
    y_label: &'static str,
}

#[path = "optimization_progress/series.rs"]
mod series;
pub(super) use series::OptimizationSeries;

enum Update {
    Point(Vec<String>, Vec<f64>),
    History(Vec<f64>, Vec<f64>),
    Reference(runmat_plot::plots::ReferenceLine),
}
impl Update {
    fn create(self) -> Result<PlotElement, String> {
        match self {
            Self::Point(labels, values) => BarChart::new(labels, values).map(PlotElement::Bar),
            Self::History(x, y) => LinePlot::new(x, y).map(PlotElement::Line),
            Self::Reference(line) => Ok(PlotElement::ReferenceLine(line)),
        }
    }
    fn apply(self, plot: &mut PlotElement) -> Result<(), String> {
        match (self, plot) {
            (Self::Point(labels, values), PlotElement::Bar(bar)) => bar.set_data(labels, values),
            (Self::History(x, y), PlotElement::Line(line)) => line.update_data(x, y),
            (Self::Reference(value), PlotElement::ReferenceLine(line)) => {
                *line = value;
                Ok(())
            }
            _ => Err("optimization chart type changed".into()),
        }
    }
}

/// Only object authority crosses callbacks. There is no numerical plot history
/// or callback payload in the solver's retained layout.
pub(crate) struct OptimizationPlotLayout {
    figure: GraphicsReceiverRef,
    axes: Vec<GraphicsReceiverRef>,
}
impl OptimizationPlotLayout {
    pub(crate) fn prepare(count: usize, runtime: Option<&RuntimeContext>) -> BuiltinResult<Self> {
        admission(runtime)?;
        if count == 0 {
            return Err(failure("a plot layout requires at least one callback"));
        }
        let mut axes = runmat_value::exact_capacity_vec(count).map_err(|source| {
            crate::build_runtime_error("optimization plot layout allocation failed")
                .with_source(source)
                .build()
        })?;
        // Existing synchronous graphics APIs retain the supplied authority only
        // for their own operation; no ambient guard can cross a callback await.
        let _scope = runtime.map(RuntimeContext::enter);
        let handle = new_figure_handle();
        configure_subplot(count, 1, 0).map_err(graphics_failure)?;
        let figure = figure_graphics_receiver(handle)
            .map_err(graphics_failure)?
            .ok_or_else(|| failure("new plot figure has no receiver"))?;
        for index in 0..count {
            axes.push(
                axes_graphics_receiver(handle, index)
                    .map_err(graphics_failure)?
                    .ok_or_else(|| failure("new plot axes has no receiver"))?,
            );
        }
        Ok(Self { figure, axes })
    }
    pub(crate) fn select(
        &self,
        index: usize,
        runtime: Option<&RuntimeContext>,
    ) -> BuiltinResult<()> {
        admission(runtime)?;
        let _scope = runtime.map(RuntimeContext::enter);
        if !graphics_receiver_valid(&self.figure) {
            return Err(failure("optimization plot figure was closed or retired"));
        }
        let receiver = self
            .axes
            .get(index)
            .ok_or_else(|| failure("plot callback index is outside its layout"))?;
        let selection =
            GraphicsPropertySelection::capture(receiver.clone()).map_err(graphics_failure)?;
        let (handle, index) = match receiver.kind() {
            GraphicsReceiverKind::Axes { figure, index } => (FigureHandle::from(figure), index),
            _ => return Err(failure("optimization plot target is not an axes receiver")),
        };
        let property = PropertyMutationTarget::selected(selection).with_runtime(runtime);
        select_current_figure_if_exists_for_property(&property, handle)
            .map_err(graphics_failure)?;
        select_axes_for_figure_for_property(&property, handle, index).map_err(graphics_failure)
    }
}

pub(crate) fn update_optimization_plot(
    kind: OptimPlotKind,
    phase: OutputState,
    point: Vec<f64>,
    iteration: f64,
    value: impl Into<Option<f64>>,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<()> {
    let data = match value.into() {
        Some(value) => OptimizationPlotData::Sample {
            point,
            iteration,
            value,
        },
        None => OptimizationPlotData::Empty,
    };
    update_series_payload(kind.into(), phase, data, runtime, None, None)
}
pub(crate) fn update_constrained_optimization_plot(
    phase: OutputState,
    iteration: f64,
    value: f64,
    violation: f64,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<()> {
    update_constrained_metric(
        phase,
        iteration,
        ConstrainedPlotSample {
            objective: Some(value),
            maximum: Some(violation),
            tolerance: 1e-6,
            feasible_count: None,
        },
        runtime,
    )
}
pub(crate) fn update_milp_plot(
    phase: OutputState,
    time: f64,
    primal: f64,
    dual: f64,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<()> {
    admission(runtime)?;
    let _scope = runtime.map(RuntimeContext::enter);
    let resolved = resolve_axes_target(PropertyMutationTarget::alias().with_runtime(runtime), None)
        .map_err(graphics_failure)?;
    update_series(
        OptimizationSeries::MilpPrimal,
        phase,
        Vec::new(),
        time,
        primal,
        runtime,
        Some(&resolved),
    )?;
    update_series(
        OptimizationSeries::MilpDual,
        phase,
        Vec::new(),
        time,
        dual,
        runtime,
        Some(&resolved),
    )
}
fn update_series(
    kind: OptimizationSeries,
    phase: OutputState,
    point: Vec<f64>,
    iteration: f64,
    value: f64,
    runtime: Option<&RuntimeContext>,
    target: Option<&ResolvedAxesTarget>,
) -> BuiltinResult<()> {
    update_series_payload(
        kind,
        phase,
        GeneticPlotData::Sample {
            point,
            iteration,
            value,
        },
        runtime,
        target,
        None,
    )
}
fn update_series_payload(
    kind: OptimizationSeries,
    phase: OutputState,
    data: GeneticPlotData,
    runtime: Option<&RuntimeContext>,
    target: Option<&ResolvedAxesTarget>,
    annotation: Option<ChartAnnotation<'_>>,
) -> BuiltinResult<()> {
    admission(runtime)?;
    let _scope = runtime.map(RuntimeContext::enter);
    let fallback;
    let resolved = match target {
        Some(target) => target,
        None => {
            fallback =
                resolve_axes_target(PropertyMutationTarget::alias().with_runtime(runtime), None)
                    .map_err(graphics_failure)?;
            &fallback
        }
    };
    let (handle, axes) = resolved.coordinates();
    let property = resolved.authority();
    let rendering_disabled = interactive_rendering_disabled();
    let host_managed = host_managed_rendering_enabled();
    let (snapshot, figure_receiver) = {
        let mut reg = property
            .registry(PropertyLocation::Axes(handle, axes))
            .map_err(graphics_failure)?;
        let slot = reg
            .figures
            .get(&handle)
            .and_then(|state| state.optimization_plots.get(&(axes, kind)))
            .cloned();
        let existing = slot.and_then(|receiver| {
            let GraphicsReceiverKind::PlotChild { id } = receiver.kind() else {
                return None;
            };
            let entry = reg.plot_children.entries.get(&id)?;
            if !entry.receiver.as_ref()?.shares_lease(&receiver)
                || entry.state.figure_axes() != (handle, axes)
            {
                return None;
            }
            let right_kind = if matches!(
                kind,
                OptimizationSeries::Surrogate(
                    surrogate::Series::Reset(_) | surrogate::Series::CheckpointResume(_)
                )
            ) {
                matches!(entry.state.as_ref(), PlotChildHandleState::ReferenceLine(_))
            } else if kind.is_bar() {
                matches!(entry.state.as_ref(), PlotChildHandleState::Bar(_))
            } else {
                matches!(entry.state.as_ref(), PlotChildHandleState::Line(_))
            };
            right_kind.then(|| entry.state.plot_index()).flatten()
        });
        // Prepare the replacement from only the selected ordinary payload.
        // Immutable scene snapshots keep their own predecessor buffers alive.
        let update = match data {
            GeneticPlotData::Empty => None,
            GeneticPlotData::Reference(value) => Some(Update::Reference(value)),
            GeneticPlotData::Bars { labels, values } => Some(Update::Point(labels, values)),
            GeneticPlotData::Edges { generation, x, y } => Some(genetic::edges(
                &reg, handle, existing, phase, generation, x, y,
            )?),
            GeneticPlotData::Sample {
                point,
                iteration,
                value,
            } => Some(match kind {
                OptimizationSeries::Standard(chart) if chart.is_point() => {
                    let labels = if chart == OptimPlotKind::AnnealStopping {
                        [
                            "FunctionTolerance",
                            "MaxFunctionEvaluations",
                            "MaxIterations",
                            "MaxTime",
                        ]
                        .into_iter()
                        .map(str::to_owned)
                        .collect()
                    } else {
                        (1..=point.len()).map(|index| index.to_string()).collect()
                    };
                    Update::Point(labels, point)
                }
                _ => {
                    let previous = if phase == OutputState::Init {
                        None
                    } else {
                        existing
                            .and_then(|index| {
                                match reg.figures.get(&handle)?.figure.plots().nth(index)? {
                                    PlotElement::Line(line) => Some(line.host_xy_f64()),
                                    _ => None,
                                }
                            })
                            .transpose()
                            .map_err(failure)?
                            .flatten()
                    };
                    let (mut iterations, mut values) = previous.unwrap_or_default();
                    // The scene payload is the only retained best-feasible history.
                    // Updating the same iteration excludes its predecessor value.
                    let value = if kind == OptimizationSeries::NonlinearBest {
                        let retained = values
                            .iter()
                            .zip(&iterations)
                            .filter(|(_, old_iteration)| **old_iteration != iteration)
                            .map(|(v, _)| *v)
                            .filter(|v| v.is_finite())
                            .reduce(f64::min);
                        match (retained, value.is_finite()) {
                            (Some(old), true) => old.min(value),
                            (Some(old), false) => old,
                            (None, _) => value,
                        }
                    } else {
                        value
                    };
                    if iterations.last() == Some(&iteration) {
                        if let Some(last) = values.last_mut() {
                            *last = value;
                        }
                    } else {
                        iterations.push(iteration);
                        values.push(value);
                    }
                    Update::History(iterations, values)
                }
            }),
        };
        if let Some(update) = update {
            if let Some(index) = existing {
                let state = get_state_mut(&mut reg, handle);
                update
                    .apply(
                        state
                            .figure
                            .get_plot_mut(index)
                            .ok_or_else(|| failure("optimization chart payload was retired"))?,
                    )
                    .map_err(failure)?;
            } else {
                let mut payload = update.create().map_err(failure)?;
                if let PlotElement::Line(line) = &mut payload {
                    match kind {
                        OptimizationSeries::MilpPrimal => {
                            line.label = Some("Primal Bound".into());
                            line.color = glam::Vec4::new(0.0, 0.447, 0.741, 1.0);
                        }
                        OptimizationSeries::MilpDual => {
                            line.label = Some("Dual Bound".into());
                            line.color = glam::Vec4::new(0.850, 0.325, 0.098, 1.0);
                        }
                        OptimizationSeries::NonlinearFeasible => {
                            line.label = Some("Feasible".into());
                            line.color = glam::Vec4::new(0.0, 0.447, 0.741, 1.0);
                        }
                        OptimizationSeries::NonlinearInfeasible => {
                            line.label = Some("Infeasible".into());
                            line.color = glam::Vec4::new(0.850, 0.325, 0.098, 1.0);
                        }
                        OptimizationSeries::NonlinearBest => {
                            line.label = Some("Best feasible".into());
                            line.color = glam::Vec4::new(0.466, 0.674, 0.188, 1.0);
                        }
                        OptimizationSeries::Standard(_) => {}
                        OptimizationSeries::Surrogate(series) => surrogate::decorate(series, line),
                        OptimizationSeries::Genetic(kind, series) => {
                            line.label = series.label(kind).map(str::to_owned);
                            if kind == OptimGeneticPlotKind::Genealogy {
                                line.set_marker(Some(
                                    runmat_plot::plots::line::LineMarkerAppearance {
                                        kind: runmat_plot::plots::scatter::MarkerStyle::Circle,
                                        size: 4.,
                                        edge_color: line.color,
                                        face_color: line.color,
                                        filled: true,
                                    },
                                ));
                            }
                        }
                    }
                }
                let receiver =
                    graphics_receiver::issue_graphics_receiver(GraphicsReceiverKind::PlotChild {
                        id: reg.next_plot_child_handle,
                    })
                    .map_err(graphics_failure)?;
                let next = reg
                    .next_plot_child_handle
                    .checked_add(1)
                    .ok_or_else(|| failure("plot child identity exhausted"))?;
                let id = reg.next_plot_child_handle;
                let state = get_state_mut(&mut reg, handle);
                let index = state.figure.add_plot_element_on_axes(payload, axes);
                state
                    .optimization_plots
                    .insert((axes, kind), receiver.clone());
                let child = SimplePlotHandleState {
                    figure: handle,
                    axes_index: axes,
                    plot_index: index,
                };
                reg.plot_children.insert(
                    id,
                    match kind {
                        OptimizationSeries::Surrogate(
                            surrogate::Series::Reset(_) | surrogate::Series::CheckpointResume(_),
                        ) => PlotChildHandleState::ReferenceLine(child),
                        chart if chart.is_bar() => PlotChildHandleState::Bar(child),
                        _ => PlotChildHandleState::Line(child),
                    },
                );
                reg.plot_children
                    .entries
                    .get_mut(&id)
                    .expect("just inserted")
                    .receiver = Some(receiver);
                reg.next_plot_child_handle = next;
            }
        } else if phase == OutputState::Init {
            // Reuse the canonical scene-index compaction owner, retiring only
            // this exact chart. Other helpers on the axes remain unchanged.
            if let Some(index) = existing {
                get_state_mut(&mut reg, handle)
                    .optimization_plots
                    .remove(&(axes, kind));
                super::clear_axes::clear_registered_plot(&mut reg, handle, index)
                    .map_err(failure)?;
            }
        }
        let state = get_state_mut(&mut reg, handle);
        if !matches!(kind, OptimizationSeries::Standard(_)) {
            state.figure.set_axes_legend_enabled(axes, true);
        }
        state
            .figure
            .set_axes_title(axes, annotation.map_or(kind.title(), |v| v.title));
        state.figure.set_axes_labels(
            axes,
            kind.x_label(),
            annotation.map_or(kind.y_label(), |v| v.y_label),
        );
        state.revision = state.revision.wrapping_add(1);
        (state.figure.owned_clone(), state.figure_receiver.clone())
    };
    property
        .notify(handle, &snapshot, FigureEventKind::Updated)
        .map_err(graphics_failure)?;
    if let (Some(runtime), Some(receiver)) = (runtime, figure_receiver) {
        if let Some(service) = runtime.service_ports().graphics_presentation() {
            if service.stage_scene(receiver)?
                == crate::context::GraphicsSceneStaging::DeferredToFlush
            {
                return Ok(());
            }
        }
    }
    present_figure_update_with_staging(
        kind.name(),
        handle,
        snapshot,
        rendering_disabled,
        host_managed,
        runtime.is_none(),
    )?;
    Ok(())
}
fn admission(runtime: Option<&RuntimeContext>) -> BuiltinResult<()> {
    crate::context::host_allocation::require_unbounded_workspace(runtime, "PlotFcns",
        "finite host-memory execution is unsupported; progress chart history and scene allocations have no complete resource bound")
}
fn failure(detail: impl std::fmt::Display) -> RuntimeError {
    crate::build_runtime_error(format!("optimization plot: {detail}"))
        .with_identifier("RunMat:graphics:OptimizationPlot")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}
fn graphics_failure(source: impl crate::runtime_error::RuntimeErrorSourceBound) -> RuntimeError {
    crate::build_runtime_error("optimization plot target or scene update failed")
        .with_identifier("RunMat:graphics:OptimizationPlot")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .with_source(source)
        .build()
}
#[cfg(test)]
#[path = "optimization_progress/tests.rs"]
mod tests;
