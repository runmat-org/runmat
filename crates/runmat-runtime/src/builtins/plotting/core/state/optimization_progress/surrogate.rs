//! Surrogate sample markers and reset annotations reuse the ordinary scene
//! payload/receiver owner. Numerical history stays only in those payloads.
use super::*;
use runmat_builtins::SurrogateSampleOrigin;
use runmat_plot::plots::{reference_line::ReferenceLineOrientation, ReferenceLine};
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(in super::super) enum Feasibility {
    Feasible,
    Infeasible,
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(in super::super) enum Series {
    Best(Feasibility),
    Incumbent(Feasibility),
    Initial(Feasibility),
    Random(Feasibility),
    Adaptive(Feasibility),
    Reset(usize),
    CheckpointResume(usize),
}
pub(crate) struct SurrogatePlotSample {
    pub objective: Option<f64>,
    pub maximum: Option<f64>,
    pub origin: Option<SurrogateSampleOrigin>,
}
pub(crate) struct SurrogatePlotStep {
    pub count: usize,
    pub best: SurrogatePlotSample,
    pub incumbent: SurrogatePlotSample,
    pub current: SurrogatePlotSample,
    pub reset: bool,
    pub resume: bool,
    pub reset_count: usize,
    pub tolerance: f64,
}
impl SurrogatePlotSample {
    fn validate(&self) -> bool {
        !self.objective.is_some_and(f64::is_infinite) && !self.maximum.is_some_and(f64::is_infinite)
    }
    fn measured(&self, tolerance: f64) -> Option<(f64, Feasibility)> {
        if self.maximum.is_some_and(|v| v.is_nan()) {
            return None;
        }
        let value = self.objective.or(self.maximum)?;
        if !value.is_finite() {
            return None;
        }
        Some((
            value,
            if self.maximum.is_none_or(|v| v <= tolerance) {
                Feasibility::Feasible
            } else {
                Feasibility::Infeasible
            },
        ))
    }
}
pub(crate) fn update_surrogate_plot(
    phase: OutputState,
    step: SurrogatePlotStep,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<()> {
    if !step.tolerance.is_finite()
        || step.tolerance < 0.0
        || !step.best.validate()
        || !step.incumbent.validate()
        || !step.current.validate()
        || step.current.measured(step.tolerance).is_some() && step.current.origin.is_none()
        || step.reset && step.reset_count == 0
    {
        return Err(failure("invalid measured surrogate plot generation"));
    }
    admission(runtime)?;
    let _scope = runtime.map(RuntimeContext::enter);
    let resolved = resolve_axes_target(PropertyMutationTarget::alias().with_runtime(runtime), None)
        .map_err(graphics_failure)?;
    if phase == OutputState::Init {
        let (handle, axes) = resolved.coordinates();
        let keys = {
            let reg = resolved
                .authority()
                .registry(PropertyLocation::Axes(handle, axes))
                .map_err(graphics_failure)?;
            reg.figures
                .get(&handle)
                .map(|state| {
                    state
                        .optimization_plots
                        .keys()
                        .filter_map(|(index, kind)| {
                            (*index == axes && matches!(kind, OptimizationSeries::Surrogate(_)))
                                .then_some(*kind)
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default()
        };
        for key in keys {
            update_series_payload(
                key,
                phase,
                OptimizationPlotData::Empty,
                runtime,
                Some(&resolved),
                None,
            )?;
        }
    }
    let best = step.best.measured(step.tolerance);
    let incumbent = step.incumbent.measured(step.tolerance);
    let current = step.current.measured(step.tolerance);
    let show = |value: Option<(f64, Feasibility)>| {
        value.map_or_else(|| "-".to_owned(), |(v, _)| v.to_string())
    };
    let title = format!(
        "Best: {}  Incumbent: {}  Current: {}",
        show(best),
        show(incumbent),
        show(current)
    );
    let annotation = Some(ChartAnnotation {
        title: &title,
        y_label: if step.current.objective.is_none() && step.current.maximum.is_some() {
            "Maximum nonlinear constraint value"
        } else {
            "Objective function"
        },
    });
    let mut published = false;
    for (series, measured) in [
        (best.map(|(_, f)| Series::Best(f)), best),
        (incumbent.map(|(_, f)| Series::Incumbent(f)), incumbent),
        (
            current.and_then(|(_, f)| {
                step.current.origin.map(|origin| match origin {
                    SurrogateSampleOrigin::Initial => Series::Initial(f),
                    SurrogateSampleOrigin::Random => Series::Random(f),
                    SurrogateSampleOrigin::Adaptive => Series::Adaptive(f),
                })
            }),
            current,
        ),
    ] {
        if let (Some(series), Some((value, _))) = (series, measured) {
            update_series_payload(
                OptimizationSeries::Surrogate(series),
                phase,
                OptimizationPlotData::Sample {
                    point: Vec::new(),
                    iteration: step.count as f64,
                    value,
                },
                runtime,
                Some(&resolved),
                annotation,
            )?;
            published = true;
        }
    }
    if !published {
        update_series_payload(
            OptimizationSeries::Surrogate(Series::Best(Feasibility::Feasible)),
            phase,
            OptimizationPlotData::Empty,
            runtime,
            Some(&resolved),
            annotation,
        )?;
    }
    if step.reset {
        let mut line = ReferenceLine::new(ReferenceLineOrientation::Vertical, step.count as f64)
            .map_err(failure)?;
        line.line_style = LineStyle::Dashed;
        line.display_name = Some("Surrogate Reset".to_owned());
        update_series_payload(
            OptimizationSeries::Surrogate(Series::Reset(step.reset_count)),
            phase,
            OptimizationPlotData::Reference(line),
            runtime,
            Some(&resolved),
            annotation,
        )?;
    }
    if step.resume {
        let mut line = ReferenceLine::new(ReferenceLineOrientation::Vertical, step.count as f64)
            .map_err(failure)?;
        line.line_style = LineStyle::Dashed;
        line.display_name = Some("Checkpoint Resume".to_owned());
        update_series_payload(
            OptimizationSeries::Surrogate(Series::CheckpointResume(step.count)),
            phase,
            OptimizationPlotData::Reference(line),
            runtime,
            Some(&resolved),
            annotation,
        )?;
    }
    Ok(())
}
pub(super) fn decorate(series: Series, line: &mut LinePlot) {
    use runmat_plot::plots::{line::LineMarkerAppearance, scatter::MarkerStyle};
    let (feasibility, label, marker) = match series {
        Series::Best(f) => (f, "Best", MarkerStyle::Circle),
        Series::Incumbent(f) => (f, "Incumbent", MarkerStyle::Cross),
        Series::Initial(f) => (f, "Initial Samples", MarkerStyle::Diamond),
        Series::Random(f) => (f, "Random Samples", MarkerStyle::Triangle),
        Series::Adaptive(f) => (f, "Adaptive Samples", MarkerStyle::Circle),
        Series::Reset(_) | Series::CheckpointResume(_) => return,
    };
    line.label = Some(if feasibility == Feasibility::Feasible {
        label.to_owned()
    } else {
        format!("Infeasible {label}")
    });
    line.line_style = LineStyle::None;
    line.color = if feasibility == Feasibility::Feasible {
        glam::Vec4::new(0.0, 0.447, 0.741, 1.0)
    } else {
        glam::Vec4::new(0.850, 0.325, 0.098, 1.0)
    };
    line.set_marker(Some(LineMarkerAppearance {
        kind: marker,
        size: if matches!(series, Series::Adaptive(_)) {
            3.0
        } else {
            5.0
        },
        edge_color: line.color,
        face_color: line.color,
        filled: matches!(series, Series::Adaptive(_)),
    }));
}
