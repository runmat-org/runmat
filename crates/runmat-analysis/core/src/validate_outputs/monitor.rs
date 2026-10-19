use std::collections::HashMap;

use crate::{
    AnalysisValidationError, MonitorCadence, MonitorStopCondition, ObservationQuantity,
    ObservationRequest, ObservationSpec, ObservationThreshold, ProcedureStep,
};

use super::{monitor_error, spec::is_scalar_result};

pub(super) fn validate_monitors(
    step: &ProcedureStep,
    ordinary: &HashMap<&str, &ObservationSpec>,
) -> Result<(), AnalysisValidationError> {
    for request in &step.observations {
        let ObservationRequest::Monitor(monitor) = request else {
            continue;
        };
        let id = monitor.observation_id.0.as_str();
        let source_id = monitor.source_observation_id.0.trim();
        if source_id.is_empty() {
            return monitor_error(id, "source observation id must be non-empty");
        }
        if source_id == id {
            return monitor_error(id, "a monitor cannot observe itself");
        }
        let Some(source) = ordinary.get(source_id) else {
            return monitor_error(
                id,
                "source must name a field, history, probe, or reduction in the same step",
            );
        };
        if !is_scalar_result(source) {
            return monitor_error(id, "source observation must produce scalar values");
        }
        match monitor.cadence {
            MonitorCadence::AcceptedIncrements { interval: 0 } => {
                return monitor_error(id, "accepted-increment interval must be nonzero");
            }
            MonitorCadence::WallSeconds { interval }
                if !interval.is_finite() || interval <= 0.0 =>
            {
                return monitor_error(id, "wall-clock interval must be finite and positive");
            }
            _ => {}
        }
        if let Some(stop) = monitor.stop {
            match stop {
                MonitorStopCondition::LessThan { threshold }
                | MonitorStopCondition::GreaterThan { threshold } => {
                    validate_threshold(id, source.quantity, threshold)?;
                }
                MonitorStopCondition::AbsoluteChangeBelow {
                    threshold,
                    consecutive,
                } => {
                    validate_threshold(id, source.quantity, threshold)?;
                    if threshold.canonical_value < 0.0 || consecutive == 0 {
                        return monitor_error(
                            id,
                            "absolute-change threshold must be nonnegative and consecutive count nonzero",
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

fn validate_threshold(
    monitor_id: &str,
    source_quantity: ObservationQuantity,
    threshold: ObservationThreshold,
) -> Result<(), AnalysisValidationError> {
    if !threshold.canonical_value.is_finite() {
        return monitor_error(monitor_id, "stop threshold must be finite");
    }
    if threshold.quantity != source_quantity {
        return monitor_error(
            monitor_id,
            "stop threshold quantity must match the source observation quantity",
        );
    }
    Ok(())
}
