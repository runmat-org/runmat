//! Counted original active signal events; no private hash inventory or growing event buffer.
use super::*;

pub(crate) fn time_breakpoints(
    model: &AnalysisModel,
    step: &ProcedureStep,
    start_time_s: f64,
    end_time_s: f64,
) -> Result<Vec<f64>, RealizationError> {
    time_breakpoints_profiled(
        model,
        step,
        start_time_s,
        end_time_s,
        step.accuracy.resources.resident_byte_budget(),
    )
    .map(|(points, _)| points)
}

/// Return the same original points and their admitted construction workspace.
pub(crate) fn time_breakpoints_profiled(
    model: &AnalysisModel,
    step: &ProcedureStep,
    start_time_s: f64,
    end_time_s: f64,
    maximum_resident_bytes: u64,
) -> Result<(Vec<f64>, u64), RealizationError> {
    use crate::topology::storage::{add, reserve, slots, SortedKeys};
    // Activation checks temporarily own one original object identifier at a time.
    let identifier_bytes = model
        .boundaries
        .iter()
        .map(|v| v.boundary_id.len())
        .chain(model.constraints.iter().map(|v| v.constraint_id().len()))
        .chain(
            model
                .excitations
                .iter()
                .map(|v| v.excitation().excitation_id.0.len()),
        )
        .max()
        .unwrap_or(0) as u64;
    if identifier_bytes > maximum_resident_bytes {
        return Err(RealizationError::resident_ceiling(
            "time event identifier workspace exceeds resident ceiling",
            identifier_bytes,
            maximum_resident_bytes,
        ));
    }
    let actions = super::super::activation::active_actions(model, Some(step))?;
    let mut count = Some(actions.clone().count());
    collect_constraint_signal_ids(model, step, &mut |_| {
        count = count.and_then(|v| v.checked_add(1));
    });
    let count = count.ok_or_else(|| RealizationError::new("active signal count overflow"))?;
    let key_bytes = slots::<&str>(count).map_err(storage_error)?;
    let id_workspace = add(key_bytes, identifier_bytes).map_err(storage_error)?;
    if id_workspace > maximum_resident_bytes {
        return Err(RealizationError::resident_ceiling(
            "time event key workspace exceeds resident ceiling",
            id_workspace,
            maximum_resident_bytes,
        ));
    }
    let mut ids = reserve(count).map_err(storage_error)?;
    ids.extend(actions.map(|a| a.action.excitation().signal_id.0.as_str()));
    collect_constraint_signal_ids(model, step, &mut |id| ids.push(id));
    let active_signal_ids = SortedKeys::from_entries(ids).map_err(storage_error)?;
    let mut count = Some(0usize);
    visit_events(
        model,
        &active_signal_ids,
        start_time_s,
        end_time_s,
        &mut |_| {
            count = count.and_then(|v| v.checked_add(1));
        },
    );
    let count = count.ok_or_else(|| RealizationError::new("time event count overflow"))?;
    let event_workspace =
        add(key_bytes, slots::<f64>(count).map_err(storage_error)?).map_err(storage_error)?;
    if event_workspace > maximum_resident_bytes {
        return Err(RealizationError::resident_ceiling(
            "time event workspace exceeds resident ceiling",
            event_workspace,
            maximum_resident_bytes,
        ));
    }
    let mut events = reserve(count).map_err(storage_error)?;
    visit_events(
        model,
        &active_signal_ids,
        start_time_s,
        end_time_s,
        &mut |value| events.push(value),
    );
    events.sort_unstable_by(f64::total_cmp);
    events.dedup_by(|left, right| left.to_bits() == right.to_bits());
    Ok((events, id_workspace.max(event_workspace)))
}

fn visit_events(
    model: &AnalysisModel,
    active_signal_ids: &crate::topology::storage::SortedKeys<&str>,
    start_time_s: f64,
    end_time_s: f64,
    visitor: &mut impl FnMut(f64),
) {
    macro_rules! collect {
        ($signal:expr) => {
            visit_time_jumps($signal, start_time_s, end_time_s, visitor)
        };
    }
    for signal in &model.signals {
        if !active_signal_ids.contains(&signal.signal_id().0.as_str()) {
            continue;
        }
        match signal {
            PhysicalSignal::Displacement(signal) => collect!(signal),
            PhysicalSignal::Rotation(signal) => collect!(signal),
            PhysicalSignal::Velocity(signal) => collect!(signal),
            PhysicalSignal::Acceleration(signal) => collect!(signal),
            PhysicalSignal::AngularVelocity(signal) => collect!(signal),
            PhysicalSignal::AngularAcceleration(signal) => collect!(signal),
            PhysicalSignal::Force(signal) => collect!(signal),
            PhysicalSignal::Moment(signal) => collect!(signal),
            PhysicalSignal::Traction(signal) => collect!(signal),
            PhysicalSignal::Pressure(signal) => collect!(signal),
            PhysicalSignal::BodyForceDensity(signal) => collect!(signal),
            PhysicalSignal::Temperature(signal) => collect!(signal),
            PhysicalSignal::TemperatureDifference(signal) => collect!(signal),
            PhysicalSignal::HeatFlux(signal) => collect!(signal),
            PhysicalSignal::HeatGenerationDensity(signal) => collect!(signal),
            PhysicalSignal::HeatRate(signal) => collect!(signal),
            PhysicalSignal::ElectricPotential(signal) => collect!(signal),
            PhysicalSignal::ElectricCurrent(signal) => collect!(signal),
            PhysicalSignal::ElectricCharge(signal) => collect!(signal),
            PhysicalSignal::ElectricCurrentDensity(signal) => collect!(signal),
            PhysicalSignal::ElectricChargeDensity(signal) => collect!(signal),
            PhysicalSignal::MagneticVectorPotential(signal) => collect!(signal),
            PhysicalSignal::MassFlowRate(signal) => collect!(signal),
            PhysicalSignal::VolumeFlowRate(signal) => collect!(signal),
        }
    }
}
fn storage_error(error: crate::topology::storage::TopologyStorageError) -> RealizationError {
    RealizationError::new(error.to_string())
}

fn collect_constraint_signal_ids<'a>(
    model: &'a AnalysisModel,
    step: &ProcedureStep,
    ids: &mut impl FnMut(&'a str),
) {
    use runmat_analysis_core::{
        AnalysisConstraint, BoundaryOperator, CoupledPhysicsFamily, FluidConstraintKind,
        PhysicsFamily, StructuralConstraintKind, ThermalConstraintKind,
    };
    let structural = matches!(
        step.family,
        PhysicsFamily::Structural | PhysicsFamily::Coupled(CoupledPhysicsFamily::Thermomechanical)
    );
    let thermal = matches!(
        step.family,
        PhysicsFamily::Thermal
            | PhysicsFamily::Coupled(CoupledPhysicsFamily::Thermomechanical)
            | PhysicsFamily::Coupled(CoupledPhysicsFamily::ConjugateHeatTransfer)
            | PhysicsFamily::Coupled(CoupledPhysicsFamily::Electrothermal)
    );
    let fluid = matches!(
        step.family,
        PhysicsFamily::Fluid
            | PhysicsFamily::Coupled(CoupledPhysicsFamily::ConjugateHeatTransfer)
            | PhysicsFamily::Coupled(CoupledPhysicsFamily::FluidStructureInteraction)
    );
    for boundary in &model.boundaries {
        if fluid
            && step.object_is_active(&runmat_analysis_core::StepObjectRef::Boundary(
                boundary.boundary_id.clone(),
            ))
        {
            if let BoundaryOperator::FluidOutletPressure { value } = &boundary.operator {
                insert_prescribed_id(ids, value);
            }
        }
    }
    for constraint in &model.constraints {
        if !step.object_is_active(&runmat_analysis_core::StepObjectRef::Constraint(
            constraint.constraint_id().to_string(),
        )) {
            continue;
        }
        match constraint {
            AnalysisConstraint::Structural(constraint) if structural => match &constraint.kind {
                StructuralConstraintKind::Prescribed(value) => {
                    for translation in &value.translations {
                        insert_prescribed_id(ids, &translation.value);
                    }
                    for rotation in &value.rotations {
                        insert_prescribed_id(ids, &rotation.value);
                    }
                }
                StructuralConstraintKind::NormalDisplacement { value, .. } => {
                    insert_prescribed_id(ids, value)
                }
                StructuralConstraintKind::PrescribedMotion(motion) => {
                    for translation in &motion.translations {
                        let runmat_analysis_core::StructuralKinematicHistory::AccelerationDriven {
                            acceleration,
                            ..
                        } = &translation.history;
                        ids(acceleration.0 .0.as_str());
                    }
                }
                _ => {}
            },
            AnalysisConstraint::Fluid(constraint) if fluid => match &constraint.kind {
                FluidConstraintKind::PrescribedVelocity { value, .. } => {
                    insert_prescribed_id(ids, value)
                }
                FluidConstraintKind::MovingWall { velocity, .. } => {
                    insert_prescribed_id(ids, velocity)
                }
                FluidConstraintKind::PressureReference { value, .. } => {
                    insert_prescribed_id(ids, value)
                }
                _ => {}
            },
            AnalysisConstraint::Thermal(constraint) if thermal => match &constraint.kind {
                ThermalConstraintKind::Prescribed { value, .. } => insert_prescribed_id(ids, value),
                ThermalConstraintKind::AffineJump {
                    temperature_jump, ..
                } => insert_prescribed_id(ids, temperature_jump),
                _ => {}
            },
            _ => {}
        }
    }
}

fn insert_prescribed_id<'a, Q>(
    ids: &mut impl FnMut(&'a str),
    value: &'a runmat_analysis_core::PrescribedValue<Q>,
) {
    if let runmat_analysis_core::PrescribedValue::Signal(signal_id) = value {
        ids(signal_id.0.as_str());
    }
}

fn visit_time_jumps<Q>(signal: &Signal<Q>, start: f64, end: f64, visitor: &mut impl FnMut(f64)) {
    match &signal.representation {
        SignalRepresentation::Tabulated(table)
            if table.domain == TabulatedSignalDomain::TimeSeconds =>
        {
            for value in table.knots.iter().filter_map(|knot| {
                (knot.abscissa > start
                    && knot.abscissa <= end
                    && matches!(knot.value, SignalKnotValue::Jump { .. }))
                .then_some(knot.abscissa)
            }) {
                visitor(value);
            }
        }
        SignalRepresentation::Waveform(waveform) => {
            for value in super::super::waveform::breakpoints(waveform, start, end) {
                visitor(value);
            }
        }
        _ => {}
    }
}
