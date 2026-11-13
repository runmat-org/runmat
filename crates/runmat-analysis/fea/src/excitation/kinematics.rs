use std::collections::{BTreeMap, BTreeSet};

use runmat_analysis_core::{
    AnalysisModel, CoordinateFrame, PhysicalSignal, ProcedureStep, SignalExtrapolation,
    SignalInterpolation, SignalKnot, SignalKnotValue, SignalOrdinate, SignalPointContinuity,
    SignalPrecision, SignalRepresentation, SignalSampleApproach, SignalShape, SignalValue,
    StructuralConstraintKind, StructuralKinematicHistory, StructuralMotionReconstruction,
    StructuralTranslationDof, TabulatedSignalDomain,
};
use thiserror::Error;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct StructuralKinematicSample {
    pub(crate) displacement_m: f64,
    pub(crate) velocity_m_per_s:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Velocity, f64>,
    pub(crate) acceleration_m_per_s2: f64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct StructuralMotionSet {
    trajectories: BTreeMap<(String, StructuralTranslationDof), AccelerationTrajectory>,
    breakpoints_s: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
struct AccelerationTrajectory {
    knots: Vec<IntegratedKnot>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct IntegratedKnot {
    time_s: runmat_types::physical::Time,
    acceleration_left_m_per_s2: f64,
    acceleration_at_m_per_s2: f64,
    acceleration_right_m_per_s2: f64,
    displacement_m: f64,
    velocity_m_per_s: runmat_types::physical::Quantity<runmat_types::physical::kind::Velocity, f64>,
}

#[derive(Clone, Debug, Error, PartialEq)]
pub(crate) enum StructuralMotionError {
    #[error("prescribed support motion `{constraint_id}` is not active in the transient step")]
    InactiveConstraint { constraint_id: String },
    #[error("prescribed support motion `{constraint_id}` duplicates component {dof:?}")]
    DuplicateComponent {
        constraint_id: String,
        dof: StructuralTranslationDof,
    },
    #[error(
        "prescribed support motion `{constraint_id}` references a missing acceleration signal"
    )]
    MissingSignal { constraint_id: String },
    #[error(
        "prescribed support motion `{constraint_id}` requires a real64 three-component time-domain acceleration history with linear interpolation, reject extrapolation, and at least two knots"
    )]
    InvalidSignal { constraint_id: String },
    #[error(
        "prescribed support motion `{constraint_id}` must cover exactly [{start_time_s}, {end_time_s}] s", start_time_s = .start_time_s.canonical_value(), end_time_s = .end_time_s.canonical_value()
    )]
    IncompleteCoverage {
        constraint_id: String,
        start_time_s: runmat_types::physical::Time,
        end_time_s: runmat_types::physical::Time,
    },
    #[error("prescribed support motion `{constraint_id}` contains invalid initial kinematics")]
    InvalidInitialKinematics { constraint_id: String },
    #[error("prescribed support motion sample is outside its materialized time domain")]
    SampleOutsideDomain,
}

impl StructuralMotionSet {
    pub(crate) fn materialize(
        model: &AnalysisModel,
        step: &ProcedureStep,
        start_time_s: f64,
        end_time_s: f64,
    ) -> Result<Self, StructuralMotionError> {
        let mut trajectories = BTreeMap::new();
        let mut breakpoints = BTreeSet::new();
        for constraint in model
            .constraints
            .iter()
            .filter_map(|value| value.structural())
        {
            let StructuralConstraintKind::PrescribedMotion(motion) = &constraint.kind else {
                continue;
            };
            if !step.object_is_active(&runmat_analysis_core::StepObjectRef::Constraint(
                constraint.constraint_id.clone(),
            )) {
                return Err(StructuralMotionError::InactiveConstraint {
                    constraint_id: constraint.constraint_id.clone(),
                });
            }
            for translation in &motion.translations {
                let StructuralKinematicHistory::AccelerationDriven {
                    acceleration,
                    initial_displacement,
                    initial_velocity,
                    reconstruction: StructuralMotionReconstruction::PiecewiseLinearAccelerationExact,
                } = &translation.history;
                if !initial_displacement.is_finite() || !initial_velocity.is_finite() {
                    return Err(StructuralMotionError::InvalidInitialKinematics {
                        constraint_id: constraint.constraint_id.clone(),
                    });
                }
                let Some(PhysicalSignal::Acceleration(signal)) = model
                    .signals
                    .iter()
                    .find(|signal| signal.signal_id() == &acceleration.0)
                else {
                    return Err(StructuralMotionError::MissingSignal {
                        constraint_id: constraint.constraint_id.clone(),
                    });
                };
                let trajectory = AccelerationTrajectory::materialize(
                    signal,
                    translation.dof,
                    start_time_s,
                    end_time_s,
                    initial_displacement.canonical_value(),
                    initial_velocity.canonical_value(),
                    &constraint.constraint_id,
                )?;
                for knot in &trajectory.knots {
                    let Some(time) = OrderedTime::new(knot.time_s.canonical_value()) else {
                        return Err(invalid_signal(&constraint.constraint_id));
                    };
                    breakpoints.insert(time);
                }
                if trajectories
                    .insert(
                        (constraint.constraint_id.clone(), translation.dof),
                        trajectory,
                    )
                    .is_some()
                {
                    return Err(StructuralMotionError::DuplicateComponent {
                        constraint_id: constraint.constraint_id.clone(),
                        dof: translation.dof,
                    });
                }
            }
        }
        Ok(Self {
            trajectories,
            breakpoints_s: breakpoints.into_iter().map(|time| time.0).collect(),
        })
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.trajectories.is_empty()
    }

    pub(crate) fn breakpoints_s(&self) -> &[f64] {
        &self.breakpoints_s
    }

    pub(crate) fn has_constraint(&self, constraint_id: &str) -> bool {
        self.trajectories
            .keys()
            .any(|(candidate, _)| candidate == constraint_id)
    }

    pub(crate) fn sample(
        &self,
        constraint_id: &str,
        dof: StructuralTranslationDof,
        time_s: f64,
        approach: SignalSampleApproach,
    ) -> Result<Option<StructuralKinematicSample>, StructuralMotionError> {
        self.trajectories
            .get(&(constraint_id.to_string(), dof))
            .map(|trajectory| trajectory.sample(time_s, approach))
            .transpose()
    }
}

impl AccelerationTrajectory {
    fn materialize(
        signal: &runmat_analysis_core::Signal<runmat_analysis_core::Acceleration>,
        dof: StructuralTranslationDof,
        start_time_s: f64,
        end_time_s: f64,
        initial_displacement_m: f64,
        initial_velocity_m_per_s: f64,
        constraint_id: &str,
    ) -> Result<Self, StructuralMotionError> {
        let SignalRepresentation::Tabulated(table) = &signal.representation else {
            return Err(invalid_signal(constraint_id));
        };
        if signal.shape != SignalShape::Vector3
            || signal.frame.as_ref() != Some(&CoordinateFrame::Global)
            || signal.precision != SignalPrecision::Real64
            || table.domain != TabulatedSignalDomain::TimeSeconds
            || table.interpolation != SignalInterpolation::Linear
            || table.extrapolation != SignalExtrapolation::Reject
            || table.knots.len() < 2
        {
            return Err(invalid_signal(constraint_id));
        }
        if table.knots.first().map(|knot| knot.abscissa) != Some(start_time_s)
            || table.knots.last().map(|knot| knot.abscissa) != Some(end_time_s)
        {
            return Err(StructuralMotionError::IncompleteCoverage {
                constraint_id: constraint_id.into(),
                start_time_s: runmat_types::physical::Time::new(start_time_s),
                end_time_s: runmat_types::physical::Time::new(end_time_s),
            });
        }
        let mut knots: Vec<IntegratedKnot> = Vec::with_capacity(table.knots.len());
        let mut displacement = initial_displacement_m;
        let mut velocity = initial_velocity_m_per_s;
        for (index, knot) in table.knots.iter().enumerate() {
            let (left, at, right) = acceleration_values(knot, dof, constraint_id)?;
            if index > 0 {
                let previous = knots[index - 1];
                let dt = knot.abscissa - previous.time_s.canonical_value();
                if !dt.is_finite() || dt <= 0.0 {
                    return Err(invalid_signal(constraint_id));
                }
                let jerk = (left - previous.acceleration_right_m_per_s2) / dt;
                displacement += velocity * dt
                    + 0.5 * previous.acceleration_right_m_per_s2 * dt * dt
                    + jerk * dt * dt * dt / 6.0;
                velocity += previous.acceleration_right_m_per_s2 * dt + 0.5 * jerk * dt * dt;
            }
            if [left, at, right, displacement, velocity]
                .iter()
                .any(|value| !value.is_finite())
            {
                return Err(invalid_signal(constraint_id));
            }
            knots.push(IntegratedKnot {
                time_s: runmat_types::physical::Time::new(knot.abscissa),
                acceleration_left_m_per_s2: left,
                acceleration_at_m_per_s2: at,
                acceleration_right_m_per_s2: right,
                displacement_m: displacement,
                velocity_m_per_s: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Velocity,
                    _,
                >::new(velocity),
            });
        }
        Ok(Self { knots })
    }

    fn sample(
        &self,
        time_s: f64,
        approach: SignalSampleApproach,
    ) -> Result<StructuralKinematicSample, StructuralMotionError> {
        if !time_s.is_finite() {
            return Err(StructuralMotionError::SampleOutsideDomain);
        }
        if let Some(knot) = self
            .knots
            .iter()
            .find(|knot| knot.time_s.canonical_value() == time_s)
        {
            let acceleration = match approach {
                SignalSampleApproach::At => knot.acceleration_at_m_per_s2,
                SignalSampleApproach::LeftLimit => knot.acceleration_left_m_per_s2,
                SignalSampleApproach::RightLimit => knot.acceleration_right_m_per_s2,
            };
            return Ok(StructuralKinematicSample {
                displacement_m: knot.displacement_m,
                velocity_m_per_s: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Velocity,
                    _,
                >::new(knot.velocity_m_per_s.canonical_value()),
                acceleration_m_per_s2: acceleration,
            });
        }
        let Some(pair) = self.knots.windows(2).find(|pair| {
            pair[0].time_s.canonical_value() < time_s && time_s < pair[1].time_s.canonical_value()
        }) else {
            return Err(StructuralMotionError::SampleOutsideDomain);
        };
        let dt = time_s - pair[0].time_s.canonical_value();
        let span = pair[1].time_s.canonical_value() - pair[0].time_s.canonical_value();
        let jerk =
            (pair[1].acceleration_left_m_per_s2 - pair[0].acceleration_right_m_per_s2) / span;
        Ok(StructuralKinematicSample {
            displacement_m: pair[0].displacement_m
                + pair[0].velocity_m_per_s.canonical_value() * dt
                + 0.5 * pair[0].acceleration_right_m_per_s2 * dt * dt
                + jerk * dt * dt * dt / 6.0,
            velocity_m_per_s: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Velocity,
                _,
            >::new(
                pair[0].velocity_m_per_s.canonical_value()
                    + pair[0].acceleration_right_m_per_s2 * dt
                    + 0.5 * jerk * dt * dt,
            ),
            acceleration_m_per_s2: pair[0].acceleration_right_m_per_s2 + jerk * dt,
        })
    }
}

fn acceleration_values(
    knot: &SignalKnot<runmat_analysis_core::Acceleration>,
    dof: StructuralTranslationDof,
    constraint_id: &str,
) -> Result<(f64, f64, f64), StructuralMotionError> {
    match &knot.value {
        SignalKnotValue::Continuous(value) => {
            let value = acceleration_ordinate(value, dof, constraint_id)?;
            Ok((value, value, value))
        }
        SignalKnotValue::Jump {
            left_limit,
            right_limit,
            continuity,
        } => {
            let left = acceleration_ordinate(left_limit, dof, constraint_id)?;
            let right = acceleration_ordinate(right_limit, dof, constraint_id)?;
            let at = match continuity {
                SignalPointContinuity::Left => left,
                SignalPointContinuity::Right => right,
            };
            Ok((left, at, right))
        }
    }
}

fn acceleration_ordinate(
    ordinate: &SignalOrdinate<runmat_analysis_core::Acceleration>,
    dof: StructuralTranslationDof,
    constraint_id: &str,
) -> Result<f64, StructuralMotionError> {
    let SignalOrdinate::Real(SignalValue::Vector3(values)) = ordinate else {
        return Err(invalid_signal(constraint_id));
    };
    let index = match dof {
        StructuralTranslationDof::Ux => 0,
        StructuralTranslationDof::Uy => 1,
        StructuralTranslationDof::Uz => 2,
    };
    Ok(values[index].canonical_value())
}

fn invalid_signal(constraint_id: &str) -> StructuralMotionError {
    StructuralMotionError::InvalidSignal {
        constraint_id: constraint_id.into(),
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct OrderedTime(f64);

impl OrderedTime {
    fn new(value: f64) -> Option<Self> {
        value.is_finite().then_some(Self(value))
    }
}

impl Eq for OrderedTime {}

impl PartialOrd for OrderedTime {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedTime {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

#[cfg(test)]
mod tests {
    use runmat_analysis_core::{
        Acceleration, Signal, SignalApproximationEvidence, SignalExtrapolation, SignalId,
        SignalPrecision, SignalProvenance, SignalShape, TabulatedSignal,
    };

    use super::*;

    fn acceleration_signal(knots: Vec<SignalKnot<Acceleration>>) -> Signal<Acceleration> {
        Signal {
            signal_id: SignalId("support-acceleration".into()),
            shape: SignalShape::Vector3,
            frame: Some(runmat_analysis_core::CoordinateFrame::Global),
            representation: SignalRepresentation::Tabulated(TabulatedSignal {
                domain: TabulatedSignalDomain::TimeSeconds,
                knots,
                interpolation: SignalInterpolation::Linear,
                extrapolation: SignalExtrapolation::Reject,
            }),
            precision: SignalPrecision::Real64,
            evidence: SignalApproximationEvidence {
                max_absolute_error: None,
                max_relative_error: None,
                derivative_max_absolute_error: None,
                sample_count: 2,
            },
            provenance: SignalProvenance::Authored,
        }
    }

    fn real(value: f64) -> SignalOrdinate<Acceleration> {
        SignalOrdinate::Real(SignalValue::Vector3([
            Acceleration::new(value),
            Acceleration::new(10.0 * value),
            Acceleration::new(100.0 * value),
        ]))
    }

    #[test]
    fn piecewise_linear_acceleration_is_integrated_exactly() {
        let signal = acceleration_signal(vec![
            SignalKnot::continuous(0.0, real(1.0)),
            SignalKnot::continuous(2.0, real(3.0)),
        ]);
        let trajectory = AccelerationTrajectory::materialize(
            &signal,
            StructuralTranslationDof::Ux,
            0.0,
            2.0,
            4.0,
            5.0,
            "support",
        )
        .unwrap();
        let midpoint = trajectory.sample(1.0, SignalSampleApproach::At).unwrap();
        assert!((midpoint.acceleration_m_per_s2 - 2.0).abs() <= 1.0e-14);
        assert!((midpoint.velocity_m_per_s.canonical_value() - 6.5).abs() <= 1.0e-14);
        assert!((midpoint.displacement_m - (4.0 + 5.0 + 0.5 + 1.0 / 6.0)).abs() <= 1.0e-14);
        let end = trajectory.sample(2.0, SignalSampleApproach::At).unwrap();
        assert!((end.velocity_m_per_s.canonical_value() - 9.0).abs() <= 1.0e-14);
        assert!((end.displacement_m - (4.0 + 10.0 + 2.0 + 4.0 / 3.0)).abs() <= 1.0e-14);
    }

    #[test]
    fn acceleration_jump_changes_only_the_selected_point_value() {
        let signal = acceleration_signal(vec![
            SignalKnot::continuous(0.0, real(0.0)),
            SignalKnot::jump(1.0, real(1.0), real(3.0), SignalPointContinuity::Left),
            SignalKnot::continuous(2.0, real(4.0)),
        ]);
        let trajectory = AccelerationTrajectory::materialize(
            &signal,
            StructuralTranslationDof::Ux,
            0.0,
            2.0,
            0.0,
            0.0,
            "support",
        )
        .unwrap();
        let left = trajectory
            .sample(1.0, SignalSampleApproach::LeftLimit)
            .unwrap();
        let at = trajectory.sample(1.0, SignalSampleApproach::At).unwrap();
        let right = trajectory
            .sample(1.0, SignalSampleApproach::RightLimit)
            .unwrap();
        assert_eq!(left.displacement_m, at.displacement_m);
        assert_eq!(
            left.velocity_m_per_s.canonical_value(),
            at.velocity_m_per_s.canonical_value()
        );
        assert_eq!(right.displacement_m, at.displacement_m);
        assert_eq!(
            right.velocity_m_per_s.canonical_value(),
            at.velocity_m_per_s.canonical_value()
        );
        assert_eq!(left.acceleration_m_per_s2, 1.0);
        assert_eq!(at.acceleration_m_per_s2, 1.0);
        assert_eq!(right.acceleration_m_per_s2, 3.0);
    }

    #[test]
    fn exact_step_coverage_is_required() {
        let signal = acceleration_signal(vec![
            SignalKnot::continuous(0.5, real(0.0)),
            SignalKnot::continuous(2.0, real(0.0)),
        ]);
        assert!(matches!(
            AccelerationTrajectory::materialize(
                &signal,
                StructuralTranslationDof::Ux,
                0.0,
                2.0,
                0.0,
                0.0,
                "support"
            ),
            Err(StructuralMotionError::IncompleteCoverage { .. })
        ));
    }

    #[test]
    fn component_selection_and_reject_extrapolation_are_mandatory() {
        let mut signal = acceleration_signal(vec![
            SignalKnot::continuous(0.0, real(1.0)),
            SignalKnot::continuous(1.0, real(2.0)),
        ]);
        let y = AccelerationTrajectory::materialize(
            &signal,
            StructuralTranslationDof::Uy,
            0.0,
            1.0,
            0.0,
            0.0,
            "support",
        )
        .unwrap();
        assert_eq!(
            y.sample(0.0, SignalSampleApproach::At)
                .unwrap()
                .acceleration_m_per_s2,
            10.0
        );
        let SignalRepresentation::Tabulated(table) = &mut signal.representation else {
            unreachable!()
        };
        table.extrapolation = SignalExtrapolation::HoldEndpoint;
        assert!(matches!(
            AccelerationTrajectory::materialize(
                &signal,
                StructuralTranslationDof::Ux,
                0.0,
                1.0,
                0.0,
                0.0,
                "support"
            ),
            Err(StructuralMotionError::InvalidSignal { .. })
        ));
    }
}
