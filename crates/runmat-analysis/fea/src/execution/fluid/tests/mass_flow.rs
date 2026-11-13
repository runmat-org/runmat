//! Full source-report inventory survives complete history admission and cold recovery.
use super::*;
use crate::execution::{
    FluidTransientBlockWorkload, FluidTransientCheckpoint, FluidTransientWorkload,
};

#[test]
fn signed_source_reports_fit_full_original_history_profiles() {
    let f =
        crate::physics::cfd::canonical::qualification::mass_flow::accelerating(2, 0.001, 0.01, 1.0);
    let (steady, _) = authored();
    let mut model = steady.model;
    model.constraints = f
        .constraints
        .iter()
        .cloned()
        .map(AnalysisConstraint::Fluid)
        .collect();
    model.boundaries = f.operators.clone();
    model.signals = f.signals.clone();
    model.excitations = f.actions.clone();
    model.excitation_cases = vec![ExcitationCase {
        case_id: ExcitationCaseId("signed-histories".into()),
        action_ids: model
            .excitations
            .iter()
            .map(|a| a.excitation().excitation_id.clone())
            .collect(),
        activation: ExcitationCaseActivation::Always,
    }];
    model.sequence = AnalysisSequence::from_steps(
        "mass-history",
        vec![ProcedureStep::fluid_transient("advance")],
    );
    model.sequence.steps[0].independent_domain = IndependentDomain::TimeSeconds {
        origin: 0.0,
        start: 0.0,
        end: 0.3,
        interval_count: 3,
    };
    model
        .initial_conditions
        .push(InitialCondition::FluidVelocity(InitialConditionValue {
            initial_condition_id: "original-u".into(),
            target: InitialConditionTarget::WholeModel,
            frame: CoordinateFrame::Global,
            value: InitialValue::UniformVector3([0.001, 0.0, 0.0].map(Velocity::new)),
            applies_before_step_id: "advance".into(),
        }));
    let workload = FluidTransientWorkload {
        model,
        step_id: ProcedureStepId("advance".into()),
        policy: f.policy,
        geometry_policy: crate::physics::cfd::canonical::qualification::fixture::geometry_policy(),
        initial_fields: Vec::new(),
    };
    let prepared = workload.prepare(&f.mesh).unwrap();
    let bound = prepared.result_encoding_profile().unwrap();
    let result = prepared.execute().unwrap();
    let actual = runmat_canonical_codec::profile_serialized(&result, super::super::LIMITS).unwrap();
    assert!(actual.encoded_bytes_upper_bound <= bound.encoded_bytes_upper_bound);
    assert!(
        actual.materialization_resident_bytes_estimate
            <= bound.materialization_resident_bytes_estimate
    );
    let request = FluidTransientBlockWorkload {
        predecessor: workload.initial_checkpoint(&f.mesh).unwrap(),
        workload: workload.clone(),
        end_interval: 3,
    };
    let original = request.execute(&f.mesh).unwrap();
    let cold = FluidTransientCheckpoint::from_canonical_bytes(&original.canonical_bytes().unwrap())
        .unwrap();
    cold.recover(&workload, &f.mesh).unwrap();
    assert_eq!(
        cold.retained_results().collect::<Vec<_>>(),
        original.retained_results().collect::<Vec<_>>()
    );
    for (_, result) in cold.retained_results() {
        for id in ["mass-inlet", "mass-outlet"] {
            assert!(result
                .residual
                .boundary_balances
                .iter()
                .any(|b| b.authority_id == id));
        }
    }
}
