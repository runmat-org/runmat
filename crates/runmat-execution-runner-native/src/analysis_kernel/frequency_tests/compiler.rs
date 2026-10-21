use super::*;
use runmat_analysis_execution::{AnalysisHostWorkload, AnalysisOutputPolicy};
use runmat_analysis_fea::{execution::AnalysisWorkKind, planning::resource::FeaResourcePlan};

pub(super) fn reject_understated(
    host: &AnalysisHostWorkload,
    inputs: &[PreparedAnalysisObjects],
    plan: &FeaResourcePlan,
    recovery: bool,
    limits: ObjectInventoryLimits,
) {
    let mut input = plan.input.clone();
    let peak = match &mut input.demand {
        FeaResourceDemand::StructuralFrequencyEntry(d)
        | FeaResourceDemand::StructuralFrequencyFieldRecovery(d) => {
            &mut d.constraint_compilation_peak_bytes
        }
        _ => panic!("original frequency compiler demand required"),
    };
    assert!(peak.0 > 1);
    peak.0 -= 1;
    let understated = build_resource_plan(input).unwrap();
    let mut substituted = inputs.to_vec();
    *substituted
        .iter_mut()
        .find(|v| v.manifest.kind == AnalysisNumericalObjectKind::ResourcePlan)
        .unwrap() = prepare_analysis_objects(
        AnalysisNumericalObjectKind::ResourcePlan,
        understated.identity.digest(),
        &understated.canonical_bytes().unwrap(),
        256,
        ObjectNamespace::InputValue,
        limits,
    )
    .unwrap();
    let error = NativeAnalysisKernel
        .execute(AnalysisKernelInvocation {
            output_policy: AnalysisOutputPolicy::new(256, limits).unwrap(),
            kind: if recovery {
                AnalysisWorkKind::RecoverField
            } else {
                AnalysisWorkKind::SolveStructuralFrequencyEntry
            },
            host,
            inputs: &substituted,
        })
        .unwrap_err();
    assert!(
        error.to_string().contains(if recovery {
            "structural frequency recovery received a substituted resource plan"
        } else {
            "frequency entry resource plan does not converge from authoritative inputs"
        }),
        "{error}"
    );
}

#[test]
fn original_compiler_reservation_is_rederived_for_both_orders_and_frequency_phases() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let (mut model, mesh) = super::super::structural_fixture::model_and_mesh(order);
        let mut step = ProcedureStep::structural_harmonic("frequency");
        step.independent_domain = IndependentDomain::FrequencyHertz(vec![20.0, 40.0]);
        model.sequence.steps = vec![step];
        let workload = StructuralFrequencyWorkload::new(StructuralFrequencyWorkloadInput {
            model,
            step_id: ProcedureStepId("frequency".into()),
            backend: ComputeBackend::Cpu,
            observation_execution: ObservationExecutionPolicy::scalable(),
            plan: FrequencyResponsePlan::new(20.0, &[40.0], &[]).unwrap(),
            solver: StructuralFrequencySolverPlan::Direct {
                policy: DirectHarmonicPolicy::default(),
            },
            prestress_transition: None,
        })
        .unwrap();
        let (_, fields) = execute_frequency(&workload, &mesh, None, None, true);
        assert_eq!(fields.len(), 2);
    }
}
