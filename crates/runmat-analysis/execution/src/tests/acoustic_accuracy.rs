use super::*;
use runmat_analysis_core::{
    AcousticConstraint, AcousticConstraintKind, AnalysisConstraint, Frequency, IndependentDomain,
    MaterialAcousticModel, PrescribedValue, Pressure, PropertyRegionTarget, RegionTarget,
};
use runmat_analysis_fea::execution::{
    AccuracySourceKind, AccuracySourceReceipt, AcousticHarmonicWorkload,
    AcousticNumericalReceiptArtifact,
};
use runmat_analysis_fea::physics::acoustic::AcousticHarmonicPolicy;
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

// Transport witness for already accepted outputs, not native-worker qualification.
struct AcceptedOutputs(Vec<PreparedAnalysisObjects>);
impl AnalysisKernel for AcceptedOutputs {
    fn execute(
        &self,
        invocation: AnalysisKernelInvocation<'_>,
    ) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
        assert_eq!(invocation.kind, AnalysisWorkKind::SolveAcousticHarmonic);
        Ok(self
            .0
            .iter()
            .cloned()
            .map(AnalysisKernelOutput::prepared)
            .collect())
    }
}

pub(super) fn workload() -> AcousticHarmonicWorkload {
    let mut model = runmat_analysis_fea::fixtures::fixture_model(
        runmat_analysis_fea::fixtures::FixtureId::CantileverLinearStatic,
    );
    model.signals.clear();
    model.excitations.clear();
    model.excitation_cases.clear();
    model.constraints.clear();
    model.boundaries.clear();
    model.materials[0].acoustic = Some(MaterialAcousticModel {
        speed_of_sound_m_per_s: runmat_types::physical::SpeedOfSound::new(2.0),
        damping_ratio: runmat_types::physical::DampingRatio::new(0.0),
    });
    model.materials[0].mass_density_kg_per_m3 = Some(runmat_types::physical::MassDensity::new(3.0));
    model.material_assignments[0].target =
        PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_solid("solid"));
    let mut step = ProcedureStep::acoustic_harmonic("acoustic");
    step.independent_domain = IndependentDomain::FrequencyHertz(vec![0.2]);
    model.sequence.steps = vec![step];
    model
        .constraints
        .push(AnalysisConstraint::Acoustic(AcousticConstraint {
            constraint_id: "pressure".into(),
            kind: AcousticConstraintKind::PrescribedPressure {
                target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                    "face:0",
                )),
                value: PrescribedValue::Constant(Pressure::new(1.0)),
            },
        }));
    AcousticHarmonicWorkload {
        model,
        step_id: ProcedureStepId("acoustic".into()),
        frequency: Frequency::new(0.2),
        solver: AcousticHarmonicPolicy::default(),
        power_balance_relative_tolerance: 1e-8,
    }
}

// witness: FEA-ACOUSTIC-ACCURACY-001/lineage @ acoustic_harmonic/tet4|tet10/prepared_controller;
//   boundary: Both-order accepted-output bridge fixture validates exact context and rejects
//   production relabeling; not actual native worker or full public gate evidence.
#[test]
fn acoustic_accuracy_admission_binds_retained_result_and_original_intent() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        for chunk in [1024, 8192] {
            let limits = ObjectInventoryLimits::default();
            let workload = workload();
            let mesh = canonical_tetrahedron_solver_mesh(order);
            let task = prepare_acoustic_harmonic_task(AcousticHarmonicTaskPreparation {
                local_execution:
                    runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
                authored_study_digest: digest("acoustic"),
                frozen_project_digest: None,
                program_revision: revision(),
                artifact_access: access(),
                workload: workload.clone(),
                mesh: mesh.clone(),
                output_policy: AnalysisOutputPolicy::new(chunk, limits).unwrap(),
                required_capabilities: Default::default(),
            })
            .unwrap();
            let result = workload.execute(&mesh).unwrap();
            let receipt = AcousticNumericalReceiptArtifact::new(&workload, &result, &mesh).unwrap();
            let object = |kind, identity, bytes: Vec<u8>| {
                prepare_analysis_objects(
                    kind,
                    identity,
                    &bytes,
                    chunk,
                    ObjectNamespace::ResultValue,
                    limits,
                )
                .unwrap()
            };
            let receipt_objects = object(
                AnalysisNumericalObjectKind::NumericalReceipt,
                receipt.identity().digest(),
                receipt.canonical_bytes().unwrap(),
            );
            let mut context: Vec<_> = task
                .inputs
                .iter()
                .filter(|v| v.manifest.kind != AnalysisNumericalObjectKind::ResourcePlan)
                .cloned()
                .collect();
            context.push(object(
                AnalysisNumericalObjectKind::SolverResult,
                result.source_result_identity().unwrap().digest(),
                result.canonical_bytes().unwrap(),
            ));
            let profiles = result
                .state_publication_profiles(context[2].manifest.payload_digest)
                .unwrap();
            let profiles = object(
                AnalysisNumericalObjectKind::StatePublicationProfiles,
                profiles.identity(),
                profiles.canonical_bytes().unwrap(),
            );
            let mut store = MemoryObjects::default();
            task.program.write_to(&mut store).unwrap();
            let completed = execute_serial_analysis(
                &task.program.request,
                &mut store,
                &AcceptedOutputs(vec![context[2].clone(), receipt_objects.clone(), profiles]),
                chunk,
                limits,
            )
            .unwrap();
            let response = AnalysisHostResponse::completed(&task.host, &completed)
                .unwrap()
                .program_response();
            let imported =
                import_acoustic_harmonic_result(&store, response, &task, limits).unwrap();
            assert_eq!(imported.result, result);
            assert_eq!(imported.numerical_receipt, receipt);
            let preparation = || AccuracyPreparation {
                solve_host: task.host.clone(),
                source_kind: AccuracySourceKind::AcousticHarmonic,
                source_numerical_receipt: receipt_objects.clone(),
                source_observations: None,
                source_context_inputs: context.clone(),
                sequence: SequenceIdentity::from_solver_contract(&workload.model.sequence).unwrap(),
                step: workload.model.sequence.steps[0].clone(),
                program_revision: revision(),
                chunk_bytes: chunk,
                inventory_limits: limits,
            };
            let prepared = prepare_accuracy(preparation()).unwrap();
            super::accuracy_import::assert_original_evidence(&prepared);
            let source = AccuracySourceReceipt::AcousticHarmonic(receipt);
            let check = |inputs: &[PreparedAnalysisObjects]| {
                validate_accuracy_source_context(
                    &prepared.request,
                    &source,
                    None,
                    &inputs.iter().collect::<Vec<_>>(),
                )
            };
            check(&context).unwrap();
            for index in 0..3 {
                let mut missing = context.clone();
                missing.remove(index);
                assert!(check(&missing).is_err());
                let mut duplicate = context.clone();
                duplicate.push(context[index].clone());
                assert!(check(&duplicate).is_err());
            }
            let mut foreign = workload.clone();
            foreign.power_balance_relative_tolerance *= 0.5;
            let other = foreign.execute(&mesh).unwrap();
            for replacement in [
                object(
                    AnalysisNumericalObjectKind::AcousticHarmonicWorkload,
                    foreign.identity().unwrap().digest(),
                    foreign.canonical_bytes().unwrap(),
                ),
                object(
                    AnalysisNumericalObjectKind::SolverResult,
                    other.source_result_identity().unwrap().digest(),
                    other.canonical_bytes().unwrap(),
                ),
            ] {
                let mut substituted = context.clone();
                let index = substituted
                    .iter()
                    .position(|v| v.manifest.kind == replacement.manifest.kind)
                    .unwrap();
                substituted[index] = replacement;
                assert!(check(&substituted).is_err());
            }
            super::accuracy_source_context_tests::assert_accuracy_plan_identity_bound(
                &prepared.request,
                &source,
                None,
                &context.iter().collect::<Vec<_>>(),
            );
            let mut missing = preparation();
            missing.source_context_inputs.pop();
            assert!(prepare_accuracy(missing).is_err());
        }
    }
}
