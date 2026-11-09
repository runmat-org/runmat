//! Rehashed, internally consistent claims must still match original scientific source evidence.
use super::*;
use runmat_analysis_core::{
    AccuracyCriterion, AccuracyTarget, AccuracyTolerance, ErrorContributionKind,
    RequiredAccuracyEvidence,
};
use runmat_analysis_fea::{
    execution::{
        AccuracySourceKind, AccuracySourceReceipt, ThermalNumericalReceiptArtifact, ThermalWorkload,
    },
    results::accuracy::{
        evaluate_accuracy, publication_gate, AccuracyArtifact, AccuracyArtifactInput,
        AccuracyEvidence, AccuracyEvidenceScope, AccuracyMeasure,
    },
    ComputeBackend, ThermalProcedureControl,
};

fn artifact(prepared: &PreparedAccuracy, evidence: Vec<AccuracyEvidence>) -> AccuracyArtifact {
    let request = &prepared.request;
    let evaluation = evaluate_accuracy(&request.plan, &evidence).unwrap();
    let gate = publication_gate(&request.plan, &evaluation).unwrap();
    AccuracyArtifact::new(AccuracyArtifactInput {
        evaluation_request: request.identity(),
        producer: request.producer,
        source_kind: request.source_kind,
        source_result: request.source_result,
        source_observations: request.source_observations,
        evidence,
        evaluation,
        gate,
    })
    .unwrap()
}

pub(super) fn assert_original_evidence(prepared: &PreparedAccuracy) {
    let limits = ObjectInventoryLimits::default();
    let receipt = AccuracySourceReceipt::from_canonical_bytes(
        prepared.request.source_kind,
        &prepared
            .inputs
            .iter()
            .find(|i| i.manifest.kind == AnalysisNumericalObjectKind::NumericalReceipt)
            .unwrap()
            .payload(),
    )
    .unwrap();
    let original = receipt.evidence(&prepared.request).unwrap();
    assert!(!original.is_empty());
    let mut store = MemoryObjects::default();
    prepared.program.write_to(&mut store).unwrap();
    let run = |evidence: Vec<AccuracyEvidence>, store: &mut MemoryObjects| {
        let report = artifact(prepared, evidence);
        let completed = execute_serial_analysis(
            &prepared.program.request,
            store,
            &AccuracyKernel(report),
            prepared.host.output_policy.chunk_bytes(),
            limits,
        )
        .ok()?;
        // A counterfeit report larger than the admitted output envelope is rejected by the
        // executing host before import; one that fits is rejected by import.
        AnalysisHostResponse::completed(&prepared.host, &completed)
            .ok()
            .map(|response| response.program_response())
    };
    let rejected = |response: Option<_>, store: &MemoryObjects| {
        response.is_none_or(|response| import_accuracy(store, response, prepared, limits).is_err())
    };
    let response = run(original.clone(), &mut store).expect("original evidence fits its plan");
    import_accuracy(&store, response.clone(), prepared, limits).unwrap();
    for mutation in 0..3 {
        let mut evidence = original.clone();
        let e = &original[0];
        let measured = if mutation == 0 {
            match e.measured {
                AccuracyMeasure::Relative(v) => {
                    AccuracyMeasure::Relative(if v == 0.0 { 1e-12 } else { 0.0 })
                }
                _ => panic!("expected source-relative evidence"),
            }
        } else {
            e.measured
        };
        let assumptions = if mutation == 1 {
            vec!["Counterfeit independently certified convergence".into()]
        } else {
            e.assumptions.clone()
        };
        let scope = if mutation == 2 {
            AccuracyEvidenceScope::Step(prepared.request.producer)
        } else {
            e.scope
        };
        evidence[0] = AccuracyEvidence::new(
            e.target.clone(),
            e.contribution,
            e.kind,
            measured,
            assumptions,
            scope,
        )
        .unwrap();
        // Report identity, report status, gate and transport hashes are all recomputed correctly.
        let forged = run(evidence, &mut store);
        assert!(rejected(forged, &store));
    }
    for criterion in prepared.request.plan.criteria.iter().filter(|criterion| {
        !original.iter().any(|e| {
            e.target == criterion.target
                && e.contribution == criterion.contribution
                && e.kind == criterion.evidence
        })
    }) {
        let mut evidence = original.clone();
        let measured = match criterion.tolerance {
            AccuracyTolerance::Relative(_) => AccuracyMeasure::Relative(0.0),
            AccuracyTolerance::Absolute { quantity, .. } => AccuracyMeasure::Absolute {
                quantity,
                canonical_value: 0.0,
            },
        };
        evidence.push(
            AccuracyEvidence::new(
                criterion.target.clone(),
                criterion.contribution,
                criterion.evidence,
                measured,
                vec!["Invented absent source evidence".into()],
                AccuracyEvidenceScope::SolverResult(prepared.request.source_result),
            )
            .unwrap(),
        );
        if prepared.request.source_kind == AccuracySourceKind::Thermal {
            let honest = artifact(prepared, original.clone());
            let counterfeit = artifact(prepared, evidence.clone());
            assert_eq!(
                honest.evaluation().report().status,
                runmat_analysis_fea::results::accuracy::AccuracyStatus::Indeterminate
            );
            assert_eq!(
                counterfeit.evaluation().report().status,
                runmat_analysis_fea::results::accuracy::AccuracyStatus::Satisfied
            );
            assert_eq!(
                publication_gate(&prepared.request.plan, honest.evaluation())
                    .unwrap()
                    .decision,
                runmat_analysis_fea::results::accuracy::PublicationDecision::Block
            );
            assert_eq!(
                publication_gate(&prepared.request.plan, counterfeit.evaluation())
                    .unwrap()
                    .decision,
                runmat_analysis_fea::results::accuracy::PublicationDecision::Allow
            );
        }
        let forged = run(evidence, &mut store);
        assert!(rejected(forged, &store));
    }
    let mut bad = PreparedAccuracy {
        request: prepared.request.clone(),
        host: prepared.host.clone(),
        program: prepared.program.clone(),
        inputs: prepared.inputs.clone(),
        resource_plan: prepared.resource_plan.clone(),
        source_observations: prepared.source_observations.clone(),
    };
    let context = bad
        .inputs
        .iter()
        .position(|i| {
            !matches!(
                i.manifest.kind,
                AnalysisNumericalObjectKind::AccuracyEvaluationRequest
                    | AnalysisNumericalObjectKind::NumericalReceipt
                    | AnalysisNumericalObjectKind::ObservationCatalog
                    | AnalysisNumericalObjectKind::ResourcePlan
            )
        })
        .unwrap();
    bad.inputs.remove(context);
    assert!(import_accuracy(&store, response, &bad, limits).is_err());
}

#[test]
fn thermal_cold_import_rejects_rehashed_missing_spatial_certificate() {
    use runmat_analysis_core::{AnalysisSequence, ProcedureStep, Temperature, ThermalDomain};
    use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mut model = runmat_analysis_fea::fixtures::fixture_model(
            runmat_analysis_fea::fixtures::FixtureId::CantileverLinearStatic,
        );
        model.constraints.clear();
        model.boundaries.clear();
        model.excitations.clear();
        model.excitation_cases.clear();
        model.signals.clear();
        model
            .boundaries
            .push(runmat_analysis_core::AnalysisBoundary {
                boundary_id: "insulated".into(),
                target: runmat_analysis_core::RegionTarget::boundary(
                    runmat_meshing_core::fixtures::fixture_face("face:0"),
                ),
                operator: runmat_analysis_core::BoundaryOperator::ThermalHeatFlux {
                    value: runmat_analysis_core::PrescribedValue::Constant(
                        runmat_analysis_core::HeatFlux::new(0.0),
                    ),
                },
            });
        model.thermal = Some(ThermalDomain {
            reference_temperature: Temperature::new(293.15),
        });
        let mut step = ProcedureStep::thermal_transient("heat");
        step.accuracy.criteria.push(AccuracyCriterion {
            criterion_id: "missing-spatial".into(),
            target: AccuracyTarget::NumericalResult,
            contribution: ErrorContributionKind::SpatialDiscretization,
            evidence: RequiredAccuracyEvidence::ConvergenceStudy,
            tolerance: AccuracyTolerance::Relative(1e-3),
            mandatory: true,
        });
        model.sequence = AnalysisSequence::from_steps("cold-import", vec![step.clone()]);
        let workload = ThermalWorkload {
            model,
            step_id: step.step_id.clone(),
            initial_fields: Vec::new(),
            backend: ComputeBackend::Cpu,
            procedure: ThermalProcedureControl::BackwardEuler { snapshot_count: 3 },
            residual_target: 1e-8,
        };
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let result = workload.execute(&mesh).unwrap();
        let receipt = ThermalNumericalReceiptArtifact::new(&workload, &result).unwrap();
        let task = prepare_thermal_task(ThermalTaskPreparation {
            maximum_resident_bytes: u64::MAX,
            local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
            field_source: None,
            authored_study_digest: digest("cold-import"),
            frozen_project_digest: None,
            program_revision: revision(),
            artifact_access: ArtifactAccess {
                authorization_scope: "test-accuracy".into(),
                encryption_context: digest("accuracy"),
            },
            workload: workload.clone(),
            mesh,
            output_policy: AnalysisOutputPolicy::new(1024, ObjectInventoryLimits::default())
                .unwrap(),
            required_capabilities: Default::default(),
        })
        .unwrap();
        let prepared = prepare_accuracy(AccuracyPreparation {
            solve_host: task.host.clone(),
            source_kind: AccuracySourceKind::Thermal,
            source_numerical_receipt: prepare_analysis_objects(
                AnalysisNumericalObjectKind::NumericalReceipt,
                receipt.identity().digest(),
                &receipt.canonical_bytes().unwrap(),
                1024,
                ObjectNamespace::ResultValue,
                ObjectInventoryLimits::default(),
            )
            .unwrap(),
            source_observations: None,
            source_context_inputs: task
                .inputs
                .iter()
                .filter(|i| i.manifest.kind != AnalysisNumericalObjectKind::ResourcePlan)
                .cloned()
                .collect(),
            sequence: SequenceIdentity::from_solver_contract(&workload.model.sequence).unwrap(),
            step,
            program_revision: revision(),
            chunk_bytes: 1024,
            inventory_limits: ObjectInventoryLimits::default(),
        })
        .unwrap();
        assert_original_evidence(&prepared);
    }
}
