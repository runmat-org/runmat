use super::*;
use runmat_analysis_core::{
    AnalysisBoundary, BoundaryOperator, PrescribedValue, RegionTarget, SurfaceEmissivity,
};

#[test]
fn nonlinear_equilibrium_uses_original_acceptance_not_inner_linear_residual() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let projection = crate::topology::AnalysisProjection::derive(&mesh).unwrap();
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        model.constraints.clear();
        model.boundaries.clear();
        model.thermal = Some(ThermalDomain {
            reference_temperature: Temperature::new(400.0),
        });
        model.sequence = AnalysisSequence::from_steps(
            "radiation_accuracy",
            vec![ProcedureStep::thermal_transient("heat")],
        );
        model.boundaries.push(AnalysisBoundary {
            boundary_id: "gray_surface".into(),
            target: RegionTarget::boundary(
                projection.boundary_faces[0].provenance[0]
                    .analysis_selection_id()
                    .unwrap(),
            ),
            operator: BoundaryOperator::ThermalRadiation {
                ambient_temperature: PrescribedValue::Constant(Temperature::new(300.0)),
                emissivity: SurfaceEmissivity::new(0.7),
            },
        });
        let workload = ThermalWorkload {
            initial_fields: Vec::new(),
            step_id: model.sequence.steps[0].step_id.clone(),
            model,
            backend: ComputeBackend::Cpu,
            procedure: ThermalProcedureControl::BackwardEuler { snapshot_count: 3 },
            residual_target: 1e-8,
        };
        let result = workload.execute(&mesh).unwrap();
        let receipt = ThermalNumericalReceiptArtifact::new(&workload, &result).unwrap();
        let mut nonlinear = criterion("nonlinear", RequiredAccuracyEvidence::ResidualBound, 1e-8);
        nonlinear.contribution = ErrorContributionKind::Nonlinear;
        let request = request(&receipt, vec![nonlinear]);
        let evidence = AccuracySourceReceipt::Thermal(receipt.clone())
            .evidence(&request)
            .unwrap();
        assert_eq!(evidence.len(), 1);
        assert_eq!(
            evidence[0].measured,
            AccuracyMeasure::Relative(
                receipt
                    .receipt()
                    .maximum_relative_nonlinear_residual
                    .unwrap()
            )
        );
        assert_eq!(
            evaluate_accuracy(&request.plan, &evidence)
                .unwrap()
                .report()
                .status,
            AccuracyStatus::Satisfied
        );
    }

    // A linear receipt cannot substitute its inner residual for missing nonlinear evidence.
    let receipt = receipt(ElementOrder::Tet4);
    let mut nonlinear = criterion(
        "missing_nonlinear",
        RequiredAccuracyEvidence::ResidualBound,
        1e-4,
    );
    nonlinear.contribution = ErrorContributionKind::Nonlinear;
    let request = request(&receipt, vec![nonlinear]);
    let evidence = AccuracySourceReceipt::Thermal(receipt)
        .evidence(&request)
        .unwrap();
    assert!(evidence.is_empty());
    let evaluated = evaluate_accuracy(&request.plan, &evidence).unwrap();
    assert_eq!(evaluated.report().status, AccuracyStatus::Indeterminate);
    assert_eq!(
        publication_gate(&request.plan, &evaluated)
            .unwrap()
            .decision,
        PublicationDecision::Block
    );
}
