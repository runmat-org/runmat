use runmat_analysis_core::{
    CoordinateFrame, DirectionalCombination, FormulationRequirement, IndependentDomain,
    ModalCombination, ProcedureStep, ResponseSpectrum, ResponseSpectrumId, SpectrumAbscissa,
    SpectrumConstruction, SpectrumDamping, SpectrumDirection, SpectrumEnvelope, SpectrumEvidence,
    SpectrumExtrapolation, SpectrumInterpolation, SpectrumResponse,
};

use crate::{
    assembly::StructuralFormulationRequirement,
    fixtures::{fixture_model, FixtureId},
    pipeline::modal_basis::directional_influence_vector,
    pipeline::modal_governance::{
        ModalConvergencePolicy, ModalTruncationPolicy as EigenspaceTruncationPolicy,
    },
    run_modal_with_options, ComputeBackend, ModalSolveOptions,
};

use super::*;

#[test]
fn modal_and_directional_combination_policies_have_independent_semantics() {
    assert_eq!(
        ResponseSpectrumDisplacementConversion::for_response(
            SpectrumResponse::RelativeDisplacement
        ),
        Some(ResponseSpectrumDisplacementConversion::Direct)
    );
    assert_eq!(
        ResponseSpectrumDisplacementConversion::for_response(SpectrumResponse::PseudoVelocity),
        Some(ResponseSpectrumDisplacementConversion::DivideByAngularFrequency)
    );
    assert_eq!(
        ResponseSpectrumDisplacementConversion::for_response(SpectrumResponse::PseudoAcceleration),
        Some(ResponseSpectrumDisplacementConversion::DivideByAngularFrequencySquared)
    );
    assert_eq!(
        ResponseSpectrumDisplacementConversion::for_response(
            SpectrumResponse::AbsoluteAcceleration
        ),
        None
    );
    let modal = vec![vec![3.0], vec![4.0]];
    assert_eq!(
        combine_modal_responses(&modal, &[1.0, 2.0], 0.05, ModalCombination::Srss).unwrap(),
        vec![5.0]
    );
    assert_eq!(
        combine_modal_responses(&modal, &[1.0, 2.0], 0.05, ModalCombination::AbsoluteSum,).unwrap(),
        vec![7.0]
    );
    assert_eq!(
        combine_modal_responses(
            &modal,
            &[1.0, 1.0],
            0.05,
            ModalCombination::CompleteQuadraticCombination,
        )
        .unwrap(),
        vec![7.0]
    );
    assert_eq!(
        combine_modal_responses(&modal, &[1.0, 2.0], 0.05, ModalCombination::TenPercent,).unwrap(),
        vec![5.0]
    );

    let first = [10.0];
    let second = [2.0];
    let third = [1.0];
    let directional = combine_directional_responses(
        &[&first, &second, &third],
        DirectionalCombination::PercentageRule {
            primary: 1.0,
            secondary: 0.4,
        },
    )
    .unwrap();
    assert!((directional[0] - 11.2).abs() <= 1.0e-14);
}

#[test]
fn spectrum_interpolation_respects_axis_and_extrapolation_policy() {
    let mut spectrum = spectrum_fixture(
        SpectrumAbscissa::FrequencyHertz(vec![1.0, 10.0]),
        vec![2.0, 20.0],
        [1.0, 0.0, 0.0],
    );
    assert!((interpolate_spectrum(&spectrum, 5.5).unwrap() - 11.0).abs() <= 1.0e-14);
    assert!(interpolate_spectrum(&spectrum, 0.5).is_err());
    spectrum.extrapolation = SpectrumExtrapolation::HoldEndpoint;
    assert_eq!(interpolate_spectrum(&spectrum, 0.5).unwrap(), 2.0);

    spectrum.abscissa = SpectrumAbscissa::PeriodSeconds(vec![0.1, 1.0]);
    spectrum.ordinates = vec![20.0, 2.0];
    spectrum.interpolation = SpectrumInterpolation::LinearPeriod;
    assert!((interpolate_spectrum(&spectrum, 2.0).unwrap() - 12.0).abs() <= 1.0e-14);
}

#[test]
fn physical_modal_basis_drives_canonical_response_result() {
    let truncation = EigenspaceTruncationPolicy::requested(2);
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let mut step = ProcedureStep::structural_modal("modes");
    step.independent_domain = IndependentDomain::Eigenvalue {
        requested_count: truncation.requested_mode_count(),
    };
    model.sequence.steps = vec![step];
    let mesh = crate::tests::single_tetrahedron_solver_mesh_for_model(&model);
    let run = run_modal_with_options(
        &model,
        ComputeBackend::Cpu,
        ModalSolveOptions {
            formulation_requirement: FormulationRequirement::Experimental,
            truncation,
            convergence: ModalConvergencePolicy::exploratory(),
            residual_vector_directions: Vec::new(),
            solver_mesh: Some(mesh),
        },
    )
    .unwrap();
    let first_mode = &run.basis.modes()[0];
    let axis = first_mode
        .directional()
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.effective_mass().total_cmp(&right.effective_mass()))
        .map(|(axis, _)| axis)
        .unwrap();
    let mut direction = [0.0; 3];
    direction[axis] = 1.0;
    let frequencies = [
        run.basis.modes()[0].frequency_hz() * 0.5,
        run.basis.modes().last().unwrap().frequency_hz() * 1.5,
    ];
    let spectrum = spectrum_fixture(
        SpectrumAbscissa::FrequencyHertz(frequencies.to_vec()),
        vec![1.0; frequencies.len()],
        direction,
    );
    let result =
        solve_modal_response_spectrum(&run.basis, std::slice::from_ref(&spectrum)).unwrap();
    assert_eq!(result.basis(), run.basis.identity());
    assert_eq!(result.contributions().len(), 1);
    assert_eq!(result.contributions()[0].modes.len(), 1);
    assert!(result.combined_response().iter().any(|value| *value > 0.0));
    let restored =
        ModalResponseSpectrumResult::from_canonical_bytes(&result.canonical_bytes().unwrap())
            .unwrap();
    assert_eq!(restored, result);
    restored
        .validate_for(&run.basis, std::slice::from_ref(&spectrum))
        .unwrap();

    let mut substituted = restored;
    substituted.contributions[0].modes[0].interpolated_ordinate *= 2.0;
    substituted.identity = substituted.derive_identity().unwrap();
    assert!(substituted
        .validate_for(&run.basis, std::slice::from_ref(&spectrum))
        .is_err());
}

#[test]
fn basis_only_solver_rejects_operator_dependent_corrections() {
    let truncation = EigenspaceTruncationPolicy::requested(1);
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let mut step = ProcedureStep::structural_modal("modes");
    step.independent_domain = IndependentDomain::Eigenvalue {
        requested_count: truncation.requested_mode_count(),
    };
    model.sequence.steps = vec![step];
    let mesh = crate::tests::single_tetrahedron_solver_mesh_for_model(&model);
    let run = run_modal_with_options(
        &model,
        ComputeBackend::Cpu,
        ModalSolveOptions {
            formulation_requirement: FormulationRequirement::Experimental,
            truncation,
            convergence: ModalConvergencePolicy::exploratory(),
            residual_vector_directions: Vec::new(),
            solver_mesh: Some(mesh),
        },
    )
    .unwrap();
    let frequency = run.basis.modes()[0].frequency_hz();
    let mut spectrum = spectrum_fixture(
        SpectrumAbscissa::FrequencyHertz(vec![frequency]),
        vec![1.0],
        [1.0, 0.0, 0.0],
    );
    spectrum.truncation.correction = runmat_analysis_core::ModalTruncationCorrection::ResidualMode;
    assert!(solve_modal_response_spectrum(&run.basis, &[spectrum]).is_err());
}

#[test]
fn operator_aware_missing_mass_closes_the_truncated_acceleration_influence() {
    let truncation = EigenspaceTruncationPolicy::requested(2);
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let mut step = ProcedureStep::structural_modal("modes");
    step.independent_domain = IndependentDomain::Eigenvalue {
        requested_count: truncation.requested_mode_count(),
    };
    model.sequence.steps = vec![step];
    let mesh = crate::tests::single_tetrahedron_solver_mesh_for_model(&model);
    let run = run_modal_with_options(
        &model,
        ComputeBackend::Cpu,
        ModalSolveOptions {
            formulation_requirement: FormulationRequirement::Experimental,
            truncation,
            convergence: ModalConvergencePolicy::exploratory(),
            residual_vector_directions: Vec::new(),
            solver_mesh: Some(mesh.clone()),
        },
    )
    .unwrap();
    let excitations = crate::excitation::realize_step_excitations(
        &model,
        &model.sequence.steps[0],
        crate::excitation::ExcitationSamplePoint::Static,
    )
    .unwrap();
    let summary = crate::assembly::assemble_structural_system(
        &model,
        &excitations,
        Some(mesh),
        StructuralFormulationRequirement::ExperimentalSyntheticOperator,
    )
    .unwrap();
    let first_mode = &run.basis.modes()[0];
    let axis = first_mode
        .directional()
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.effective_mass().total_cmp(&right.effective_mass()))
        .map(|(axis, _)| axis)
        .unwrap();
    let mut direction = [0.0; 3];
    direction[axis] = 1.0;
    let frequencies = [
        first_mode.frequency_hz() * 0.5,
        run.basis.modes().last().unwrap().frequency_hz() * 1.5,
    ];
    let mut spectrum = spectrum_fixture(
        SpectrumAbscissa::FrequencyHertz(frequencies.to_vec()),
        vec![2.0; frequencies.len()],
        direction,
    );
    spectrum.response = SpectrumResponse::AbsoluteAcceleration;
    spectrum.truncation.correction = runmat_analysis_core::ModalTruncationCorrection::MissingMass {
        zero_period_acceleration: 2.0,
    };

    assert!(solve_modal_response_spectrum(&run.basis, &[spectrum.clone()]).is_err());
    let result = solve_modal_response_spectrum_with_operator(
        &summary,
        &run.basis,
        std::slice::from_ref(&spectrum),
    )
    .unwrap();
    let contribution = &result.contributions()[0];
    let participation = contribution.modes[0].directional_participation_factor;
    let influence = directional_influence_vector(&summary, direction);
    let expected = influence
        .iter()
        .zip(first_mode.dof_shape())
        .map(|(influence, shape)| 2.0 * (influence - participation * shape))
        .collect::<Vec<_>>();
    assert!(vectors_close(
        &expected,
        &contribution.truncation.correction_response
    ));
    result
        .validate_for_operator(&summary, &run.basis, std::slice::from_ref(&spectrum))
        .unwrap();
    let recovery = recover_modal_response_spectrum_fields(&summary, &run.basis, &result).unwrap();
    assert_eq!(
        recovery.nodal().translation().quantity(),
        ResponseSpectrumNodalQuantity::AbsoluteAcceleration
    );
    assert_eq!(
        recovery.nodal().translation().values(),
        &result.combined_response()[..summary.structural_dof_layout.node_dof_count()]
    );
    assert_eq!(
        recovery.deformation().unavailability(),
        Some(ResponseSpectrumDeformationRecoveryUnavailability::MissingMassStaticCorrectionNotQualified)
    );
    recovery
        .validate_exact_for(&summary, &run.basis, &result)
        .unwrap();

    let direction_result =
        solve_modal_response_spectrum_direction_with_operator(&summary, &run.basis, &spectrum)
            .unwrap();
    let restored_direction = ModalResponseSpectrumDirectionResult::from_canonical_bytes(
        &direction_result.canonical_bytes().unwrap(),
    )
    .unwrap();
    restored_direction
        .validate_for(&summary, &run.basis)
        .unwrap();
    let assembled = assemble_modal_response_spectrum_directions(
        &run.basis,
        std::slice::from_ref(&spectrum),
        &[restored_direction],
    )
    .unwrap();
    assert_eq!(assembled, result);
}

#[test]
fn directional_residual_vector_is_generated_in_the_basis_and_combined_as_a_mode() {
    let truncation = EigenspaceTruncationPolicy::requested(2);
    let mut model = fixture_model(FixtureId::StructuralShellPlateMomentReference);
    let mut step = ProcedureStep::structural_modal("modes");
    step.independent_domain = IndependentDomain::Eigenvalue {
        requested_count: truncation.requested_mode_count(),
    };
    model.sequence.steps = vec![step];
    let run = run_modal_with_options(
        &model,
        ComputeBackend::Cpu,
        ModalSolveOptions {
            formulation_requirement: FormulationRequirement::Experimental,
            truncation,
            convergence: ModalConvergencePolicy::exploratory(),
            residual_vector_directions: vec![SpectrumDirection {
                frame: CoordinateFrame::Global,
                vector: [1.0, 0.5, 0.25],
            }],
            solver_mesh: None,
        },
    )
    .unwrap();
    let excitations = crate::excitation::realize_step_excitations(
        &model,
        &model.sequence.steps[0],
        crate::excitation::ExcitationSamplePoint::Static,
    )
    .unwrap();
    let summary = crate::assembly::assemble_structural_system(
        &model,
        &excitations,
        None,
        StructuralFormulationRequirement::ExperimentalSyntheticOperator,
    )
    .unwrap();
    let direction = [1.0, 0.5, 0.25];
    let basis = run.basis;
    let residual = &basis.residual_vectors()[0];
    assert!(residual.frequency_hz() > basis.modes().last().unwrap().frequency_hz());
    assert!(residual.static_equilibrium_residual_ratio() <= 1.0e-8);
    assert!(residual.maximum_mass_orthogonality_ratio() <= 1.0e-8);
    basis.validate_for(&summary).unwrap();

    let mut spectrum = spectrum_fixture(
        SpectrumAbscissa::FrequencyHertz(vec![
            basis.modes()[0].frequency_hz() * 0.5,
            residual.frequency_hz() * 1.5,
        ]),
        vec![1.0, 1.0],
        direction,
    );
    spectrum.truncation.correction = runmat_analysis_core::ModalTruncationCorrection::ResidualMode;
    let result = solve_modal_response_spectrum(&basis, std::slice::from_ref(&spectrum)).unwrap();
    let contribution = &result.contributions()[0];
    assert_eq!(contribution.truncation.residual_vector_count, 1);
    assert_eq!(
        contribution.modes.last().unwrap().kind,
        ModalResponseSpectrumModeKind::ResidualVector
    );
    result
        .validate_for(&basis, std::slice::from_ref(&spectrum))
        .unwrap();
}

#[test]
fn rigid_response_separates_periodic_and_in_phase_components() {
    let modal = vec![vec![3.0], vec![4.0]];
    let (periodic, rigid) = combine_periodic_and_rigid_responses(
        &modal,
        &[1.0, 2.0],
        &[0.0, 1.0],
        0.05,
        ModalCombination::CompleteQuadraticCombination,
    )
    .unwrap();
    assert_eq!(periodic, vec![3.0]);
    assert_eq!(rigid, vec![4.0]);
    assert_eq!(
        (periodic[0] * periodic[0] + rigid[0] * rigid[0]).sqrt(),
        5.0
    );
}

#[test]
fn linear_field_envelope_recovers_signed_mode_before_statistical_combination() {
    let truncation = EigenspaceTruncationPolicy::requested(2);
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let mut step = ProcedureStep::structural_modal("modes");
    step.independent_domain = IndependentDomain::Eigenvalue {
        requested_count: truncation.requested_mode_count(),
    };
    model.sequence.steps = vec![step];
    let mesh = crate::tests::single_tetrahedron_solver_mesh_for_model(&model);
    let run = run_modal_with_options(
        &model,
        ComputeBackend::Cpu,
        ModalSolveOptions {
            formulation_requirement: FormulationRequirement::Experimental,
            truncation,
            convergence: ModalConvergencePolicy::exploratory(),
            residual_vector_directions: Vec::new(),
            solver_mesh: Some(mesh.clone()),
        },
    )
    .unwrap();
    let excitations = crate::excitation::realize_step_excitations(
        &model,
        &model.sequence.steps[0],
        crate::excitation::ExcitationSamplePoint::Static,
    )
    .unwrap();
    let summary = crate::assembly::assemble_structural_system(
        &model,
        &excitations,
        Some(mesh),
        StructuralFormulationRequirement::ExperimentalSyntheticOperator,
    )
    .unwrap();
    let first_mode = &run.basis.modes()[0];
    let axis = first_mode
        .directional()
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.effective_mass().total_cmp(&right.effective_mass()))
        .map(|(axis, _)| axis)
        .unwrap();
    let mut direction = [0.0; 3];
    direction[axis] = 1.0;
    let frequency = first_mode.frequency_hz();
    let spectrum = spectrum_fixture(
        SpectrumAbscissa::FrequencyHertz(vec![frequency]),
        vec![0.002],
        direction,
    );
    let result = solve_modal_response_spectrum_with_operator(
        &summary,
        &run.basis,
        std::slice::from_ref(&spectrum),
    )
    .unwrap();
    let contribution = &result.contributions()[0].modes[0];
    let signed_displacement = first_mode
        .dof_shape()
        .iter()
        .map(|value| value * contribution.generalized_peak)
        .collect::<Vec<_>>();
    let directly_recovered =
        crate::post::fields::recover_deformation_result_fields(&summary, &signed_displacement)
            .unwrap();
    let expected_stress = directly_recovered
        .iter()
        .find(|field| field.field_id == crate::FEA_FIELD_STRUCTURAL_STRESS)
        .unwrap()
        .as_host_f64()
        .unwrap()
        .iter()
        .map(|value| value.abs())
        .collect::<Vec<_>>();

    let fields = recover_modal_response_spectrum_fields(&summary, &run.basis, &result).unwrap();
    assert_eq!(
        fields
            .field(StructuralLinearFieldQuantity::Stress)
            .unwrap()
            .values(),
        expected_stress
    );
    assert!(fields
        .field(StructuralLinearFieldQuantity::Strain)
        .is_some());
    assert!(fields
        .field(StructuralLinearFieldQuantity::Displacement)
        .is_some());
    assert!(fields.fields().unwrap().iter().all(|field| !matches!(
        field.quantity(),
        StructuralLinearFieldQuantity::ShellMembraneForce
            | StructuralLinearFieldQuantity::ShellBendingMoment
            | StructuralLinearFieldQuantity::ShellTransverseShear
    )));
    fields
        .validate_exact_for(&summary, &run.basis, &result)
        .unwrap();
    let restored =
        ResponseSpectrumFieldRecovery::from_canonical_bytes(&fields.canonical_bytes().unwrap())
            .unwrap();
    assert_eq!(restored, fields);

    let mut pseudo_acceleration = spectrum;
    pseudo_acceleration.response = SpectrumResponse::PseudoAcceleration;
    pseudo_acceleration.ordinates = vec![0.002 * (std::f64::consts::TAU * frequency).powi(2)];
    pseudo_acceleration.provenance_digest = "sha256:equivalent-pseudo-acceleration".into();
    let pseudo_result = solve_modal_response_spectrum_with_operator(
        &summary,
        &run.basis,
        std::slice::from_ref(&pseudo_acceleration),
    )
    .unwrap();
    let pseudo_fields =
        recover_modal_response_spectrum_fields(&summary, &run.basis, &pseudo_result).unwrap();
    assert_eq!(
        pseudo_fields.fields().unwrap().len(),
        fields.fields().unwrap().len()
    );
    for (pseudo, displacement) in pseudo_fields
        .fields()
        .unwrap()
        .iter()
        .zip(fields.fields().unwrap())
    {
        assert_eq!(pseudo.quantity(), displacement.quantity());
        assert_eq!(pseudo.shape(), displacement.shape());
        assert!(pseudo
            .values()
            .iter()
            .zip(displacement.values())
            .all(|(left, right)| (left - right).abs() <= 1.0e-12 * right.abs().max(1.0)));
    }
}

#[test]
fn direct_acceleration_field_is_retained_when_deformation_is_unavailable() {
    let truncation = EigenspaceTruncationPolicy::requested(1);
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let mut step = ProcedureStep::structural_modal("modes");
    step.independent_domain = IndependentDomain::Eigenvalue {
        requested_count: truncation.requested_mode_count(),
    };
    model.sequence.steps = vec![step];
    let mesh = crate::tests::single_tetrahedron_solver_mesh_for_model(&model);
    let run = run_modal_with_options(
        &model,
        ComputeBackend::Cpu,
        ModalSolveOptions {
            formulation_requirement: FormulationRequirement::Experimental,
            truncation,
            convergence: ModalConvergencePolicy::exploratory(),
            residual_vector_directions: Vec::new(),
            solver_mesh: Some(mesh.clone()),
        },
    )
    .unwrap();
    let excitations = crate::excitation::realize_step_excitations(
        &model,
        &model.sequence.steps[0],
        crate::excitation::ExcitationSamplePoint::Static,
    )
    .unwrap();
    let summary = crate::assembly::assemble_structural_system(
        &model,
        &excitations,
        Some(mesh),
        StructuralFormulationRequirement::ExperimentalSyntheticOperator,
    )
    .unwrap();
    let first_mode = &run.basis.modes()[0];
    let axis = first_mode
        .directional()
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.effective_mass().total_cmp(&right.effective_mass()))
        .map(|(axis, _)| axis)
        .unwrap();
    let mut direction = [0.0; 3];
    direction[axis] = 1.0;
    let mut spectrum = spectrum_fixture(
        SpectrumAbscissa::FrequencyHertz(vec![first_mode.frequency_hz()]),
        vec![1.0],
        direction,
    );
    spectrum.response = SpectrumResponse::AbsoluteAcceleration;
    let result = solve_modal_response_spectrum_with_operator(
        &summary,
        &run.basis,
        std::slice::from_ref(&spectrum),
    )
    .unwrap();
    let recovery = recover_modal_response_spectrum_fields(&summary, &run.basis, &result).unwrap();
    assert_eq!(recovery.response(), SpectrumResponse::AbsoluteAcceleration);
    assert_eq!(
        recovery.nodal().translation().quantity(),
        ResponseSpectrumNodalQuantity::AbsoluteAcceleration
    );
    assert_eq!(
        recovery.nodal().translation().values(),
        &result.combined_response()[..summary.structural_dof_layout.node_dof_count()]
    );
    assert!(recovery.nodal().rotation().is_none());
    assert_eq!(
        recovery.deformation().unavailability(),
        Some(
            ResponseSpectrumDeformationRecoveryUnavailability::ResponseQuantityNotDisplacementConvertible {
                response: SpectrumResponse::AbsoluteAcceleration,
            }
        )
    );
    recovery
        .validate_exact_for(&summary, &run.basis, &result)
        .unwrap();

    for response in [
        SpectrumResponse::RelativeDisplacement,
        SpectrumResponse::RelativeVelocity,
        SpectrumResponse::RelativeAcceleration,
        SpectrumResponse::AbsoluteAcceleration,
        SpectrumResponse::PseudoVelocity,
        SpectrumResponse::PseudoAcceleration,
    ] {
        let mut response_spectrum = spectrum.clone();
        response_spectrum.response = response;
        let response_result = solve_modal_response_spectrum_with_operator(
            &summary,
            &run.basis,
            std::slice::from_ref(&response_spectrum),
        )
        .unwrap();
        let response_recovery =
            recover_modal_response_spectrum_fields(&summary, &run.basis, &response_result).unwrap();
        assert_eq!(response_recovery.response(), response);
        assert_eq!(
            response_recovery.nodal().translation().quantity(),
            ResponseSpectrumNodalQuantity::for_translation(response)
        );
        assert_eq!(
            response_recovery.nodal().translation().values(),
            &response_result.combined_response()[..summary.structural_dof_layout.node_dof_count()]
        );
        assert_eq!(
            response_recovery.deformation().fields().is_some(),
            ResponseSpectrumDisplacementConversion::for_response(response).is_some()
        );
        let restored = ResponseSpectrumFieldRecovery::from_canonical_bytes(
            &response_recovery.canonical_bytes().unwrap(),
        )
        .unwrap();
        assert_eq!(restored, response_recovery);
    }
}

#[test]
fn rotational_layout_publishes_typed_angular_envelopes() {
    let truncation = EigenspaceTruncationPolicy::requested(6);
    let mut model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
    let mut step = ProcedureStep::structural_modal("beam_modes");
    step.independent_domain = IndependentDomain::Eigenvalue {
        requested_count: truncation.requested_mode_count(),
    };
    model.sequence.steps = vec![step];
    let run = run_modal_with_options(
        &model,
        ComputeBackend::Cpu,
        ModalSolveOptions {
            formulation_requirement: FormulationRequirement::Experimental,
            truncation,
            convergence: ModalConvergencePolicy::exploratory(),
            residual_vector_directions: Vec::new(),
            solver_mesh: None,
        },
    )
    .unwrap();
    let excitations = crate::excitation::realize_step_excitations(
        &model,
        &model.sequence.steps[0],
        crate::excitation::ExcitationSamplePoint::Static,
    )
    .unwrap();
    let summary = crate::assembly::assemble_structural_system(
        &model,
        &excitations,
        None,
        StructuralFormulationRequirement::ExperimentalSyntheticOperator,
    )
    .unwrap();
    let first_mode = &run.basis.modes()[0];
    let axis = first_mode
        .directional()
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.effective_mass().total_cmp(&right.effective_mass()))
        .map(|(axis, _)| axis)
        .unwrap();
    let mut direction = [0.0; 3];
    direction[axis] = 1.0;
    let mut spectrum = spectrum_fixture(
        SpectrumAbscissa::FrequencyHertz(vec![first_mode.frequency_hz()]),
        vec![1.0],
        direction,
    );
    for response in [
        SpectrumResponse::RelativeDisplacement,
        SpectrumResponse::RelativeVelocity,
        SpectrumResponse::RelativeAcceleration,
        SpectrumResponse::AbsoluteAcceleration,
        SpectrumResponse::PseudoVelocity,
        SpectrumResponse::PseudoAcceleration,
    ] {
        spectrum.response = response;
        let result = solve_modal_response_spectrum_with_operator(
            &summary,
            &run.basis,
            std::slice::from_ref(&spectrum),
        )
        .unwrap();
        let recovery =
            recover_modal_response_spectrum_fields(&summary, &run.basis, &result).unwrap();
        assert_eq!(
            recovery.nodal().translation().quantity(),
            ResponseSpectrumNodalQuantity::for_translation(response)
        );
        let rotation = recovery.nodal().rotation().unwrap();
        assert_eq!(
            rotation.quantity(),
            ResponseSpectrumNodalQuantity::for_rotation(response)
        );
        assert_eq!(rotation.shape(), recovery.nodal().translation().shape());
        use crate::assembly::dofs::StructuralDofKind::{Rx, Ry, Rz, Ux, Uy, Uz};
        for node in 0..summary.structural_dof_layout.node_count() {
            for (component, (translation_kind, rotation_kind)) in
                [(Ux, Rx), (Uy, Ry), (Uz, Rz)].into_iter().enumerate()
            {
                let field_index = node * 3 + component;
                let translation_index = summary
                    .structural_dof_layout
                    .index(node, translation_kind)
                    .unwrap();
                let rotation_index = summary
                    .structural_dof_layout
                    .index(node, rotation_kind)
                    .unwrap();
                assert_eq!(
                    recovery.nodal().translation().values()[field_index],
                    result.combined_response()[translation_index]
                );
                assert_eq!(
                    rotation.values()[field_index],
                    result.combined_response()[rotation_index]
                );
            }
        }
        assert_eq!(
            recovery.deformation().fields().is_some(),
            ResponseSpectrumDisplacementConversion::for_response(response).is_some()
        );
    }
}

fn spectrum_fixture(
    abscissa: SpectrumAbscissa,
    ordinates: Vec<f64>,
    direction: [f64; 3],
) -> ResponseSpectrum {
    ResponseSpectrum {
        spectrum_id: ResponseSpectrumId("design".into()),
        abscissa,
        ordinates,
        damping: SpectrumDamping::Ratio(0.05),
        response: SpectrumResponse::RelativeDisplacement,
        envelope: SpectrumEnvelope::Maximax,
        interpolation: SpectrumInterpolation::LinearFrequency,
        extrapolation: SpectrumExtrapolation::Reject,
        direction: SpectrumDirection {
            frame: CoordinateFrame::Global,
            vector: direction,
        },
        modal_combination: ModalCombination::CompleteQuadraticCombination,
        directional_combination: DirectionalCombination::Srss,
        truncation: runmat_analysis_core::ModalTruncationPolicy {
            minimum_cumulative_mass_ratio: 0.0,
            correction: runmat_analysis_core::ModalTruncationCorrection::None,
        },
        rigid_response: runmat_analysis_core::RigidResponsePolicy::Disabled,
        construction: SpectrumConstruction::AuthoredDesign,
        evidence: SpectrumEvidence {
            sampling_adequacy_ratio: 1.0,
            max_interpolation_error: None,
        },
        provenance_digest: "sha256:design".into(),
    }
}
