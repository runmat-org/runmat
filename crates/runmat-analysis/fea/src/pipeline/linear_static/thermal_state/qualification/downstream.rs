//! Retained thermal-prestress consumer regressions; not complete independent qualification.

use super::*;
use crate::execution::{
    StructuralBucklingSolverPolicy, StructuralBucklingTemplate, StructuralFrequencyEntryRequest,
    StructuralPrestressedFrequencySolverTemplate, StructuralPrestressedFrequencyTemplate,
};
use crate::pipeline::structural_frequency::{
    execute_prestressed_structural_frequency_entry, prepare_prestressed_structural_frequency,
};
use runmat_execution::Digest;

fn transition(
    model: &AnalysisModel,
    mesh: &SolverMeshArtifact,
    state: &CanonicalSemanticState,
) -> CanonicalStateTransition {
    CanonicalStateTransition::bind_upstream(
        model,
        &model.sequence.steps[2].step_id,
        state,
        StateUse::Prestress,
        Digest::from_bytes(*mesh.canonical_digest.bytes()),
    )
    .unwrap()
}

#[test]
fn solved_thermal_buckling_factor_is_inverse_to_heating_preload() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let solve = |delta| {
            let mut step = ProcedureStep::structural_buckling(
                "thermal_buckling",
                "qualification_expansion",
                "geometric_stiffness",
            );
            step.independent_domain = IndependentDomain::Eigenvalue { requested_count: 1 };
            let (model, mesh, state) =
                modal::thermal_preload(order, delta, step.clone(), |_, _| {});
            let retained = state.clone();
            let bound = transition(&model, &mesh, &state);
            let workload = StructuralBucklingTemplate {
                schema_version: 1,
                model,
                step_id: step.step_id,
                backend: ComputeBackend::Cpu,
                observation_execution:
                    crate::results::observation::ObservationExecutionPolicy::scalable(),
                solver: StructuralBucklingSolverPolicy::new(1, 1.0e-9, 100).unwrap(),
            }
            .bind(bound)
            .unwrap();
            let result = crate::execute_structural_buckling_workload(&workload, &mesh, &state);
            assert_eq!(state, retained);
            result
        };
        let base = solve(DELTA).unwrap();
        let doubled = solve(2.0 * DELTA).unwrap();
        assert_eq!(base.modes().len(), 1);
        assert_eq!(doubled.modes().len(), 1);
        let first = base.modes()[0].load_factor;
        let second = doubled.modes()[0].load_factor;
        assert!(first > 0.0 && second > 0.0);
        assert!(
            (2.0 * second / first - 1.0).abs() < 1.0e-6,
            "{order:?}: {first}, {second}"
        );
        assert_eq!(
            crate::execution::StructuralBucklingSolverResult::from_canonical_bytes(
                &base.canonical_bytes().unwrap()
            )
            .unwrap(),
            base
        );
        // The current solver publishes positive compressive buckling factors only.
        // Zero/tensile initial stress cannot be advertised as a converged positive mode.
        for delta in [0.0, -DELTA] {
            match solve(delta) {
                Ok(result) => assert!(
                    result.modes().is_empty(),
                    "{order:?}: unexpected buckling for delta {delta}"
                ),
                Err(FeaRunError::Solve(message)) => assert!(
                    message.contains("retained 0/1 positive modes")
                        || message.contains("K-orthonormal subspace"),
                    "{order:?}: unexpected rejection {message}"
                ),
                Err(error) => panic!("{order:?}: unrelated buckling rejection: {error}"),
            }
        }
    }
}

fn configure_probe(model: &mut AnalysisModel, mesh: &SolverMeshArtifact) {
    use crate::fixtures::{replace_fixture_excitations, FixtureExcitation, FixtureExcitationKind};
    // Exercise a genuinely complex response, not just a real undamped solve with zero phase.
    model.materials[0].damping = Some(MechanicalDampingModel::Rayleigh {
        mass_coefficient: RayleighMassCoefficient::new(0.2),
        stiffness_coefficient: RayleighStiffnessCoefficient::new(2.0e-4),
    });
    let projection = crate::topology::AnalysisProjection::derive(mesh).unwrap();
    let inclined = projection
        .boundary_faces
        .iter()
        .find(|face| {
            face.node_indices.iter().all(|&index| {
                (mesh.topology.nodes[index]
                    .coordinates_m
                    .canonical_value()
                    .iter()
                    .sum::<f64>()
                    - 1.0)
                    .abs()
                    < 1.0e-12
            })
        })
        .unwrap();
    let selection = inclined.provenance[0].analysis_selection_id().unwrap();
    replace_fixture_excitations(
        model,
        vec![FixtureExcitation {
            id: "harmonic_probe".into(),
            region_id: selection.clone(),
            kind: FixtureExcitationKind::Force([0.0, 1000.0, 0.0]),
        }],
    );
    let ExcitationAction::Force(probe) = &mut model.excitations[0] else {
        unreachable!()
    };
    probe.target = ExcitationTarget::Region(RegionTarget::boundary(selection));
    probe.distribution = ExcitationDistribution::PreserveResultant;
    // Explicit activation prevents the harmonic probe from contaminating the thermal/static
    // producer. Preserve the thermal boundary while selecting only structural constraints for
    // the perturbation step; global material/reference-temperature declarations stay untouched.
    let structural: Vec<_> = model
        .constraints
        .iter()
        .filter_map(|constraint| match constraint {
            AnalysisConstraint::Structural(c) => Some(StepObjectActivation {
                object: StepObjectRef::Constraint(c.constraint_id.clone()),
                lifecycle: ActivationLifecycle::Activate,
            }),
            _ => None,
        })
        .collect();
    let thermal: Vec<_> = model
        .constraints
        .iter()
        .filter_map(|constraint| match constraint {
            AnalysisConstraint::Thermal(c) => Some(StepObjectActivation {
                object: StepObjectRef::Constraint(c.constraint_id.clone()),
                lifecycle: ActivationLifecycle::Activate,
            }),
            _ => None,
        })
        .collect();
    model.sequence.steps[0].activations = thermal;
    model.sequence.steps[1].activations = structural.clone();
    model.sequence.steps[2].activations = structural;
    model.sequence.steps[2]
        .activations
        .push(StepObjectActivation {
            object: StepObjectRef::Excitation("harmonic_probe".into()),
            lifecycle: ActivationLifecycle::Activate,
        });
}

#[test]
fn solved_thermal_harmonic_response_matches_dense_retained_tangent_system() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let hz = 60.0;
        let mut step = ProcedureStep::structural_prestressed_harmonic(
            "thermal_harmonic",
            "qualification_expansion",
            "geometric_stiffness",
        );
        step.independent_domain = IndependentDomain::FrequencyHertz(vec![hz]);
        let (model, mesh, state) =
            modal::thermal_preload(order, DELTA, step.clone(), configure_probe);
        let retained = state.clone();
        let bound = transition(&model, &mesh, &state);
        let workload = StructuralPrestressedFrequencyTemplate {
            schema_version: 1,
            model,
            step_id: step.step_id,
            backend: ComputeBackend::Cpu,
            observation_execution:
                crate::results::observation::ObservationExecutionPolicy::scalable(),
            plan: crate::frequency::FrequencyResponsePlan::new(hz, &[], &[]).unwrap(),
            solver: StructuralPrestressedFrequencySolverTemplate::Direct {
                policy: crate::solve::dynamics::DirectHarmonicPolicy::default(),
            },
        }
        .bind(bound, None)
        .unwrap();
        let prepared = prepare_prestressed_structural_frequency(&workload, &mesh, &state).unwrap();
        let request = StructuralFrequencyEntryRequest::new(
            &workload,
            prepared.study,
            prepared.step,
            prepared.execution_plan,
            prepared.scope,
            workload.plan().entries().unwrap()[0],
        )
        .unwrap();
        let result = execute_prestressed_structural_frequency_entry(
            &workload, &mesh, &request, &state, None,
        )
        .unwrap();
        assert_eq!(state, retained);
        let assembly =
            crate::assembly::complex_structural::assemble_complex_structural_perturbation_system(
                workload.model(),
                workload.procedure_step().unwrap(),
                hz,
                Some(mesh.clone()),
                crate::assembly::StructuralFormulationRequirement::PhysicalDiscretization,
            )
            .unwrap();
        let system = &assembly.real.operator;
        let free: Vec<_> = system
            .constrained_dofs()
            .iter()
            .enumerate()
            .filter_map(|(i, fixed)| (!fixed).then_some(i))
            .collect();
        let n = free.len();
        assert!(n > 0);
        let omega = 2.0 * std::f64::consts::PI * hz;
        let geometric = state
            .payload
            .structural_linearization()
            .unwrap()
            .geometric_stiffness();
        let operators = system.operators();
        let mut dense = vec![vec![0.0; 2 * n]; 2 * n];
        let mut rhs = vec![0.0; 2 * n];
        for (i, &row) in free.iter().enumerate() {
            rhs[i] = system.load()[row];
            rhs[n + i] = assembly.imaginary_load[row];
            for (j, &column) in free.iter().enumerate() {
                let real = operators.stiffness().get(row, column) + geometric.get(row, column)
                    - omega * omega * operators.mass().get(row, column);
                let imaginary = omega * operators.damping().get(row, column);
                dense[i][j] = real;
                dense[n + i][n + j] = real;
                dense[i][n + j] = -imaginary;
                dense[n + i][j] = imaginary;
            }
        }
        // Independent dense partial-pivot elimination; only elastic assembly and published Kg
        // are shared. This verifies transfer/operator consumption, not independent element laws.
        let reference = dense_solve(dense, rhs);
        let actual = result.sample().displacement();
        let scale = reference.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        assert!(scale > 0.0);
        assert!(reference[n..]
            .iter()
            .any(|value| value.abs() > 1.0e-4 * scale));
        for (i, &row) in free.iter().enumerate() {
            assert!((actual.real()[row] - reference[i]).abs() < 1.0e-7 * scale);
            assert!((actual.imaginary()[row] - reference[n + i]).abs() < 1.0e-7 * scale);
        }
        assert_eq!(
            crate::execution::StructuralFrequencyEntryResult::from_canonical_bytes(
                &result.canonical_bytes().unwrap()
            )
            .unwrap(),
            result
        );
    }
}

fn dense_solve(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Vec<f64> {
    let n = rhs.len();
    for column in 0..n {
        let pivot = (column..n)
            .max_by(|&a, &b| matrix[a][column].abs().total_cmp(&matrix[b][column].abs()))
            .unwrap();
        assert!(matrix[pivot][column].is_finite() && matrix[pivot][column] != 0.0);
        matrix.swap(column, pivot);
        rhs.swap(column, pivot);
        for row in column + 1..n {
            let factor = matrix[row][column] / matrix[column][column];
            for j in column..n {
                matrix[row][j] -= factor * matrix[column][j];
            }
            rhs[row] -= factor * rhs[column];
        }
    }
    let mut result = vec![0.0; n];
    for row in (0..n).rev() {
        result[row] = (rhs[row]
            - (row + 1..n)
                .map(|j| matrix[row][j] * result[j])
                .sum::<f64>())
            / matrix[row][row];
        assert!(result[row].is_finite());
    }
    result
}
