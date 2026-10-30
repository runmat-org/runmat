use super::*;
mod certificate;
mod closure_diagnostics;
mod periodic;
mod transmission_loss;
use crate::{
    contracts::SolverPreconditioner,
    fixtures::{fixture_model, replace_fixture_excitations, FixtureId},
    topology::test_meshes::{quadratic_slab_mesh, slab_mesh},
};
use runmat_analysis_core::{
    AcousticConstraint, AcousticConstraintKind, AnalysisBoundary, AnalysisConstraint,
    BoundaryOperator, IndependentDomain, MaterialAcousticModel, PrescribedValue, Pressure,
    PropertyRegionTarget, RegionTarget, SpecificAcousticImpedance, TemporalImplementation,
};

pub(in crate::execution) fn workload() -> AcousticHarmonicWorkload {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    replace_fixture_excitations(&mut model, Vec::new());
    model.constraints.clear();
    model.boundaries.clear();
    model.materials[0].acoustic = Some(MaterialAcousticModel {
        speed_of_sound_m_per_s: runmat_types::physical::SpeedOfSound::new(2.0),
        damping_ratio: runmat_types::physical::DampingRatio::new(0.0),
    });
    model.materials[0].mass_density_kg_per_m3 =
        (Some(3.0)).map(runmat_types::physical::MassDensity::new);
    model.material_assignments[0].target =
        PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_solid("slab"));
    let mut step = ProcedureStep::acoustic_harmonic("acoustic");
    step.independent_domain = IndependentDomain::FrequencyHertz(vec![0.2]);
    model.sequence.steps = vec![step];
    model
        .constraints
        .push(AnalysisConstraint::Acoustic(AcousticConstraint {
            constraint_id: "pressure".into(),
            kind: AcousticConstraintKind::PrescribedPressure {
                target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                    "slab_left",
                )),
                value: PrescribedValue::Constant(Pressure::new(1.0)),
            },
        }));
    model.boundaries.push(AnalysisBoundary {
        boundary_id: "matched".into(),
        target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("slab_right")),
        operator: BoundaryOperator::AcousticImpedance {
            impedance: SpecificAcousticImpedance::new(6.0),
        },
    });
    AcousticHarmonicWorkload {
        model,
        step_id: ProcedureStepId("acoustic".into()),
        frequency: Frequency::new(0.2),
        solver: AcousticHarmonicPolicy {
            relative_tolerance: 1e-10,
            ..Default::default()
        },
        power_balance_relative_tolerance: 1e-8,
    }
}

#[test]
fn cold_canonical_acoustic_objects_preserve_independent_fields_and_exact_lineage() {
    for quadratic in [false, true] {
        let mesh = if quadratic {
            quadratic_slab_mesh(2)
        } else {
            slab_mesh(4)
        };
        let authored = workload();
        let bytes = authored.canonical_bytes().unwrap();
        let cold = AcousticHarmonicWorkload::from_canonical_bytes(&bytes).unwrap();
        let mesh: SolverMeshArtifact =
            serde_json::from_slice(&serde_json::to_vec(&mesh).unwrap()).unwrap();
        let result = cold.execute(&mesh).unwrap();
        assert_eq!(
            result.formulation().realization().temporals,
            vec![TemporalImplementation::AcousticCanonicalHarmonicSolve]
        );
        let result_bytes = result.canonical_bytes().unwrap();
        let imported = AcousticHarmonicSolverResult::from_canonical_bytes(&result_bytes).unwrap();
        imported.validate_for(&authored, &mesh).unwrap();
        assert_eq!(result, imported);
        assert_eq!(
            result.source_result_identity().unwrap(),
            imported.source_result_identity().unwrap()
        );
        let mut error = 0.0;
        for (node, p) in mesh.topology.nodes.iter().zip(imported.pressure_pa()) {
            let phase = std::f64::consts::PI * 0.2 * node.coordinates_m.canonical_value()[0];
            error += (p.real - phase.cos()).powi(2) + (p.imaginary + phase.sin()).powi(2);
        }
        assert!((error / imported.pressure_pa().len() as f64).sqrt() < 0.01);
        assert_eq!(imported.ordered_node_ids().len(), mesh.topology.nodes.len());
        assert_eq!(
            imported.ordered_element_ids().len(),
            mesh.topology.volume_elements.len()
        );
        assert!(imported
            .intensity_w_per_m2()
            .iter()
            .all(|i| (i[0] - 1.0 / 12.0).abs() < 0.01));
        assert!(imported
            .energy_density_j_per_m3()
            .iter()
            .all(|e| (e - 1.0 / 24.0).abs() < 0.01));
        let mut other = cold.clone();
        if let AnalysisConstraint::Acoustic(c) = &mut other.model.constraints[0] {
            if let AcousticConstraintKind::PrescribedPressure { value, .. } = &mut c.kind {
                *value = PrescribedValue::Constant(Pressure::new(2.0));
            }
        }
        assert_ne!(cold.identity().unwrap(), other.identity().unwrap());
        assert!(imported.validate_for(&other, &mesh).is_err());
        let mut other_mesh = mesh.clone();
        for node in &mut other_mesh.topology.nodes {
            {
                let quantity_change = 1.0;
                let quantity_index = 0;
                let quantity_slot = &mut node.coordinates_m;
                let mut quantity_coordinates = quantity_slot.canonical_value();
                quantity_coordinates[quantity_index] += quantity_change;
                *quantity_slot = runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new(quantity_coordinates);
            };
        }
        other_mesh.seal_canonical_digest().unwrap();
        assert!(imported.validate_for(&cold, &other_mesh).is_err());
    }
}

#[test]
fn acoustic_portable_admission_and_result_import_fail_closed() {
    let mut workload = workload();
    let mesh = slab_mesh(2);
    let result = workload.execute(&mesh).unwrap();
    assert!(validate_solver_formulation_compatibility(
        result.formulation(),
        SolverMethod::AcousticDomainGraphHelmholtzHarmonic,
        SolverPreconditioner::None
    )
    .is_err());
    for (field, value) in [
        ("relative_residual", serde_json::json!(0.5)),
        ("power_relative_error", serde_json::json!(0.5)),
        ("energy_density_j_per_m3", serde_json::json!([-1.0])),
        ("pressure_pa", serde_json::json!([])),
        ("ordered_node_ids", serde_json::json!([1, 1])),
    ] {
        let mut malformed = serde_json::to_value(&result).unwrap();
        malformed[field] = value;
        assert!(
            serde_json::from_value::<AcousticHarmonicSolverResult>(malformed).is_err(),
            "{field}"
        );
    }
    workload.frequency = Frequency::new(0.3);
    assert!(workload.validate().is_err());
    workload.frequency = Frequency::new(0.2);
    let mut wrong_state = workload.clone();
    wrong_state.model.sequence.steps[0].state_outputs[0].kind = StateKind::Thermal;
    assert!(wrong_state.validate().is_err());
    workload
        .model
        .sequence
        .steps
        .push(workload.model.sequence.steps[0].clone());
    assert!(workload.canonical_bytes().is_err());
}

// witness: FEA-ACOUSTIC-OUTPUTS-001/intensity_and_sound_power @ acoustic_harmonic/tet10/numerical_kernel;
//   coverage: partial; boundary: plane-wave power through a matched duct end equals |p|²A/(2ρc) and
//   the accounted input power to 0.1 %; the monopole sphere reference is not run.
#[test]
fn sound_power_through_the_matched_end_is_the_plane_wave_intensity_times_area() {
    use crate::results::{
        field::RecoveredFieldCatalog,
        observation::{evaluate_observations, resolve_observation_plan, ObservationLimits},
    };
    use runmat_analysis_core::*;
    // p = e^{-ikx} of 1 Pa into ρc = 6 Pa·s/m: I = |p|²/(2ρc) = 1/12 W/m² over the unit end.
    let mut authored = workload();
    authored.model.sequence.steps[0]
        .observations
        .push(ObservationRequest::Reduction(ObservationSpec {
            observation_id: ObservationId("power".into()),
            quantity: ObservationQuantity::AcousticIntensity,
            value_shape: ObservationValueShape::Vector3,
            target: ObservationTarget::Spatial(ExcitationTarget::Region(RegionTarget::boundary(
                runmat_meshing_core::fixtures::fixture_face("slab_right"),
            ))),
            source_location: FieldLocation::Element,
            output_location: FieldLocation::Global,
            recovery: RecoveryPolicy::Native,
            component: ObservationComponent::All,
            frame: CoordinateFrame::Global,
            sign_convention: SignConvention::Native,
            complex_representation: ComplexRepresentation::Real,
            sampling: SamplingPolicy::Final,
            reduction: Some(ReductionPolicy {
                spatial: SpatialReduction::NormalFlux,
                series: SeriesReduction::None,
                extrema_ties: ExtremaTiePolicy::FirstCanonical,
            }),
            storage: ObservationStoragePolicy {
                precision: SignalPrecision::Real64,
                compression: CompressionPolicy::Lossless,
                retention: RetentionPolicy::Canonical,
                partial_access: PartialAccessPolicy::ManifestAndChunks,
            },
            purpose: ObservationPurpose::Publication,
            cost: ObservationCostPolicy::UpperBound(ObservationCostEstimate {
                operation_count: 1_000_000,
                peak_resident_bytes: 10_000_000,
                stored_bytes: 1_000_000,
                transfer_bytes: 1_000_000,
                collective_count: 0,
            }),
        }));
    let mesh = quadratic_slab_mesh(2);
    let result = authored.execute(&mesh).unwrap();
    let request = result.native_field_request(&authored, &mesh).unwrap();
    let artifact = result
        .recover_native_fields(&authored, &mesh, &request)
        .unwrap();
    let definitions = artifact
        .fields()
        .iter()
        .map(|field| field.definition().clone())
        .collect::<Vec<_>>();
    let projection = crate::topology::AnalysisProjection::derive(&mesh).unwrap();
    let catalog = RecoveredFieldCatalog::new_for_topology(
        projection.projection_digest,
        artifact.entity_counts(),
        artifact.into_fields(),
    )
    .unwrap();
    let plan = resolve_observation_plan(
        &authored.model.sequence.steps[0],
        &projection,
        &definitions,
        ObservationLimits::default(),
    )
    .unwrap();
    let observed = evaluate_observations(&plan, &projection, &catalog).unwrap();
    let power = observed[0].field.as_host_f64().unwrap()[0];
    assert!((power - 1.0 / 12.0).abs() < 1e-3 / 12.0, "{power}");
    // The input power the solver accounts for leaves through the matched end.
    assert!((power - result.mean_input_power_w()).abs() < 1e-3 * power);
}
