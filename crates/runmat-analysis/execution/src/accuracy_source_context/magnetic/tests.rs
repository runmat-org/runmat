//! Source authority regressions using real retained fields; not native-worker qualification.
use super::*;
use crate::prepare_analysis_objects;
use runmat_analysis_core::*;
use runmat_analysis_fea::{
    execution::{AccuracySourceKind, AccuracySourceReceipt, ObservationEvaluationRequest},
    fixtures::{fixture_model, FixtureId},
    results::{
        field::{RecoveredFieldCatalog, SampledVolumeFieldTopology, VolumeFieldSamplingRule},
        observation::{
            evaluate_sampled_observations, resolve_sampled_observation_plan, ObservationArtifact,
            ObservationArtifactInput, ObservationFieldCatalog, ObservationFieldSelection,
            ObservationLimits, ObservationPublicationEntry, ObservationSamplePublication,
            ObservationSelectionCoordinate,
        },
    },
};
use runmat_execution_artifact::{object::ObjectInventoryLimits, ObjectNamespace};
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

fn workload() -> MagnetostaticWorkload {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.structural = None;
    model.electromagnetic = Some(ElectromagneticDomain::Magnetostatic {
        formulation: MagnetostaticFormulation::MagneticVectorPotential,
    });
    model.signals.clear();
    model.excitations.clear();
    model.excitation_cases.clear();
    model.constraints.clear();
    model.boundaries.clear();
    model.initial_conditions.clear();
    model.sequence = AnalysisSequence::from_steps(
        "magnetic-bridge",
        vec![ProcedureStep::electromagnetic_static("potential")],
    );
    model.sequence.steps[0].state_outputs = vec![StateOutput {
        state_id: StateId("retained".into()),
        kind: StateKind::ElectromagneticStatic,
        completeness: StateCompleteness::Complete,
        persistence: StatePersistence::Canonical,
    }];
    model.material_assignments[0].target =
        PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_solid("solid"));
    model.materials[0].electrical = Some(MaterialElectricalModel {
        reference_temperature_k: runmat_types::physical::Temperature::new(293.15),
        conductivity_s_per_m: runmat_types::physical::ElectricalConductivity::new(0.0),
        relative_permittivity: runmat_types::physical::RelativePermittivity::new(1.0),
        relative_permeability: runmat_types::physical::RelativePermeability::new(1.0),
        resistive_heating_coefficient: runmat_types::physical::InverseTemperature::new(0.0),
        conductivity_frequency_response: Vec::new(),
    });
    for face in 0..4 {
        model
            .constraints
            .push(AnalysisConstraint::Magnetic(MagneticConstraint {
                constraint_id: format!("ground:{face}"),
                kind: MagneticConstraintKind::VectorPotentialGround {
                    target: RegionTarget::boundary(
                        runmat_meshing_core::fixtures::fixture_boundary_selection(
                            &(format!("face:{face}")),
                        ),
                    ),
                },
            }));
    }
    model
        .constraints
        .push(AnalysisConstraint::Magnetic(MagneticConstraint {
            constraint_id: "gauge".into(),
            kind: MagneticConstraintKind::CoulombGauge {
                target: RegionTarget::volume(runmat_meshing_core::fixtures::fixture_solid("solid")),
            },
        }));
    MagnetostaticWorkload {
        model,
        step_id: ProcedureStepId("potential".into()),
        relative_residual_target: 1e-10,
    }
}

fn observation() -> ObservationRequest {
    ObservationRequest::Field(ObservationSpec {
        observation_id: ObservationId("B".into()),
        quantity: ObservationQuantity::MagneticFluxDensity,
        value_shape: ObservationValueShape::Vector3,
        target: ObservationTarget::WholeModel,
        source_location: FieldLocation::IntegrationPoint,
        output_location: FieldLocation::IntegrationPoint,
        recovery: RecoveryPolicy::Native,
        component: ObservationComponent::All,
        frame: CoordinateFrame::Global,
        sign_convention: SignConvention::Native,
        complex_representation: ComplexRepresentation::Real,
        sampling: SamplingPolicy::Final,
        reduction: None,
        storage: ObservationStoragePolicy {
            precision: SignalPrecision::Real64,
            compression: CompressionPolicy::Lossless,
            retention: RetentionPolicy::Canonical,
            partial_access: PartialAccessPolicy::ManifestAndChunks,
        },
        purpose: ObservationPurpose::Publication,
        cost: ObservationCostPolicy::UpperBound(ObservationCostEstimate {
            operation_count: 1_000_000,
            peak_resident_bytes: 16 * 1024 * 1024,
            stored_bytes: 16 * 1024 * 1024,
            transfer_bytes: 16 * 1024 * 1024,
            collective_count: 0,
        }),
    })
}

fn objects(kind: Kind, identity: Digest, bytes: Vec<u8>) -> PreparedAnalysisObjects {
    prepare_analysis_objects(
        kind,
        identity,
        &bytes,
        1024,
        ObjectNamespace::InputValue,
        ObjectInventoryLimits::default(),
    )
    .unwrap()
}

fn context(
    workload: &MagnetostaticWorkload,
    mesh: &SolverMeshArtifact,
    result: &MagnetostaticSolverResult,
) -> [PreparedAnalysisObjects; 3] {
    [
        objects(
            Kind::MagnetostaticWorkload,
            workload.identity().unwrap().digest(),
            workload.canonical_bytes().unwrap(),
        ),
        objects(
            Kind::Mesh,
            Digest::from_bytes(*mesh.canonical_digest.bytes()),
            mesh.canonical_encode().unwrap(),
        ),
        objects(
            Kind::SolverResult,
            result.source_result_identity().unwrap().digest(),
            result.canonical_bytes().unwrap(),
        ),
    ]
}

fn catalog(
    workload: &MagnetostaticWorkload,
    mesh: &SolverMeshArtifact,
    result: &MagnetostaticSolverResult,
) -> ObservationPublicationCatalog {
    let recovery = result.native_field_request(workload, mesh).unwrap();
    let fields = result
        .recover_native_fields(workload, mesh, &recovery)
        .unwrap();
    let manifest = fields.manifest().unwrap();
    let sampling = result.sampling_topology();
    let step = &workload.model.sequence.steps[0];
    let definitions = fields
        .fields()
        .iter()
        .map(|f| f.definition().clone())
        .collect::<Vec<_>>();
    let plan = resolve_sampled_observation_plan(
        step,
        sampling,
        mesh,
        &definitions,
        ObservationLimits::default(),
    )
    .unwrap();
    let source = ObservationFieldCatalog::sampled_recovery(
        recovery.clone(),
        manifest.clone(),
        sampling.clone(),
    )
    .unwrap();
    let selected = source
        .canonical_selected_fields(&plan, ObservationSelectionCoordinate::SampledRecovery)
        .unwrap();
    let selection =
        ObservationFieldSelection::sampled_recovery(&recovery, &manifest, sampling, selected)
            .unwrap();
    let evaluation = ObservationEvaluationRequest::new(selection, plan).unwrap();
    let values = RecoveredFieldCatalog::new_for_topology(
        sampling.identity().unwrap().discretization_digest(),
        sampling.entity_counts(),
        fields.into_fields(),
    )
    .unwrap();
    let results = evaluate_sampled_observations(&evaluation.plan, sampling, mesh, &values).unwrap();
    let artifact = ObservationArtifact::new(ObservationArtifactInput {
        evaluation_request: evaluation.identity(),
        producer: evaluation.producer,
        source_selection: evaluation.selection.identity(),
        source: evaluation.selection.source(),
        topology_digest: sampling.identity().unwrap().discretization_digest(),
        plan_identity: evaluation.plan.identity,
        results,
    })
    .unwrap();
    let publication = ObservationPublicationCatalog::new(
        result.producer(),
        evaluation.plan.clone(),
        source,
        step,
        vec![ObservationPublicationEntry::Sample(Box::new(
            ObservationSamplePublication::new(evaluation, &artifact).unwrap(),
        ))],
        ObservationHistorySourceContext::Magnetostatic { workload, mesh },
    )
    .unwrap();
    ObservationPublicationCatalog::from_canonical_bytes(&publication.canonical_bytes().unwrap())
        .unwrap()
}

fn request(
    receipt: &MagneticNumericalReceiptArtifact,
    workload: &MagnetostaticWorkload,
    observations: Option<&ObservationPublicationCatalog>,
) -> AccuracyEvaluationRequest {
    // Constructing every variant freshly proves rejection isn't merely a stale digest.
    let source_result = observations.map_or(receipt.source_result(), |catalog| {
        let ObservationFieldCatalogSource::SampledRecovery {
            recovery_request, ..
        } = catalog.source().source()
        else {
            panic!("sampled magnetic catalog");
        };
        recovery_request.source.solver_result().unwrap()
    });
    AccuracyEvaluationRequest::new(
        receipt.receipt().producer,
        AccuracySourceKind::Magnetostatic,
        source_result,
        observations,
        workload.model.sequence.steps[0].accuracy.clone(),
    )
    .unwrap()
}

#[test]
fn magnetic_accuracy_original_catalog_and_rehashed_missing_extra_catalogs_both_orders() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let mut workload = workload();
        workload.model.sequence.steps[0].observations = vec![observation()];
        let result = workload.execute(&mesh).unwrap();
        let receipt = MagneticNumericalReceiptArtifact::new(&workload, &result, &mesh).unwrap();
        let catalog = catalog(&workload, &mesh, &result);
        let inputs = context(&workload, &mesh, &result);
        let refs = inputs.iter().collect::<Vec<_>>();
        let original = request(&receipt, &workload, Some(&catalog));
        crate::validate_accuracy_source_context(
            &original,
            &AccuracySourceReceipt::Magnetostatic(receipt.clone()),
            Some(&catalog),
            &refs,
        )
        .unwrap();
        let omitted = request(&receipt, &workload, None);
        assert_ne!(omitted.identity(), original.identity());
        assert!(validate(&omitted, &receipt, None, &refs)
            .unwrap_err()
            .to_string()
            .contains("authored requirements"));
        let mut no_observations = workload.clone();
        no_observations.model.sequence.steps[0].observations.clear();
        let no_result = no_observations.execute(&mesh).unwrap();
        let no_receipt =
            MagneticNumericalReceiptArtifact::new(&no_observations, &no_result, &mesh).unwrap();
        let no_inputs = context(&no_observations, &mesh, &no_result);
        let no_refs = no_inputs.iter().collect::<Vec<_>>();
        let plain = request(&no_receipt, &no_observations, None);
        validate(&plain, &no_receipt, None, &no_refs).unwrap();
        let extra = request(&no_receipt, &no_observations, Some(&catalog));
        assert_ne!(extra.identity(), plain.identity());
        assert!(validate(&extra, &no_receipt, Some(&catalog), &no_refs)
            .unwrap_err()
            .to_string()
            .contains("authored requirements"));
        let mut changed = original.plan.clone();
        changed.accuracy_plan_id = AccuracyPlanId("weakened-new-identity".into());
        let changed = AccuracyEvaluationRequest::new(
            original.producer,
            original.source_kind,
            original.source_result,
            Some(&catalog),
            changed,
        )
        .unwrap();
        assert!(validate(&changed, &receipt, Some(&catalog), &refs)
            .unwrap_err()
            .to_string()
            .contains("authored requirements"));
    }
}

#[test]
fn magnetic_accuracy_rejects_foreign_rehashed_catalog_and_sampling_both_orders() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let mut workload = workload();
        workload.model.sequence.steps[0].observations = vec![observation()];
        let result = workload.execute(&mesh).unwrap();
        let receipt = MagneticNumericalReceiptArtifact::new(&workload, &result, &mesh).unwrap();
        let original = catalog(&workload, &mesh, &result);
        let inputs = context(&workload, &mesh, &result);
        let refs = inputs.iter().collect::<Vec<_>>();
        let mut foreign_workload = workload.clone();
        foreign_workload.model.materials[0]
            .electrical
            .as_mut()
            .unwrap()
            .relative_permeability = runmat_types::physical::RelativePermeability::new(2.0);
        let foreign_result = foreign_workload.execute(&mesh).unwrap();
        let foreign = catalog(&foreign_workload, &mesh, &foreign_result);
        assert_ne!(original.identity(), foreign.identity());
        let changed = request(&receipt, &workload, Some(&foreign));
        // All catalog-local hashes, authored declarations and sampled selections agree;
        // rejection must bind back to the ORIGINAL retained native result.
        assert!(validate(&changed, &receipt, Some(&foreign), &refs)
            .unwrap_err()
            .to_string()
            .contains("original native result"));
        let other = if order == ElementOrder::Tet4 {
            ElementOrder::Tet10
        } else {
            ElementOrder::Tet4
        };
        let foreign_mesh = canonical_tetrahedron_solver_mesh(other);
        let foreign_result = workload.execute(&foreign_mesh).unwrap();
        let foreign_sampling = catalog(&workload, &foreign_mesh, &foreign_result);
        assert_ne!(original.identity(), foreign_sampling.identity());
        let changed = request(&receipt, &workload, Some(&foreign_sampling));
        assert!(validate(&changed, &receipt, Some(&foreign_sampling), &refs).is_err());
        let recovery = result.native_field_request(&workload, &mesh).unwrap();
        let manifest = result
            .recover_native_fields(&workload, &mesh, &recovery)
            .unwrap()
            .manifest()
            .unwrap();
        let q4 = SampledVolumeFieldTopology::derive(
            &mesh,
            VolumeFieldSamplingRule::TetrahedronFourPoint,
        )
        .unwrap();
        assert!(ObservationFieldCatalog::sampled_recovery(recovery, manifest, q4).is_err());
        // Declarations are deliberately outside step producer identity: authenticate the
        // original workload/receipt even when a substituted step keeps its procedure ID.
        let mut changed_workload = workload.clone();
        let ObservationRequest::Field(spec) =
            &mut changed_workload.model.sequence.steps[0].observations[0]
        else {
            panic!("field");
        };
        spec.component = ObservationComponent::Y;
        let changed_inputs = context(&changed_workload, &mesh, &result);
        assert!(validate(
            &request(&receipt, &workload, Some(&original)),
            &receipt,
            Some(&original),
            &changed_inputs.iter().collect::<Vec<_>>()
        )
        .is_err());
    }
}
