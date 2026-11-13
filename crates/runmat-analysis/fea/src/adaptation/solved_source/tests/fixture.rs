use super::*;
pub(super) fn model(mesh: &SolverMeshArtifact) -> AnalysisModel {
    let mut model =
        crate::fixtures::fixture_model(crate::fixtures::FixtureId::CantileverLinearStatic);
    crate::fixtures::replace_fixture_excitations(&mut model, Vec::new());
    let MaterialElasticityModel::IsotropicLinear(elasticity) =
        model.materials[0].elasticity.as_mut().unwrap()
    else {
        panic!("fixture elasticity")
    };
    elasticity.youngs_modulus_pa = runmat_types::physical::YoungsModulus::new(100.0);
    elasticity.poisson_ratio = runmat_types::physical::PoissonRatio::new(0.25);
    model.constraints = mesh
        .topology
        .nodes
        .iter()
        .map(|node| {
            AnalysisConstraint::Structural(StructuralConstraint {
                constraint_id: format!("affine_boundary_{}", node.node_id),
                frame: CoordinateFrame::Global,
                kind: StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
                    target: RegionTarget::point(format!("node:{}", node.node_id)),
                    translations: [
                        (
                            StructuralTranslationDof::Ux,
                            0.02 * node.coordinates_m.canonical_value()[0],
                        ),
                        (StructuralTranslationDof::Uy, 0.0),
                        (StructuralTranslationDof::Uz, 0.0),
                    ]
                    .into_iter()
                    .map(|(dof, value)| TranslationDofValue {
                        dof,
                        value: PrescribedValue::Constant(Displacement::new(value)),
                    })
                    .collect(),
                    rotations: Vec::new(),
                }),
            })
        })
        .collect();
    model.sequence.steps[0].observations = vec![ObservationRequest::Reduction(ObservationSpec {
        observation_id: ObservationId("tip_displacement".into()),
        quantity: ObservationQuantity::Displacement,
        value_shape: ObservationValueShape::Vector3,
        target: ObservationTarget::WholeModel,
        source_location: FieldLocation::Node,
        output_location: FieldLocation::Node,
        recovery: RecoveryPolicy::Native,
        component: ObservationComponent::X,
        frame: CoordinateFrame::Global,
        sign_convention: SignConvention::Native,
        complex_representation: ComplexRepresentation::Real,
        sampling: SamplingPolicy::Final,
        reduction: Some(ReductionPolicy {
            spatial: SpatialReduction::Maximum,
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
            operation_count: 100_000,
            peak_resident_bytes: 1_000_000,
            stored_bytes: 1_000_000,
            transfer_bytes: 1_000_000,
            collective_count: 0,
        }),
    })];
    model
}
pub(super) fn solve(model: &AnalysisModel, mesh: &SolverMeshArtifact) -> FeaRunResult {
    crate::run_linear_static_with_options(
        model,
        ComputeBackend::Cpu,
        LinearStaticSolveOptions {
            solver_mesh: Some(mesh.clone()),
            require_solver_mesh_for_solid: true,
            ..LinearStaticSolveOptions::default()
        },
    )
    .unwrap()
}
pub(super) fn source<'a>(
    model: &'a AnalysisModel,
    mesh: &'a SolverMeshArtifact,
    run: &'a FeaRunResult,
    id: &'a ObservationId,
) -> StructuralAdaptationSolvedSourceInput<'a> {
    StructuralAdaptationSolvedSourceInput {
        model,
        mesh,
        run,
        observation_id: id,
        limits: StructuralAdaptationSourceLimits {
            canonical: CanonicalLimits::new(64 * 1024 * 1024, 1_000_000, 4096, 64),
            maximum_materialization_bytes: 256 * 1024 * 1024,
            observations: ObservationLimits::default(),
        },
    }
}
