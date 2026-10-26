use super::*;
pub(super) fn model(mesh: &SolverMeshArtifact) -> AnalysisModel {
    let mut model = runmat_analysis_fea::fixtures::fixture_model(
        runmat_analysis_fea::fixtures::FixtureId::CantileverLinearStatic,
    );
    model.signals.clear();
    model.excitations.clear();
    model.excitation_cases.clear();
    model.geometry_revision = mesh.geometry.geometry_revision as u32;
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

pub(super) fn plan() -> AccuracyPlan {
    let mut plan = AccuracyPlan::baseline("adaptive");
    plan.exceeded_limit = AccuracyLimitAction::Refine;
    plan.resources.maximum_refinement_cycles = 4;
    plan.criteria = [
        (
            AccuracyTarget::NumericalResult,
            ErrorContributionKind::Recovery,
            RequiredAccuracyEvidence::APosterioriEstimator,
            0.2,
        ),
        (
            AccuracyTarget::NumericalResult,
            ErrorContributionKind::Recovery,
            RequiredAccuracyEvidence::ConvergenceStudy,
            0.95,
        ),
        (
            AccuracyTarget::Observation(ObservationId("tip_displacement".into())),
            ErrorContributionKind::SpatialDiscretization,
            RequiredAccuracyEvidence::ConvergenceStudy,
            1.0e-3,
        ),
        (
            AccuracyTarget::NumericalResult,
            ErrorContributionKind::FieldTransfer,
            RequiredAccuracyEvidence::TransferBound,
            0.05,
        ),
        (
            AccuracyTarget::GlobalEquilibrium,
            ErrorContributionKind::Algebraic,
            RequiredAccuracyEvidence::ResidualBound,
            1.0e-8,
        ),
    ]
    .into_iter()
    .enumerate()
    .map(
        |(i, (target, contribution, evidence, tolerance))| AccuracyCriterion {
            criterion_id: format!("criterion_{i}"),
            target,
            contribution,
            tolerance: AccuracyTolerance::Relative(tolerance),
            evidence,
            mandatory: true,
        },
    )
    .collect();
    plan
}

pub(super) fn lineage(
    source: &SolverMeshArtifact,
    target: &SolverMeshArtifact,
) -> (SolverMeshTransferMap, SolverMeshAdaptationLineage) {
    let cell =
        |mesh: &SolverMeshArtifact, element: &SolverVolumeElement| SolverMeshAdaptationCell {
            stable_identity: element.stable_identity,
            node_identities: std::array::from_fn(|i| {
                mesh.topology.nodes[element.node_ids[i] as usize - 1].stable_identity
            }),
            region_id: element.region_id.clone(),
        };
    let parent = cell(source, &source.topology.volume_elements[0]);
    let mut children: Vec<_> = target
        .topology
        .volume_elements
        .iter()
        .map(|e| cell(target, e))
        .collect();
    children.sort_by_key(|cell| cell.stable_identity);
    let inserted = &target.topology.nodes[4];
    let mut map = SolverMeshTransferMap {
        schema_version: SOLVER_MESH_TRANSFER_SCHEMA_VERSION,
        source_artifact_digest: source.canonical_digest,
        target_artifact_digest: target.canonical_digest,
        geometry: source.geometry.clone(),
        node_transfers: vec![SolverEntityTransfer {
            target_stable_identity: inserted.stable_identity,
            method: SolverTransferMethod::BarycentricInterpolation,
            sources: source
                .topology
                .nodes
                .iter()
                .map(|node| SolverTransferSource {
                    stable_identity: node.stable_identity,
                    weight: 0.25,
                })
                .collect(),
        }],
        volume_element_transfers: children
            .iter()
            .map(|child| SolverEntityTransfer {
                target_stable_identity: child.stable_identity,
                method: SolverTransferMethod::CentroidProjection,
                sources: vec![SolverTransferSource {
                    stable_identity: parent.stable_identity,
                    weight: 1.0,
                }],
            })
            .collect(),
        boundary_face_transfers: Vec::new(),
        boundary_edge_transfers: Vec::new(),
    };
    map.volume_element_transfers
        .sort_by_key(|transfer| transfer.target_stable_identity);
    for transfer in &mut map.node_transfers {
        transfer
            .sources
            .sort_by_key(|source| source.stable_identity);
    }
    let lineage = SolverMeshAdaptationLineage {
        schema_version: SOLVER_MESH_ADAPTATION_LINEAGE_SCHEMA_VERSION,
        source_artifact_digest: source.canonical_digest,
        target_artifact_digest: target.canonical_digest,
        transfer_map_digest: map.canonical_digest().unwrap(),
        geometry: source.geometry.clone(),
        kind: SolverMeshAdaptationKind::HRefinement,
        marks: vec![SolverMeshAdaptationMark {
            element_stable_identity: parent.stable_identity,
            indicator_value: 1.0,
        }],
        requested_removal_node_identities: Vec::new(),
        mutations: vec![SolverMeshAdaptationMutation {
            source_mark_identity: Some(parent.stable_identity),
            node_identity: inserted.stable_identity,
            node_coordinates_m: inserted.coordinates_m.canonical_value(),
            removed_cells: vec![parent],
            created_cells: children,
        }],
    };
    lineage.validate_against(source, target, &map).unwrap();
    (map, lineage)
}

pub(super) fn source_limits() -> StructuralAdaptationSourceLimits {
    StructuralAdaptationSourceLimits {
        canonical: runmat_canonical_codec::CanonicalLimits::new(
            64 * 1024 * 1024,
            1_000_000,
            4096,
            64,
        ),
        maximum_materialization_bytes: 256 * 1024 * 1024,
        observations: ObservationLimits::default(),
    }
}

pub(super) fn assert_affine_original(
    mesh: &SolverMeshArtifact,
    run: &runmat_analysis_fea::contracts::FeaRunResult,
) {
    let displacements = run
        .field(runmat_analysis_fea::contracts::FEA_FIELD_STRUCTURAL_DISPLACEMENT)
        .unwrap()
        .as_host_f64()
        .unwrap();
    assert_eq!(displacements.len(), mesh.topology.nodes.len() * 3);
    for (values, node) in displacements.chunks_exact(3).zip(&mesh.topology.nodes) {
        for (actual, expected) in
            values
                .iter()
                .zip([0.02 * node.coordinates_m.canonical_value()[0], 0.0, 0.0])
        {
            assert!((actual - expected).abs() < 1e-12);
        }
    }
    for values in run
        .field(runmat_analysis_fea::contracts::FEA_FIELD_STRUCTURAL_INTEGRATION_POINT_STRESS)
        .unwrap()
        .as_host_f64()
        .unwrap()
        .chunks_exact(6)
    {
        for (actual, expected) in values.iter().zip([2.4, 0.8, 0.8, 0.0, 0.0, 0.0]) {
            assert!((actual - expected).abs() < 1e-12);
        }
    }
}
