use std::collections::BTreeMap;

use runmat_analysis_core::{
    AnalysisBoundary, AnalysisConstraint, CoordinateFrame, ExcitationCaseId,
    ExcitationDistribution, ExcitationId, ExcitationKinematics, HeatFlux, HeatTransferCoefficient,
    PhysicalQuantityKind, PhysicalSignal, PrescribedValue, ProjectionRequirement, RegionTarget,
    Signal, SignalApproximationEvidence, SignalExtrapolation, SignalId, SignalInterpolation,
    SignalKnot, SignalOrdinate, SignalPrecision, SignalProvenance, SignalRepresentation,
    SignalShape, SignalValue, TabulatedSignal, TabulatedSignalDomain, Temperature,
    ThermalConstraint, ThermalConstraintKind,
};
use runmat_execution::Digest;
use runmat_meshing_core::{ElementOrder, PersistentEntityId, PersistentEntityKind};

use super::*;
use crate::{
    excitation::{
        ComplexScalar, RealizedExcitation, RealizedExcitationKind, RealizedExcitationPolicy,
    },
    fixtures::{fixture_model, FixtureId},
    solve::{
        backend::{cpu_reference::CpuReferenceBackend, kind::LinearAlgebraBackendKind},
        linear::solve_operator_system,
        preconditioner::SpdPreconditionerKind,
    },
};

// Direct assembly fixtures prepare shared ownership just as the production preassembly does.
pub(super) fn assemble(
    model: &AnalysisModel,
    mesh: &runmat_meshing_core::SolverMeshArtifact,
    support: &crate::algebra::SparsePattern,
    excitations: &RealizedExcitations,
    sample: ExcitationSamplePoint<'_>,
) -> Result<ThermalOperatorAssembly, ThermalAssemblyError> {
    let properties = crate::properties::compile_thermal_property_map(model, Some(mesh))
        .map_err(|error| ThermalAssemblyError::InvalidMaterialCoverage(error.to_string()))?;
    super::assemble(model, mesh, &properties, support, excitations, sample)
}

fn heat_generation(value: f64) -> RealizedExcitations {
    RealizedExcitations::new(vec![RealizedExcitation {
        excitation_id: ExcitationId("uniform_heat".into()),
        case_ids: vec![ExcitationCaseId("uniform_heat_case".into())],
        signal_id: SignalId("uniform_heat_signal".into()),
        quantity: PhysicalQuantityKind::HeatGenerationDensity,
        sample: crate::excitation::ProjectionSample::Static,
        target: RealizedExcitationTarget::WholeModel,
        kind: RealizedExcitationKind::HeatGeneration(ComplexScalar::real(value)),
        policy: RealizedExcitationPolicy {
            frame: CoordinateFrame::Global,
            distribution: ExcitationDistribution::IntensityPerVolume,
            kinematics: ExcitationKinematics::FixedFrame,
            projection: ProjectionRequirement {
                preserve_resultant: true,
                preserve_moment: false,
                relative_tolerance: 1.0e-12,
            },
        },
    }])
}

fn thermal_reference_model() -> (
    AnalysisModel,
    runmat_meshing_core::SolverMeshArtifact,
    usize,
    String,
) {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.constraints.clear();
    model.boundaries.clear();
    model.excitations.clear();
    let material = &mut model.materials[0];
    let thermal = material
        .thermal
        .as_mut()
        .expect("thermal fixture declares thermal capability");
    thermal.conductivity_w_per_mk = 1.0;
    thermal.specific_heat_j_per_kgk = 1.0;
    material.mass_density_kg_per_m3 = Some(1.0);

    let mut mesh =
        runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let coordinates = mesh
        .topology
        .nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    let base_face = mesh
        .topology
        .boundary_faces
        .iter_mut()
        .find(|face| {
            face.node_ids
                .iter()
                .all(|node| coordinates[node][2].abs() <= 1.0e-12)
        })
        .expect("canonical tetrahedron has a z=0 base face");
    let base_region = PersistentEntityId {
        kind: PersistentEntityKind::Face,
        source_topology_id: "thermal_base".into(),
        assembly_path: vec!["root".into()],
    };
    base_face.provenance.push(base_region.clone());
    base_face.provenance.sort();
    let base_nodes = base_face.node_ids.clone();
    let free_node_id = mesh
        .topology
        .nodes
        .iter()
        .find(|node| !base_nodes.contains(&node.node_id))
        .expect("one tetrahedron node is opposite its base")
        .node_id;
    let free_node_index = mesh
        .topology
        .nodes
        .iter()
        .position(|node| node.node_id == free_node_id)
        .unwrap();
    mesh.seal_canonical_digest().unwrap();
    (model, mesh, free_node_index, base_region.source_topology_id)
}

#[test]
fn unused_material_declarations_do_not_change_thermal_operators_or_diagnostics() {
    let (model, mesh, _, _) = thermal_reference_model();
    let mut with_unused_material = model.clone();
    let mut unused = with_unused_material.materials[0].clone();
    unused.material_id = "unused_extreme_material".into();
    unused.name = "Unused extreme material".into();
    unused.mass_density_kg_per_m3 = Some(9.0e8);
    let thermal = unused.thermal.as_mut().unwrap();
    thermal.conductivity_w_per_mk = 7.0e11;
    thermal.specific_heat_j_per_kgk = 3.0e10;
    with_unused_material.materials.push(unused);

    let baseline = assemble(
        &model,
        &mesh,
        crate::planning::resource::ThermalOperatorResourceFacts::derive(&mesh)
            .unwrap()
            .support(),
        &RealizedExcitations::default(),
        ExcitationSamplePoint::Static,
    )
    .unwrap();
    let changed = assemble(
        &with_unused_material,
        &mesh,
        crate::planning::resource::ThermalOperatorResourceFacts::derive(&mesh)
            .unwrap()
            .support(),
        &RealizedExcitations::default(),
        ExcitationSamplePoint::Static,
    )
    .unwrap();

    assert_eq!(baseline.property_identity, changed.property_identity);
    assert_eq!(baseline.constitutive_stats, changed.constitutive_stats);
    assert_eq!(baseline.conductivity, changed.conductivity);
    assert_eq!(baseline.capacity, changed.capacity);
}

#[test]
fn prepared_properties_are_reused_across_load_samples_and_reject_topology_substitution() {
    let (model, mesh, _, _) = thermal_reference_model();
    let properties = crate::properties::compile_thermal_property_map(&model, Some(&mesh)).unwrap();
    let operators = crate::planning::resource::ThermalOperatorResourceFacts::derive(&mesh).unwrap();
    let first = super::assemble(
        &model,
        &mesh,
        &properties,
        operators.support(),
        &heat_generation(1.0),
        ExcitationSamplePoint::Static,
    )
    .unwrap();
    let second = super::assemble(
        &model,
        &mesh,
        &properties,
        operators.support(),
        &heat_generation(2.0),
        ExcitationSamplePoint::Static,
    )
    .unwrap();
    assert_eq!(first.property_identity, properties.identity);
    assert_eq!(second.property_identity, properties.identity);
    assert_eq!(first.conductivity, second.conductivity);
    assert_eq!(first.capacity, second.capacity);
    let fresh = assemble(
        &model,
        &mesh,
        operators.support(),
        &heat_generation(2.0),
        ExcitationSamplePoint::Static,
    )
    .unwrap();
    assert_eq!(second.conductivity, fresh.conductivity);
    assert_eq!(second.capacity, fresh.capacity);
    assert_eq!(second.source, fresh.source);
    for (one, two) in first.source.iter().zip(&second.source) {
        assert!((two - 2.0 * one).abs() < 1e-14);
    }
    let mut substituted = properties.clone();
    substituted.identity.mesh_digest = None;
    assert!(matches!(
        super::assemble(
            &model,
            &mesh,
            &substituted,
            operators.support(),
            &heat_generation(1.0),
            ExcitationSamplePoint::Static
        ),
        Err(ThermalAssemblyError::InvalidMaterialCoverage(_))
    ));
    for change_revision in [false, true] {
        let mut changed_model = model.clone();
        if change_revision {
            changed_model.geometry_revision += 1;
        } else {
            changed_model.geometry_id.push_str("_different");
        }
        assert!(matches!(
            super::assemble(
                &changed_model,
                &mesh,
                &properties,
                operators.support(),
                &heat_generation(1.0),
                ExcitationSamplePoint::Static
            ),
            Err(ThermalAssemblyError::InvalidMaterialCoverage(_))
        ));
    }
}

fn prescribe_base(model: &mut AnalysisModel, base_region: String, temperature_k: f64) {
    model
        .constraints
        .push(AnalysisConstraint::Thermal(ThermalConstraint {
            constraint_id: "base_temperature".into(),
            kind: ThermalConstraintKind::Prescribed {
                target: RegionTarget::boundary(base_region),
                value: PrescribedValue::Constant(Temperature::new(temperature_k)),
            },
        }));
}

#[test]
fn steady_nullspace_proof_requires_an_anchor_in_every_component() {
    let (mut model, mesh, _, base_region) = thermal_reference_model();
    prescribe_base(&mut model, base_region, 300.0);
    let assembly = assemble(
        &model,
        &mesh,
        crate::planning::resource::ThermalOperatorResourceFacts::derive(&mesh)
            .unwrap()
            .support(),
        &RealizedExcitations::default(),
        ExcitationSamplePoint::Static,
    )
    .unwrap();
    let anchored = assembly.steady_operator().unwrap();
    assert!(anchored.nullspace_evidence().is_some());

    let mut disconnected_projection = assembly.projection.clone();
    disconnected_projection
        .node_neighbors
        .iter_mut()
        .for_each(Vec::clear);
    let constrained = assembly
        .prescribed_temperature_k
        .iter()
        .map(Option::is_some)
        .collect::<Vec<_>>();
    assert_eq!(
        NullspaceEliminationEvidence::prove_thermal(
            &disconnected_projection,
            &assembly.conductivity,
            &constrained,
            &assembly.positive_robin_weight,
        ),
        Err(crate::algebra::NullspaceEvidenceError::NullModeViolation)
    );

    let unanchored = assemble(
        &thermal_reference_model().0,
        &mesh,
        crate::planning::resource::ThermalOperatorResourceFacts::derive(&mesh)
            .unwrap()
            .support(),
        &RealizedExcitations::default(),
        ExcitationSamplePoint::Static,
    )
    .unwrap()
    .steady_operator()
    .unwrap();
    assert!(unanchored.nullspace_evidence().is_none());
}

#[test]
fn thermal_nullspace_proof_rejects_metadata_only_semidefinite_claims() {
    let (mut model, mesh, _, base_region) = thermal_reference_model();
    prescribe_base(&mut model, base_region, 300.0);
    let assembly = assemble(
        &model,
        &mesh,
        crate::planning::resource::ThermalOperatorResourceFacts::derive(&mesh)
            .unwrap()
            .support(),
        &RealizedExcitations::default(),
        ExcitationSamplePoint::Static,
    )
    .unwrap();
    let constrained = assembly
        .prescribed_temperature_k
        .iter()
        .map(Option::is_some)
        .collect::<Vec<_>>();
    let false_semidefinite = RealSparseMatrix::diagonal(
        vec![1.0; assembly.projection.nodes.len()],
        OperatorRole::Conductivity,
        MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
    )
    .unwrap();

    assert_eq!(
        NullspaceEliminationEvidence::prove_thermal(
            &assembly.projection,
            &false_semidefinite,
            &constrained,
            &assembly.positive_robin_weight,
        ),
        Err(crate::algebra::NullspaceEvidenceError::NullModeViolation),
    );
}

#[test]
fn positive_convection_anchors_a_steady_thermal_component() {
    let (mut model, mesh, _, _) = thermal_reference_model();
    let projection = AnalysisProjection::derive(&mesh).unwrap();
    let selector = projection.boundary_faces[0].provenance[0]
        .analysis_selection_id()
        .unwrap();
    model.boundaries.push(AnalysisBoundary {
        boundary_id: "positive_robin_anchor".into(),
        target: RegionTarget::boundary(selector),
        operator: BoundaryOperator::ThermalConvection {
            ambient_temperature: PrescribedValue::Constant(Temperature::new(295.0)),
            coefficient: HeatTransferCoefficient::new(4.0),
        },
    });
    let system = assemble(
        &model,
        &mesh,
        crate::planning::resource::ThermalOperatorResourceFacts::derive(&mesh)
            .unwrap()
            .support(),
        &RealizedExcitations::default(),
        ExcitationSamplePoint::Static,
    )
    .unwrap()
    .steady_operator()
    .unwrap();
    let prepared = system.prepare_linear().unwrap();
    assert!(prepared.admissibility().is_some());
    assert!(crate::algebra::validate_prepared_solver_compatibility(
        &prepared,
        crate::algebra::RealSolver::ConjugateGradient,
        crate::algebra::RealPreconditioner::Jacobi,
    )
    .is_ok());
}

#[test]
fn deserialization_discards_process_local_nullspace_proof() {
    let (mut model, mesh, _, base_region) = thermal_reference_model();
    prescribe_base(&mut model, base_region, 300.0);
    let system = assemble(
        &model,
        &mesh,
        crate::planning::resource::ThermalOperatorResourceFacts::derive(&mesh)
            .unwrap()
            .support(),
        &RealizedExcitations::default(),
        ExcitationSamplePoint::Static,
    )
    .unwrap()
    .steady_operator()
    .unwrap();
    let encoded = serde_json::to_value(system).unwrap();
    assert!(encoded.get("nullspace_evidence").is_none());
    let decoded = serde_json::from_value::<AssembledRealSystem>(encoded).unwrap();
    assert!(decoded.nullspace_evidence().is_none());
}

#[test]
fn deserialization_rejects_caller_asserted_nullspace_proof() {
    let (mut model, mesh, _, base_region) = thermal_reference_model();
    prescribe_base(&mut model, base_region, 300.0);
    let system = assemble(
        &model,
        &mesh,
        crate::planning::resource::ThermalOperatorResourceFacts::derive(&mesh)
            .unwrap()
            .support(),
        &RealizedExcitations::default(),
        ExcitationSamplePoint::Static,
    )
    .unwrap()
    .steady_operator()
    .unwrap();
    let mut encoded = serde_json::to_value(system).unwrap();
    encoded["nullspace_evidence"] = serde_json::json!({
        "proof_identity": Digest::sha256("caller-asserted").to_string()
    });
    assert!(serde_json::from_value::<AssembledRealSystem>(encoded).is_err());
}

mod numerical;
mod signals;
mod target_resolution;
