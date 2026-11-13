use std::collections::BTreeMap;

use runmat_analysis_core::{
    AuthoredBoundary, AuthoredBoundaryKind, AuthoredConstraint, AuthoredConstraintKind,
    AuthoredConstraintTarget, AuthoredMaterialAssignment, AuthoredMaterialRef,
    AuthoredMaterialSource, AuthoredMeshRef, AuthoredModel, AuthoredPropertyTarget,
    AuthoredRegionSource, AuthoredTargetCardinality, IndependentDomain, MaterialAcousticModel,
    MaterialModel, MaterialProvenance, PrescribedValue, Pressure, ProcedureStep,
    RealizedGeometryInstance, RealizedStudy, SpecificAcousticImpedance,
};
use runmat_geometry_core::{
    GeometryAuthoringCapabilities, GeometryDefinitionRef, GeometryDefinitionSource, GeometryDigest,
    GeometryRegionRef, GeometryRegionRole, ImportedGeometryDefinition, ResolvedGeometryRegion,
    ResolvedGeometryRegions,
};
use runmat_meshing_core::{
    fixtures::canonical_tetrahedron_solver_mesh, sort_solver_node_exact_parameters, ElementOrder,
    SolverMeshArtifact, SolverNodeExactParameter, StableDigest,
};

use super::*;
use crate::execution::{
    CanonicalSemanticState, FieldTopologyIdentity, SequenceIdentity, StepIdentity,
};
use runmat_execution::Digest;

const MODULE: &str = "tests/acoustic-tetrahedron.fea";
const GEOMETRY: &str = "acoustic_cell";

#[test]
fn authored_tetrahedron_compiles_plans_and_executes_for_supported_orders() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let fixture = authored_tetrahedron(order);
        let product = build_acoustic_study_product(AcousticStudyProductInput {
            study: &fixture.study,
            geometry: &fixture.geometry,
            regions: &fixture.regions,
            materials: &fixture.materials,
            mesh: &fixture.mesh,
            planning: AcousticStudyPlanningPolicy::standard(),
        })
        .unwrap();

        assert_eq!(
            plan_acoustic_study_product(
                product.compiled.clone(),
                &fixture.mesh,
                AcousticStudyPlanningPolicy::standard(),
            )
            .unwrap(),
            product,
        );

        product.compiled.validate_against(&fixture.mesh).unwrap();
        assert_eq!(product.plan.frequency_plan().frequencies_hz(), &[0.2]);
        for (ordinal, frequency) in product.plan.frequencies().iter().enumerate() {
            assert_eq!(frequency.entry().ordinal() as usize, ordinal);
            let result = frequency.workload().execute(&fixture.mesh).unwrap();
            assert_eq!(
                result.pressure_pa().len(),
                fixture.mesh.topology.nodes.len()
            );
            assert_eq!(
                result.velocity_m_per_s().len(),
                fixture.mesh.topology.volume_elements.len()
            );
            for node_id in &fixture.mesh.topology.boundary_faces[0].node_ids {
                let index = result
                    .ordered_node_ids()
                    .iter()
                    .position(|candidate| candidate == node_id)
                    .unwrap();
                let pressure = result.pressure_pa()[index];
                assert!((pressure.real - 1.0).abs() < 1.0e-11);
                assert!(pressure.imaginary.abs() < 1.0e-11);
            }
            assert!(result.velocity_m_per_s().iter().all(|value| value
                .real
                .iter()
                .chain(&value.imaginary)
                .all(|component| component.is_finite())));
            let state = CanonicalSemanticState::from_acoustic_result(
                frequency.workload(),
                &fixture.mesh,
                &result,
            )
            .unwrap();
            let expected_producer = StepIdentity::from_solver_contract(
                SequenceIdentity::from_solver_contract(&product.compiled.model.sequence).unwrap(),
                &product.compiled.model.sequence.steps[0],
            )
            .unwrap();
            assert_eq!(state.producer, expected_producer);
            assert_eq!(
                state.state_id,
                product.compiled.model.sequence.steps[0].state_outputs[0].state_id
            );
            assert_eq!(
                state.source_result,
                result.source_result_identity().unwrap()
            );
            assert_eq!(
                state.topology_digest,
                Digest::from_bytes(*fixture.mesh.canonical_digest.bytes())
            );
            assert!(matches!(
                state.payload.acoustic().unwrap().recovery_request.topology,
                FieldTopologyIdentity::VolumeMesh { mesh_digest }
                    if mesh_digest == fixture.mesh.canonical_digest
            ));
            assert_eq!(
                CanonicalSemanticState::from_canonical_bytes(&state.canonical_bytes().unwrap())
                    .unwrap(),
                state
            );
        }
    }
}

#[test]
fn post_sealing_product_retains_material_and_mesh_admission() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mut fixture = authored_tetrahedron(order);
        fixture
            .materials
            .values_mut()
            .next()
            .unwrap()
            .acoustic
            .as_mut()
            .unwrap()
            .damping_ratio = runmat_types::physical::Quantity::new(1.0);
        let compiled = compile_analysis_study(AnalysisStudyCompilationInput {
            study: &fixture.study,
            geometry: &fixture.geometry,
            regions: &fixture.regions,
            materials: &fixture.materials,
            mesh: &fixture.mesh,
        })
        .unwrap();
        assert!(matches!(
            plan_acoustic_study_product(
                compiled.clone(),
                &fixture.mesh,
                AcousticStudyPlanningPolicy::standard()
            ),
            Err(AcousticStudyProductError::UnsupportedMaterial),
        ));
        let foreign = authored_tetrahedron(if order == ElementOrder::Tet4 {
            ElementOrder::Tet10
        } else {
            ElementOrder::Tet4
        });
        assert!(matches!(
            plan_acoustic_study_product(
                compiled,
                &foreign.mesh,
                AcousticStudyPlanningPolicy::standard()
            ),
            Err(AcousticStudyProductError::Compilation(_)),
        ));
    }
}

struct AuthoredAcousticFixture {
    study: RealizedStudy,
    geometry: runmat_geometry_core::GeometryDocument,
    regions: ResolvedGeometryRegions,
    materials: BTreeMap<AuthoredMaterialRef, MaterialModel>,
    mesh: SolverMeshArtifact,
}

fn authored_tetrahedron(order: ElementOrder) -> AuthoredAcousticFixture {
    let (geometry, topology, _) = runmat_geometry_fixtures::exact_tetrahedron();
    let solid = topology.solids[0].id.clone();
    let face_order = [0, 1, 3, 2];
    let mut mesh = canonical_tetrahedron_solver_mesh(order);
    mesh.geometry.source_digest = StableDigest::from_bytes(*geometry.source.content_digest.bytes());
    mesh.geometry.geometry_revision = geometry.revision.revision;
    mesh.geometry.persistent_mapping_version = geometry.revision.persistent_mapping_version;
    for node in &mut mesh.topology.nodes {
        node.provenance = if node.exact_parameters.is_empty() {
            vec![solid.clone()]
        } else {
            for (parameter, topology_index) in node.exact_parameters.iter_mut().zip(face_order) {
                let SolverNodeExactParameter::Surface { source_face_id, .. } = parameter else {
                    unreachable!("the canonical quadratic tetrahedron uses surface parameters")
                };
                *source_face_id = topology.faces[topology_index].id.clone();
            }
            sort_solver_node_exact_parameters(&mut node.exact_parameters);
            let mut provenance = face_order
                .iter()
                .map(|index| topology.faces[*index].id.clone())
                .chain(std::iter::once(solid.clone()))
                .collect::<Vec<_>>();
            provenance.sort();
            provenance
        };
    }
    mesh.topology.volume_elements[0].provenance = vec![solid.clone()];
    mesh.topology.volume_elements[0].region_id = topology.regions[0].id.clone();
    mesh.topology.regions[0].region_id = topology.regions[0].id.clone();
    for (face, topology_index) in mesh.topology.boundary_faces.iter_mut().zip(face_order) {
        face.provenance = vec![topology.faces[topology_index].id.clone()];
    }
    mesh.seal_canonical_digest().unwrap();

    let geometry_ref = |region_id: &str, role| {
        AuthoredRegionSource::Geometry(GeometryRegionRef {
            module_path: MODULE.into(),
            geometry_id: GEOMETRY.into(),
            region_id: region_id.into(),
            role,
        })
    };
    let material_ref = AuthoredMaterialRef {
        module_path: MODULE.into(),
        material_id: "fluid".into(),
    };
    let mut step = ProcedureStep::acoustic_harmonic("harmonic");
    step.independent_domain = IndependentDomain::FrequencyHertz(vec![0.2]);
    let model = AuthoredModel {
        module_path: MODULE.into(),
        model_id: "acoustic_cell".into(),
        electromagnetic: None,
        thermal: None,
        thermo_mechanical: None,
        inputs: Vec::new(),
        includes: Vec::new(),
        sections: Vec::new(),
        layups: Vec::new(),
        orientations: Vec::new(),
        material_assignments: vec![AuthoredMaterialAssignment {
            material: AuthoredMaterialSource::Declaration(material_ref.clone()),
            target: AuthoredPropertyTarget::Solid(geometry_ref("fluid", GeometryRegionRole::Solid)),
        }],
        property_assignments: Vec::new(),
        constraints: vec![AuthoredConstraint {
            constraint_id: "unit_pressure".into(),
            kind: AuthoredConstraintKind::AcousticPressure {
                target: AuthoredConstraintTarget {
                    region: geometry_ref("inlet", GeometryRegionRole::Face),
                    cardinality: AuthoredTargetCardinality::One,
                },
                value: PrescribedValue::Constant(Pressure::new(1.0)),
            },
        }],
        boundaries: vec![AuthoredBoundary {
            boundary_id: "absorbing_faces".into(),
            target: geometry_ref("outlet", GeometryRegionRole::Face),
            kind: AuthoredBoundaryKind::AcousticImpedance {
                impedance: SpecificAcousticImpedance::new(6.0),
            },
        }],
        connectors: Vec::new(),
        interfaces: Vec::new(),
        signals: Vec::new(),
        parameterized_signals: Vec::new(),
        excitations: Vec::new(),
        excitation_cases: Vec::new(),
        response_spectra: Vec::new(),
        random_excitations: Vec::new(),
        initial_conditions: Vec::new(),
    };
    let source = GeometryDefinitionSource::Imported(Box::new(ImportedGeometryDefinition {
        preview_extent_m: None,
        unit_evidence: geometry.source.unit_evidence.clone(),
        logical_path: "tests/acoustic-tetrahedron.step".into(),
        format: geometry.source.format,
        authored_units: geometry.source.unit_evidence.uniform_unit(),
        content_digest: geometry.source.content_digest,
        topology_digest: GeometryDigest::from_bytes([91; 32]),
        capabilities: GeometryAuthoringCapabilities {
            topology: true,
            persistent_ids: true,
            ..Default::default()
        },
        exchange: None,
        shared_interfaces: false,
    }));
    let study = RealizedStudy {
        module_path: MODULE.into(),
        study_id: "acoustic_cell".into(),
        geometry: RealizedGeometryInstance {
            definition: GeometryDefinitionRef {
                module_path: MODULE.into(),
                geometry_id: GEOMETRY.into(),
            },
            source,
            bindings: BTreeMap::new(),
        },
        mesh: AuthoredMeshRef {
            module_path: MODULE.into(),
            mesh_id: GEOMETRY.into(),
        },
        model,
        included_models: Vec::new(),
        study_inputs: Vec::new(),
        assessments: Vec::new(),
        outputs: Vec::new(),
        sequence: runmat_analysis_core::AnalysisSequence::from_steps("harmonic", vec![step]),
    };
    let regions = BTreeMap::from([
        (
            "fluid".into(),
            ResolvedGeometryRegion {
                role: GeometryRegionRole::Solid,
                entity_ids: vec![solid],
            },
        ),
        (
            "inlet".into(),
            ResolvedGeometryRegion {
                role: GeometryRegionRole::Face,
                entity_ids: vec![topology.faces[0].id.clone()],
            },
        ),
        (
            "outlet".into(),
            ResolvedGeometryRegion {
                role: GeometryRegionRole::Face,
                entity_ids: face_order[1..]
                    .iter()
                    .map(|index| topology.faces[*index].id.clone())
                    .collect(),
            },
        ),
    ]);
    let materials = BTreeMap::from([(
        material_ref,
        MaterialModel {
            fluid: None,
            material_id: "fluid".into(),
            name: "Test fluid".into(),
            provenance: MaterialProvenance::Authored,
            mass_density_kg_per_m3: (Some(3.0)).map(runmat_types::physical::MassDensity::new),
            elasticity: None,
            damping: None,
            thermal: None,
            acoustic: Some(MaterialAcousticModel {
                speed_of_sound_m_per_s: runmat_types::physical::SpeedOfSound::new(2.0),
                damping_ratio: runmat_types::physical::DampingRatio::new(0.0),
            }),
            electrical: None,
            plastic: None,
        },
    )]);
    AuthoredAcousticFixture {
        study,
        geometry,
        regions,
        materials,
        mesh,
    }
}
