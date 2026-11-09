use std::collections::BTreeMap;

use super::support::{source_module, TestHost};
use runmat_analysis_core::{AuthoredInputValue, AuthoredStudyInputBinding};
use runmat_analysis_fea::excitation::SpatialProjectionBalance;
use runmat_analysis_fea::{
    run_linear_static_with_options, ComputeBackend, LinearStaticSolveOptions,
    FEA_FIELD_STRUCTURAL_DISPLACEMENT, FEA_FIELD_STRUCTURAL_REACTION_FORCE,
    FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY,
};
use runmat_analysis_language::{
    CheckRequest, FeaCompiler, GeometryInspection, GeometryInspectionCapability, LogicalSourcePath,
    ModuleGraphRequest, SourceDigest,
};
use runmat_geometry_core::{GeometryRegionRole, ResolvedGeometryRegion};
use runmat_meshing_core::{fixtures, CanonicalMeshingContract, ElementOrder, StableDigest};
use runmat_types::PhysicalDimension;

const SOURCE: &str = r#"
geometry part {
    file = "./part.step"
    units = m
    region body : solid {
        select persistent_id "BODY_ID"
        expect exactly_one
    }
    region support : face {
        select persistent_id "SUPPORT_ID"
        expect exactly_one
    }
    region loaded : face {
        select persistent_id "LOAD_ID"
        expect exactly_one
    }
}
mesh part on geometry.part {
    tetrahedra {
        order = linear
        target_size = 0.5 m
        maximum_deviation = 0.01 m
    }
}
material steel {
    mass_density = 7850 kg/m^3
    elastic {
        youngs_modulus = 200 GPa
        poisson_ratio = 0.3
    }
}
model loaded {
    input service_load : force
    assign material.steel to geometry.part.region.body
    constraint base {
        structural.fixed { target = geometry.part.region.support }
    }
    load downward {
        force on geometry.part.region.loaded = [0, -input.service_load, 0]
    }
    load_case service { include load.downward }
}
study static {
    input commanded_load : force {
        default = 12 N
        minimum = 0 N
        maximum = 24 N
    }
    geometry = geometry.part
    mesh = mesh.part
    model = model.loaded(service_load = input.commanded_load)
    analysis solve { structural static {} }
}
"#;

fn close(actual: f64, expected: f64, scale: f64) {
    assert!(
        (actual - expected).abs() <= 1.0e-8 * scale.max(1.0e-20),
        "actual={actual} expected={expected} scale={scale}"
    );
}

#[test]
fn two_authored_force_inputs_compile_to_distinct_physical_static_solutions() {
    let (document, topology, _) = runmat_geometry_fixtures::exact_tetrahedron();
    let host = TestHost::default().with_geometry_inspection(
        "project/part.step",
        GeometryInspection {
            preview_extent_m: None,
            unit_evidence: Default::default(),
            content_digest: SourceDigest(*document.source.content_digest.bytes()),
            topology_digest: SourceDigest([19; 32]),
            capabilities: vec![
                GeometryInspectionCapability::Topology,
                GeometryInspectionCapability::PersistentIds,
            ],
            exchange: None,
            resolved_units: None,
            authored_names: Default::default(),
        },
    );
    let source = SOURCE
        .replace(
            "BODY_ID",
            &topology.solids[0].id.analysis_selection_id().unwrap(),
        )
        .replace(
            "SUPPORT_ID",
            &topology.faces[0].id.analysis_selection_id().unwrap(),
        )
        .replace(
            "LOAD_ID",
            &topology.faces[1].id.analysis_selection_id().unwrap(),
        );
    let program = FeaCompiler::new(&host).check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", &source, 1),
        },
    });
    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    let resolved = program.resolve_study("project/main.fea", "static").unwrap();
    let regions = BTreeMap::from([
        (
            "body".into(),
            ResolvedGeometryRegion {
                role: GeometryRegionRole::Solid,
                entity_ids: vec![topology.solids[0].id.clone()],
            },
        ),
        (
            "support".into(),
            ResolvedGeometryRegion {
                role: GeometryRegionRole::Face,
                entity_ids: vec![topology.faces[0].id.clone()],
            },
        ),
        (
            "loaded".into(),
            ResolvedGeometryRegion {
                role: GeometryRegionRole::Face,
                entity_ids: vec![topology.faces[1].id.clone()],
            },
        ),
    ]);
    let mut mesh = fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    mesh.geometry.source_digest = StableDigest::from_bytes(*document.source.content_digest.bytes());
    mesh.geometry.geometry_revision = document.revision.revision;
    mesh.geometry.persistent_mapping_version = document.revision.persistent_mapping_version;
    for node in &mut mesh.topology.nodes {
        node.provenance = vec![topology.solids[0].id.clone()];
    }
    mesh.topology.volume_elements[0].region_id = topology.regions[0].id.clone();
    mesh.topology.volume_elements[0].provenance = vec![topology.solids[0].id.clone()];
    mesh.topology.regions[0].region_id = topology.regions[0].id.clone();
    for (face, source) in mesh.topology.boundary_faces.iter_mut().zip(&topology.faces) {
        face.provenance = vec![source.id.clone()];
    }
    for (edge, source) in mesh.topology.boundary_edges.iter_mut().zip(&topology.edges) {
        edge.provenance = vec![source.id.clone()];
    }
    mesh.seal_canonical_digest().unwrap();
    mesh.validate_canonical().unwrap();

    let mut definitions = Vec::new();
    let mut workload_definitions = Vec::new();
    let mut solutions = Vec::new();
    for newtons in [12.0, 24.0] {
        let realized = resolved
            .study
            .realize_with_models(
                &resolved.models,
                &resolved.geometry,
                &[AuthoredStudyInputBinding {
                    input_id: "commanded_load".into(),
                    value: AuthoredInputValue::Quantity(
                        runmat_types::physical::DimensionedScalar::try_from_dimension(
                            PhysicalDimension::FORCE,
                            None,
                            newtons,
                        )
                        .unwrap(),
                    ),
                }],
            )
            .unwrap();
        let product = runmat_analysis_fea::study::build_structural_study_product(
            runmat_analysis_fea::study::StructuralStudyProductInput {
                study: &realized,
                geometry: &document,
                regions: &regions,
                materials: &resolved.material_catalog,
                mesh: &mesh,
                planning: runmat_analysis_fea::study::StructuralStudyPlanningPolicy {
                    backend: ComputeBackend::Cpu,
                    numerics:
                        runmat_analysis_fea::study::StructuralNumericalPlanningPolicy::production(),
                },
            },
        )
        .unwrap();
        let compiled = product.compiled;
        compiled.validate_against(&mesh).unwrap();
        definitions.push(compiled.definition_digest().unwrap());
        workload_definitions.push(product.workloads.canonical_digest().unwrap());
        let result = run_linear_static_with_options(
            &compiled.model,
            ComputeBackend::Cpu,
            LinearStaticSolveOptions {
                solver_mesh: Some(mesh.clone()),
                ..LinearStaticSolveOptions::default()
            },
        )
        .unwrap();
        assert!(result.structural_static_balance.as_ref().unwrap().accepted);
        let evidence = &result.spatial_projection_evidence[0];
        evidence.validate().unwrap();
        assert!(evidence.accepted);
        let SpatialProjectionBalance::VectorResultant {
            projected_resultant,
            projected_moment,
            ..
        } = evidence.balance
        else {
            panic!("authored face force must retain a vector-resultant projection");
        };
        let expected_force = [0.0, -newtons, 0.0];
        let expected_moment = [newtons / 3.0, 0.0, -newtons / 3.0];
        let projected_about_origin = [
            projected_moment.real[0] + evidence.reference_point_m[1] * projected_resultant.real[2]
                - evidence.reference_point_m[2] * projected_resultant.real[1],
            projected_moment.real[1] + evidence.reference_point_m[2] * projected_resultant.real[0]
                - evidence.reference_point_m[0] * projected_resultant.real[2],
            projected_moment.real[2] + evidence.reference_point_m[0] * projected_resultant.real[1]
                - evidence.reference_point_m[1] * projected_resultant.real[0],
        ];
        for axis in 0..3 {
            close(
                projected_resultant.real[axis],
                expected_force[axis],
                newtons,
            );
            close(projected_about_origin[axis], expected_moment[axis], newtons);
        }
        let displacement = result
            .field(FEA_FIELD_STRUCTURAL_DISPLACEMENT)
            .unwrap()
            .as_host_f64()
            .unwrap();
        let reactions = result
            .field(FEA_FIELD_STRUCTURAL_REACTION_FORCE)
            .unwrap()
            .as_host_f64()
            .unwrap();
        let energy = result
            .field(FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY)
            .unwrap()
            .as_host_f64()
            .unwrap()[0];
        let shear_modulus = 200.0e9 / (2.0 * (1.0 + 0.3));
        let free_vertex_y = -2.0 * newtons / shear_modulus;
        close(displacement[10], free_vertex_y, free_vertex_y.abs());
        close(energy, 0.5 * (newtons / 3.0) * -free_vertex_y, energy);
        let mut reaction_force = [0.0; 3];
        let mut reaction_moment = [0.0; 3];
        for (node, force) in mesh.topology.nodes.iter().zip(reactions.chunks_exact(3)) {
            for axis in 0..3 {
                reaction_force[axis] += force[axis];
            }
            let [x, y, z] = node.coordinates_m.canonical_value();
            reaction_moment[0] += y * force[2] - z * force[1];
            reaction_moment[1] += z * force[0] - x * force[2];
            reaction_moment[2] += x * force[1] - y * force[0];
        }
        for axis in 0..3 {
            close(reaction_force[axis], -expected_force[axis], newtons);
            close(reaction_moment[axis], -expected_moment[axis], newtons);
        }
        solutions.push((displacement[10], energy));
    }
    assert_ne!(definitions[0], definitions[1]);
    assert_ne!(workload_definitions[0], workload_definitions[1]);
    close(solutions[1].0, 2.0 * solutions[0].0, solutions[1].0.abs());
    close(solutions[1].1, 4.0 * solutions[0].1, solutions[1].1);
}
