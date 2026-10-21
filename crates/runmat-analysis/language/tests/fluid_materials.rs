pub mod support;

use runmat_analysis_core::{MaterialFluidModel, PropertyRegionTarget};
use runmat_analysis_language::*;
use runmat_geometry_core::{GeometryRegionRole, ResolvedGeometryRegion};
use runmat_meshing_core::{fixtures, ElementOrder, StableDigest};
use std::collections::BTreeMap;
use support::{source_module, TestHost};

fn check_material(body: &str) -> CheckedFeaProgram {
    let source = format!("material water {{ {body} }}");
    FeaCompiler::new(&TestHost::default()).check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", &source, 1),
        },
    })
}

#[test]
fn exact_closed_newtonian_law_keeps_shared_density_and_typed_viscosity() {
    let program = check_material("mass_density = 1000 kg/m^3; fluid { law = incompressible_newtonian; dynamic_viscosity = 0.001 Pa*s }");
    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    let material = program.materials.values().next().unwrap();
    let coefficients = material.require_incompressible_newtonian().unwrap();
    assert_eq!(coefficients.density.kilograms_per_cubic_meter, 1000.0);
    assert_eq!(coefficients.dynamic_viscosity.pascal_seconds, 0.001);
    assert!(matches!(
        material.fluid,
        Some(MaterialFluidModel::IncompressibleNewtonian { .. })
    ));
}

#[test]
fn fluid_material_rejects_missing_wrong_duplicate_and_nonpositive_properties() {
    for body in [
        "mass_density = 1000 kg/m^3; fluid { dynamic_viscosity = 0.001 Pa*s }",
        "mass_density = 1000 kg/m^3; fluid { law = compressible_newtonian; dynamic_viscosity = 0.001 Pa*s }",
        "mass_density = 1000 kg/m^3; fluid { law = incompressible_newtonian }",
        "fluid { law = incompressible_newtonian; dynamic_viscosity = 0.001 Pa*s }",
        "mass_density = 0 kg/m^3; fluid { law = incompressible_newtonian; dynamic_viscosity = 0.001 Pa*s }",
        "mass_density = 1000 kg/m^3; fluid { law = incompressible_newtonian; dynamic_viscosity = -0.001 Pa*s }",
        "mass_density = 1000 kg/m^3; fluid { law = incompressible_newtonian; dynamic_viscosity = 1 Pa }",
        "mass_density = 1000 kg/m^3; fluid { law = incompressible_newtonian; dynamic_viscosity = 0.001 Pa*s; density = 1000 kg/m^3 }",
        "mass_density = 1000 kg/m^3; fluid { law = incompressible_newtonian; dynamic_viscosity = 0.001 Pa*s; dynamic_viscosity = 0.002 Pa*s }",
    ] {
        let program = check_material(body);
        assert!(!program.is_valid(), "invalid material accepted: {body}");
    }
}

#[test]
fn authored_cell_material_compiles_through_realization_to_exact_mesh_ownership() {
    let (document, topology, _) = runmat_geometry_fixtures::exact_tetrahedron();
    let source = r#"
geometry part {
    file = "./part.step"
    units = m
    region volume : cell { select id "region_000001"; expect exactly_one }
}
mesh part on geometry.part { tetrahedra { order = linear; target_size = 0.5 m; maximum_deviation = 0.01 m } }
material water { mass_density = 1000 kg/m^3; fluid { law = incompressible_newtonian; dynamic_viscosity = 0.001 Pa*s } }
model flow { assign material.water to geometry.part.region.volume }
study steady { geometry = geometry.part; mesh = mesh.part; model = model.flow; formulation = experimental; analysis solve { fluid static {} } }
"#;
    let host = TestHost::default().with_geometry_inspection(
        "project/part.step",
        GeometryInspection {
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
    let program = FeaCompiler::new(&host).check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", source, 1),
        },
    });
    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    let resolved = program.resolve_study("project/main.fea", "steady").unwrap();
    let realized = resolved
        .study
        .realize_with_models(&resolved.models, &resolved.geometry, &[])
        .unwrap();
    let regions = BTreeMap::from([(
        "volume".into(),
        ResolvedGeometryRegion {
            role: GeometryRegionRole::Cell,
            entity_ids: vec![topology.regions[0].id.clone()],
        },
    )]);
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
    for (face, original) in mesh.topology.boundary_faces.iter_mut().zip(&topology.faces) {
        face.provenance = vec![original.id.clone()];
    }
    for (edge, original) in mesh.topology.boundary_edges.iter_mut().zip(&topology.edges) {
        edge.provenance = vec![original.id.clone()];
    }
    mesh.seal_canonical_digest().unwrap();
    let compiled = runmat_analysis_fea::study::compile_analysis_study(
        runmat_analysis_fea::study::AnalysisStudyCompilationInput {
            study: &realized,
            geometry: &document,
            regions: &regions,
            materials: &resolved.material_catalog,
            mesh: &mesh,
        },
    )
    .unwrap();
    compiled.validate_against(&mesh).unwrap();
    assert!(
        compiled.model.cfd.is_none(),
        "no coefficient authority before compiled Cell admission"
    );
    assert!(matches!(
        compiled.model.material_assignments[0].target,
        PropertyRegionTarget::Cell(_)
    ));
    assert_eq!(compiled.properties.cell_materials().len(), 1);
    assert_eq!(compiled.properties.solid_materials().len(), 0);
    let domain = runmat_analysis_fea::study::derive_canonical_fluid_material(
        &compiled.model,
        &mesh,
        &compiled.properties,
    )
    .unwrap();
    assert_eq!(domain.density.kilograms_per_cubic_meter, 1000.0);
    assert_eq!(domain.dynamic_viscosity.pascal_seconds, 0.001);
}
