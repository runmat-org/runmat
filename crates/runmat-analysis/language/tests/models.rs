pub mod support;

use runmat_analysis_language::*;
use support::{source_module, TestHost};

// witness: FEA-AUTHORING-LANGUAGE-002/model_composition @
//   shared_infrastructure/mesh_independent/source_check; coverage: partial;
//   boundary: Included models and material assignments retain their qualified source modules across
//   imports; the full list of model members is not exercised.
#[test]
fn model_composition_and_material_assignments_retain_qualified_sources() {
    let geometry_module = r#"
geometry part {
    construct {
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }
    region body : solid {
        select result
        expect exactly_one
    }
}
"#;
    let library_module = r#"
material steel {
    mass_density = 7850 kg/m^3
}
model mounting {
}
"#;
    let source = r#"
import geometry part from "./geometry.fea"
import material steel from "./library.fea"
import model mounting from "./library.fea"

model bracket {
    include model.mounting
    assign material.steel to geometry.part.region.body
}
"#;
    let host = TestHost::default()
        .with_module("project/geometry.fea", geometry_module, 2)
        .with_module("project/library.fea", library_module, 3);
    let program = FeaCompiler::new(&host).check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", source, 1),
        },
    });
    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    let model = program
        .models
        .iter()
        .find_map(|((path, key), model)| {
            (path.as_str() == "project/main.fea" && key.name == "bracket").then_some(model)
        })
        .unwrap();
    assert_eq!(model.includes[0].model.module_path, "project/library.fea");
    assert_eq!(model.includes[0].model.model_id, "mounting");
    let assignment = &model.material_assignments[0];
    let runmat_analysis_core::AuthoredMaterialSource::Declaration(material) = &assignment.material
    else {
        panic!("expected declared material");
    };
    assert_eq!(material.module_path, "project/library.fea");
    assert_eq!(material.material_id, "steel");
    let runmat_analysis_core::AuthoredPropertyTarget::Solid(
        runmat_analysis_core::AuthoredRegionSource::Geometry(region),
    ) = &assignment.target
    else {
        panic!("expected solid geometry region");
    };
    assert_eq!(region.module_path, "project/geometry.fea");
    assert_eq!(region.geometry_id, "part");
    assert_eq!(region.region_id, "body");
    assert_eq!(region.role, runmat_geometry_core::GeometryRegionRole::Solid);
}

// witness: FEA-AUTHORING-GEOMETRY-002/typed_property_assignments @
//   shared_infrastructure/mesh_independent/source_check; coverage: partial;
//   boundary: Parameterized model assignments remain strongly typed; section, layup and orientation
//   forms are not covered.
#[test]
fn parameterized_model_assignments_remain_strongly_typed() {
    let source = r#"
model reusable(region body : region<solid>) {
    input body_material : material
    assign input.body_material to input.body
}
"#;
    let program = FeaCompiler::new(&TestHost::default()).check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", source, 1),
        },
    });
    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    let model = program.models.values().next().unwrap();
    assert_eq!(model.inputs.len(), 2);
    let assignment = &model.material_assignments[0];
    assert!(matches!(
        assignment.material,
        runmat_analysis_core::AuthoredMaterialSource::ModelInput(_)
    ));
    assert!(matches!(
        assignment.target,
        runmat_analysis_core::AuthoredPropertyTarget::Solid(
            runmat_analysis_core::AuthoredRegionSource::ModelInput(_)
        )
    ));
}

// witness: FEA-AUTHORING-LANGUAGE-002/model_composition @
//   shared_infrastructure/mesh_independent/source_check; coverage: partial;
//   boundary: Composition cycles and conflicting assignments are rejected.
#[test]
fn model_composition_rejects_cycles_and_conflicting_assignments() {
    let source = r#"
geometry part {
    construct {
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }
    region body : solid {
        select result
        expect exactly_one
    }
}
material steel { mass_density = 7850 kg/m^3 }
material aluminum { mass_density = 2700 kg/m^3 }
model base {
    assign material.steel to geometry.part.region.body
}
model conflict {
    include model.base
    assign material.aluminum to geometry.part.region.body
}
model cycle_a {
    include model.cycle_b
}
model cycle_b {
    include model.cycle_a
}
"#;
    let program = FeaCompiler::new(&TestHost::default()).check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", source, 1),
        },
    });
    assert!(!program.is_valid());
    assert!(program
        .models
        .values()
        .any(|model| model.model_id == "base"));
    assert!(!program
        .models
        .values()
        .any(|model| matches!(model.model_id.as_str(), "conflict" | "cycle_a" | "cycle_b")));
    assert!(program.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == "fea.semantic.model" && diagnostic.message.contains("conflicting")
    }));
    assert!(program.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == "fea.semantic.model" && diagnostic.message.contains("cycle")
    }));
}

#[test]
fn sections_layups_and_orientations_lower_as_distinct_typed_properties() {
    let source = r#"
geometry part {
    construct {
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }
    region rail : edge {
        select result
        expect exactly_one
    }
    region skin : face {
        select result
        expect exactly_one
    }
}
material aluminum { mass_density = 2700 kg/m^3 }
material carbon { mass_density = 1600 kg/m^3 }
model airframe {
    orientation laminate_axes {
        cartesian {
            origin = [0 m, 0 m, 0 m]
            x_axis = [1, 0, 0]
            y_axis = [0, 1, 0]
        }
    }
    section rail_section {
        rectangular_beam {
            width = 20 mm
            height = 40 mm
        }
    }
    section inspection_skin {
        homogeneous_shell {
            thickness = 2 mm
            reference_surface = offset_from_mid_surface
            reference_offset = 0.25 mm
            shear_correction = 0.8333333333333333
        }
    }
    layup skin_laminate {
        reference_orientation = orientation.laminate_axes
        ply_order = along_surface_normal
        reference_surface = mid_surface
        ply zero {
            material = material.carbon
            thickness = 0.2 mm
            angle = 0 deg
        }
        ply ninety {
            material = material.carbon
            thickness = 0.2 mm
            angle = 90 deg
        }
    }
    assign material.aluminum to geometry.part.region.rail
    assign section.rail_section to geometry.part.region.rail
    assign layup.skin_laminate to geometry.part.region.skin
    assign orientation.laminate_axes to geometry.part.region.skin
}
"#;
    let program = FeaCompiler::new(&TestHost::default()).check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", source, 1),
        },
    });
    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    let model = program.models.values().next().unwrap();
    assert_eq!(model.sections.len(), 2);
    assert_eq!(model.orientations.len(), 1);
    assert_eq!(model.layups.len(), 1);
    assert_eq!(model.layups[0].plies.len(), 2);
    assert_eq!(
        model.layups[0].plies[1].angle_rad.canonical_value(),
        std::f64::consts::FRAC_PI_2
    );
    assert_eq!(model.property_assignments.len(), 3);
    assert!(matches!(
        model.sections[0],
        runmat_analysis_core::AuthoredSectionDefinition::RectangularBeam(_)
    ));
    let runmat_analysis_core::AuthoredSectionDefinition::HomogeneousShell(shell) =
        &model.sections[1]
    else {
        panic!("second section must be the homogeneous shell")
    };
    assert_eq!(
        shell.reference_surface,
        runmat_analysis_core::HomogeneousShellReferenceSurface::OffsetFromMidSurface(0.00025)
    );
}

// witness: FEA-AUTHORING-GEOMETRY-002/complete_property_coverage @
//   shared_infrastructure/mesh_independent/source_check; coverage: partial;
//   boundary: Property roles and composite ownership fail closed; gaps, overlaps and stale
//   revisions across a whole mesh are checked in the properties requirement.
#[test]
fn property_roles_and_composite_ownership_fail_closed() {
    let source = r#"
geometry part {
    construct {
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }
    region skin : face {
        select result
        expect exactly_one
    }
}

material carbon { mass_density = 1600 kg/m^3 }
model invalid {
    orientation axes {
        global_cartesian {}
    }
    section rail {
        rectangular_beam {
            width = 1 mm
            height = 1 mm
        }
    }
    layup skin_laminate {
        reference_orientation = orientation.axes
        ply_order = along_surface_normal
        reference_surface = mid_surface
        ply one {
            material = material.carbon
            thickness = 0.1 mm
            angle = 0 deg
        }
    }
    assign material.carbon to geometry.part.region.skin
    assign section.rail to geometry.part.region.skin
    assign layup.skin_laminate to geometry.part.region.skin
}
"#;
    let program = FeaCompiler::new(&TestHost::default()).check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", source, 1),
        },
    });
    assert!(!program.is_valid());
    assert!(program.models.is_empty());
    assert!(program.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == "fea.semantic.model"
            && (diagnostic.message.contains("section assignments")
                || diagnostic.message.contains("laminate ownership"))
    }));
}

#[test]
fn essential_constraints_lower_across_physics_with_exact_regions_and_units() {
    let source = r#"
geometry part {
    construct {
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }
    region wall : face {
        select result
        expect one_or_more
    }
    region volume : solid {
        select result
        expect exactly_one
    }
}
model multiphysics {
    constraint support {
        structural.fixed { target = geometry.part.region.wall }
    }
    constraint temperature {
        thermal.temperature {
            target = geometry.part.region.wall
            value = 20 degC
        }
    }
    constraint voltage {
        electrical.potential {
            target = geometry.part.region.wall
            value = 12 V
        }
    }
    constraint terminal_current {
        electrical.integral_current {
            target = geometry.part.region.wall
            value = 2 A
        }
    }
    constraint magnetic_ground {
        magnetic.vector_potential_ground { target = geometry.part.region.wall }
    }
    constraint magnetic_value {
        magnetic.vector_potential {
            target = geometry.part.region.wall
            value = 0 T*m
        }
    }
    constraint magnetic_gauge {
        magnetic.coulomb_gauge { target = geometry.part.region.volume }
    }
    constraint sound_source {
        acoustic.pressure {
            target = geometry.part.region.wall
            value = 1 Pa
        }
    }
    constraint inlet {
        fluid.velocity {
            target = geometry.part.region.wall
            value = [3 m/s, 0 m/s, 0 m/s]
        }
    }
}
"#;
    let program = FeaCompiler::new(&TestHost::default()).check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", source, 1),
        },
    });
    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    let model = program.models.values().next().unwrap();
    assert_eq!(model.constraints.len(), 9);
    let temperature = model
        .constraints
        .iter()
        .find(|constraint| constraint.constraint_id == "temperature")
        .unwrap();
    let runmat_analysis_core::AuthoredConstraintKind::ThermalTemperature {
        value: runmat_analysis_core::PrescribedValue::Constant(value),
        ..
    } = &temperature.kind
    else {
        panic!("expected thermal constraint");
    };
    assert_eq!(value.canonical_value(), 293.15);
}

#[test]
fn essential_constraints_reject_wrong_topology_and_dimensions() {
    let source = r#"
geometry part {
    construct {
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }
    region volume : solid {
        select result
        expect exactly_one
    }
}
model invalid {
    constraint support {
        structural.fixed { target = geometry.part.region.volume }
    }
    constraint temperature {
        thermal.temperature {
            target = geometry.part.region.volume
            value = 4 Pa
        }
    }
}
"#;
    let program = FeaCompiler::new(&TestHost::default()).check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", source, 1),
        },
    });
    assert!(!program.is_valid());
    assert!(program.models.is_empty());
    assert!(program.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == "fea.semantic.model"
            && (diagnostic.message.contains("boundary topology")
                || diagnostic.message.contains("required dimension"))
    }));
}
