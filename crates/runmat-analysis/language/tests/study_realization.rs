mod support;

#[path = "study_realization/study_input_static_solve.rs"]
mod study_input_static_solve;

use runmat_analysis_language::*;
use support::{source_module, TestHost};

fn check(source: &str) -> CheckedFeaProgram {
    let host = TestHost::default()
        .with_geometry("project/unused.step", 3)
        .with_function("project/unused.m", "unused", Vec::new(), Vec::new());
    FeaCompiler::new(&host).check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", source, 1),
        },
    })
}

#[test]
fn study_realization_captures_the_exact_transitive_model_composition() {
    let library_module = r#"
model fastener {
}
model mounting {
    include model.fastener
}
"#;
    let source = r#"
import model mounting from "./library.fea"

geometry part {
    construct {
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }
}
mesh part on geometry.part {
    tetrahedra {
        order = linear
        target_size = 0.1 m
        maximum_deviation = 0.01 m
    }
}
model bracket {
    include model.mounting
}
study response {
    geometry = geometry.part
    mesh = mesh.part
    model = model.bracket
    analysis solve {
        structural static {
        }
    }
}
"#;
    let host = TestHost::default().with_module("project/library.fea", library_module, 2);
    let program = FeaCompiler::new(&host).check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", source, 1),
        },
    });
    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    let study = program.studies.values().next().unwrap();
    let root = program
        .models
        .values()
        .find(|model| model.model_id == "bracket")
        .unwrap();
    let geometry = program.geometries.values().next().unwrap();
    assert!(matches!(
        study.realize(root, geometry, &[]),
        Err(runmat_analysis_core::AuthoredStudyRealizationError::MissingModel(_))
    ));
    let catalog = program
        .models
        .values()
        .map(|model| (model.reference(), model.clone()))
        .collect();
    let realized = study.realize_with_models(&catalog, geometry, &[]).unwrap();
    assert_eq!(realized.model.model_id, "bracket");
    assert_eq!(realized.included_models.len(), 2);
    assert_eq!(realized.included_models[0].model_id, "fastener");
    assert_eq!(realized.included_models[1].model_id, "mounting");
    assert!(realized
        .included_models
        .iter()
        .all(|model| model.module_path == "project/library.fea"));

    let resolved = program
        .resolve_study("project/main.fea", "response")
        .unwrap();
    assert_eq!(resolved.geometry.geometry_id, "part");
    assert_eq!(resolved.mesh.mesh_id, "part");
    assert_eq!(resolved.models.len(), 3);
    assert!(resolved
        .models
        .contains_key(&runmat_analysis_core::AuthoredModelRef {
            module_path: "project/library.fea".into(),
            model_id: "fastener".into(),
        }));
}

#[test]
fn study_resolution_retains_derived_geometry_dependencies_in_execution_order() {
    let source = r#"
geometry stock {
    construct {
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }
}
geometry drilled from geometry.stock {
    derive {
        feature hole = cylinder(
            radius = 0.1 m,
            height = 1 m,
            axis = global([0, 0, 1]),
            origin = [0.5, 0.5, 0] m,
        )
        result = subtract(source.result, feature.hole)
    }
}
geometry relieved from geometry.drilled {
    derive {
        feature relief = cylinder(
            radius = 0.05 m,
            height = 1 m,
            axis = global([0, 1, 0]),
            origin = [0.25, 0, 0.25] m,
        )
        result = subtract(source.result, feature.relief)
    }
}
mesh volume on geometry.relieved {
    tetrahedra {
        order = linear
        target_size = 0.25 m
        maximum_deviation = 0.01 m
    }
}
model structure {}
study response {
    geometry = geometry.relieved
    mesh = mesh.volume
    model = model.structure
    analysis solve { structural static {} }
}
"#;
    let program = check(source);
    assert!(program.is_valid(), "{:#?}", program.diagnostics);

    let resolved = program
        .resolve_study("project/main.fea", "response")
        .unwrap();
    assert_eq!(
        resolved
            .geometry_dependencies
            .iter()
            .map(|geometry| geometry.geometry_id.as_str())
            .collect::<Vec<_>>(),
        ["stock", "drilled"]
    );
    assert_eq!(resolved.geometry.geometry_id, "relieved");
}

#[test]
fn study_realization_substitutes_regions_and_evaluates_parameterized_loads() {
    let source = r#"
geometry part {
    construct {
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }
    region pad : face {
        select result
        expect one_or_more
    }
}
mesh part on geometry.part {
    tetrahedra {
        order = linear
        target_size = 0.1 m
        maximum_deviation = 0.01 m
    }
}
model loaded(region pad : region<face>) {
    input service_load : force
    load downward {
        force on input.pad = [0, -input.service_load, 0]
    }
    load_case service { include load.downward }
}
study response {
    input commanded_load : force {
        default = 1 kN
        minimum = 0 N
        maximum = 5 kN
    }
    geometry = geometry.part
    mesh = mesh.part
    model = model.loaded(
        pad = geometry.part.region.pad,
        service_load = input.commanded_load,
    )
    analysis solve {
        structural static {
        }
    }
}
"#;
    let program = check(source);
    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    let authored_model = program.models.values().next().unwrap();
    let authored_study = program.studies.values().next().unwrap();
    let geometry = program.geometries.values().next().unwrap();
    let realized = authored_study
        .realize(authored_model, geometry, &[])
        .unwrap();
    assert!(realized.model.inputs.is_empty());
    assert!(realized.model.parameterized_signals.is_empty());
    let runmat_analysis_core::AuthoredExcitationTarget::Region(
        runmat_analysis_core::AuthoredRegionSource::Geometry(region),
    ) = &realized.model.excitations[0].target
    else {
        panic!("model region input must resolve to the exact study geometry region");
    };
    assert_eq!(region.region_id, "pad");
    let runmat_analysis_core::PhysicalSignal::Force(signal) = &realized.model.signals[0] else {
        panic!("parameterized force must realize to a concrete force signal");
    };
    let runmat_analysis_core::SignalRepresentation::Constant(
        runmat_analysis_core::SignalValue::Vector3(values),
    ) = &signal.representation
    else {
        panic!("parameterized load must realize as a constant vector");
    };
    assert_eq!(values[1].canonical_value(), -1000.0);

    let overridden = authored_study
        .realize(
            authored_model,
            geometry,
            &[runmat_analysis_core::AuthoredStudyInputBinding {
                input_id: "commanded_load".into(),
                value: runmat_analysis_core::AuthoredInputValue::Quantity(
                    runmat_types::physical::DimensionedScalar::try_from_dimension(
                        runmat_types::PhysicalDimension::FORCE,
                        None,
                        2000.0,
                    )
                    .unwrap(),
                ),
            }],
        )
        .unwrap();
    let runmat_analysis_core::PhysicalSignal::Force(signal) = &overridden.model.signals[0] else {
        panic!("parameterized force must realize to a concrete force signal");
    };
    let runmat_analysis_core::SignalRepresentation::Constant(
        runmat_analysis_core::SignalValue::Vector3(values),
    ) = &signal.representation
    else {
        panic!("parameterized load must realize as a constant vector");
    };
    assert_eq!(values[1].canonical_value(), -2000.0);

    let out_of_bounds = authored_study.realize(
        authored_model,
        geometry,
        &[runmat_analysis_core::AuthoredStudyInputBinding {
            input_id: "commanded_load".into(),
            value: runmat_analysis_core::AuthoredInputValue::Quantity(
                runmat_types::physical::DimensionedScalar::try_from_dimension(
                    runmat_types::PhysicalDimension::FORCE,
                    None,
                    6000.0,
                )
                .unwrap(),
            ),
        }],
    );
    assert!(matches!(
        out_of_bounds,
        Err(runmat_analysis_core::AuthoredStudyRealizationError::InvalidStudyInput { .. })
    ));
}

// witness: FEA-THERMAL-PUBLIC-001/film_coefficient_input @
//   thermal_steady/tet4|tet10/numerical_kernel; coverage: partial;
//   boundary: Film-coefficient and emissivity inputs realize per bound sweep case at the language
//   layer; no solve is run.
#[test]
fn film_coefficient_and_emissivity_inputs_realize_per_bound_case() {
    let source = r#"
geometry part {
    construct {
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }
    region surface : face {
        select result
        expect one_or_more
    }
}
mesh part on geometry.part {
    tetrahedra {
        order = linear
        target_size = 0.1 m
        maximum_deviation = 0.01 m
    }
}
model cooled {
    input film : heat_transfer_coefficient { default = 35 W/m^2/K }
    input surface_emissivity : dimensionless { default = 0.8 }
    thermal { reference_temperature = 25 degC }
    load air_cooling {
        thermal.convection {
            target = geometry.part.region.surface
            ambient_temperature = 25 degC
            coefficient = 2 * input.film
        }
    }
    load radiating {
        thermal.radiation {
            target = geometry.part.region.surface
            ambient_temperature = 25 degC
            emissivity = input.surface_emissivity
        }
    }
}
study cooling {
    input film : heat_transfer_coefficient {
        default = 35 W/m^2/K
        minimum = 0 W/m^2/K
    }
    geometry = geometry.part
    mesh = mesh.part
    model = model.cooled(film = input.film)
    analysis heat { thermal steady {} }
}
"#;
    let program = check(source);
    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    let authored_model = program.models.values().next().unwrap();
    assert!(authored_model.boundaries.iter().all(|boundary| matches!(
        &boundary.kind,
        runmat_analysis_core::AuthoredBoundaryKind::ThermalConvection {
            coefficient: runmat_analysis_core::AuthoredBoundaryCoefficient::ModelInput(_),
            ..
        } | runmat_analysis_core::AuthoredBoundaryKind::ThermalRadiation {
            emissivity: runmat_analysis_core::AuthoredBoundaryCoefficient::ModelInput(_),
            ..
        }
    )));
    let authored_study = program.studies.values().next().unwrap();
    let geometry = program.geometries.values().next().unwrap();
    let realize = |film: f64| {
        let realized = authored_study
            .realize(
                authored_model,
                geometry,
                &[runmat_analysis_core::AuthoredStudyInputBinding {
                    input_id: "film".into(),
                    value: runmat_analysis_core::AuthoredInputValue::Quantity(
                        runmat_types::physical::DimensionedScalar::try_from_dimension(
                            runmat_types::PhysicalDimension::HEAT_TRANSFER_COEFFICIENT,
                            None,
                            film,
                        )
                        .unwrap(),
                    ),
                }],
            )
            .unwrap();
        let mut coefficient = None;
        let mut emissivity = None;
        for boundary in &realized.model.boundaries {
            match &boundary.kind {
                runmat_analysis_core::AuthoredBoundaryKind::ThermalConvection {
                    coefficient: runmat_analysis_core::AuthoredBoundaryCoefficient::Constant(value),
                    ..
                } => coefficient = Some(value.canonical_value()),
                runmat_analysis_core::AuthoredBoundaryKind::ThermalRadiation {
                    emissivity: runmat_analysis_core::AuthoredBoundaryCoefficient::Constant(value),
                    ..
                } => emissivity = Some(value.canonical_value()),
                other => panic!("realized boundary keeps an input-dependent value: {other:?}"),
            }
        }
        (coefficient.unwrap(), emissivity.unwrap())
    };
    assert_eq!(realize(10.0), (20.0, 0.8));
    assert_eq!(realize(100.0), (200.0, 0.8));
    assert!(matches!(
        authored_study.realize(
            authored_model,
            geometry,
            &[runmat_analysis_core::AuthoredStudyInputBinding {
                input_id: "film".into(),
                value: runmat_analysis_core::AuthoredInputValue::Quantity(
                    runmat_types::physical::DimensionedScalar::try_from_dimension(
                        runmat_types::PhysicalDimension::HEAT_TRANSFER_COEFFICIENT,
                        None,
                        -1.0,
                    )
                    .unwrap(),
                ),
            }],
        ),
        Err(runmat_analysis_core::AuthoredStudyRealizationError::InvalidStudyInput { .. })
    ));
}

// witness: FEA-THERMAL-PUBLIC-001/film_coefficient_input @
//   thermal_steady/tet4|tet10/numerical_kernel; coverage: partial;
//   boundary: A film-coefficient input rejects a mismatched quantity.
#[test]
fn film_coefficient_input_rejects_a_mismatched_quantity() {
    let source = r#"
geometry part {
    construct {
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }
    region surface : face {
        select result
        expect one_or_more
    }
}
model cooled {
    input film : heat_flux { default = 35 W/m^2 }
    thermal { reference_temperature = 25 degC }
    load air_cooling {
        thermal.convection {
            target = geometry.part.region.surface
            ambient_temperature = 25 degC
            coefficient = input.film
        }
    }
}
"#;
    let program = check(source);
    assert!(!program.is_valid());
    assert!(program
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.message.contains("`coefficient`")));
}
