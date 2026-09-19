mod support;

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
    assert_eq!(values[1].newtons, -1000.0);

    let overridden = authored_study
        .realize(
            authored_model,
            geometry,
            &[runmat_analysis_core::AuthoredStudyInputBinding {
                input_id: "commanded_load".into(),
                value: runmat_analysis_core::AuthoredInputValue::Quantity {
                    dimension: runmat_types::QuantityDimension::Force,
                    value_si: 2000.0,
                },
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
    assert_eq!(values[1].newtons, -2000.0);

    let out_of_bounds = authored_study.realize(
        authored_model,
        geometry,
        &[runmat_analysis_core::AuthoredStudyInputBinding {
            input_id: "commanded_load".into(),
            value: runmat_analysis_core::AuthoredInputValue::Quantity {
                dimension: runmat_types::QuantityDimension::Force,
                value_si: 6000.0,
            },
        }],
    );
    assert!(matches!(
        out_of_bounds,
        Err(runmat_analysis_core::AuthoredStudyRealizationError::InvalidStudyInput { .. })
    ));
}
