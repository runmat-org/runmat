pub mod support;

use runmat_analysis_language::*;
use support::{source_module, TestHost};

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
        runmat_analysis_core::AuthoredPropertyRegionSource::Geometry(region),
    ) = &assignment.target
    else {
        panic!("expected solid geometry region");
    };
    assert_eq!(region.module_path, "project/geometry.fea");
    assert_eq!(region.geometry_id, "part");
    assert_eq!(region.region_id, "body");
    assert_eq!(region.role, runmat_geometry_core::GeometryRegionRole::Solid);
}

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
    assert_eq!(model.property_inputs.len(), 2);
    let assignment = &model.material_assignments[0];
    assert!(matches!(
        assignment.material,
        runmat_analysis_core::AuthoredMaterialSource::ModelInput(_)
    ));
    assert!(matches!(
        assignment.target,
        runmat_analysis_core::AuthoredPropertyTarget::Solid(
            runmat_analysis_core::AuthoredPropertyRegionSource::ModelInput(_)
        )
    ));
}

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
