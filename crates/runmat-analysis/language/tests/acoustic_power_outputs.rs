pub mod support;
use runmat_analysis_core::{ObservationQuantity, ObservationRequest, SpatialReduction};
use runmat_analysis_language::*;
use support::{source_module, TestHost};

fn check(outputs: &str) -> CheckedFeaProgram {
    let source = format!(
        r#"
geometry part {{
    construct {{
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }}
    region outlet : face {{
        select result
        expect one_or_more
    }}
}}
mesh part on geometry.part {{
    tetrahedra {{ order = linear
        target_size = 0.5 m
        maximum_deviation = 0.01 m }}
}}
model physics {{ }}
study response {{
    geometry = geometry.part
    mesh = mesh.part
    model = model.physics
    analysis sound {{ acoustic harmonic {{ frequencies = [100] Hz }} }}
    {outputs}
}}
"#
    );
    FeaCompiler::new(&TestHost::default()).check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", &source, 1),
        },
    })
}

// witness: FEA-PUBLIC-LANGUAGE-001/output_expression_forms @
//   shared_infrastructure/mesh_independent/source_check; coverage: partial;
//   boundary: sound_power and intensity flux lower to the same normal-flux observation; the other
//   output forms are separate.
#[test]
fn sound_power_and_intensity_flux_lower_to_the_same_normal_flux_observation() {
    for output in [
        "output power = sound_power(analysis.sound, through = geometry.part.region.outlet)",
        "output power = flux(analysis.sound.field.element.acoustic_intensity, through = geometry.part.region.outlet)",
    ] {
        let program = check(output);
        assert!(program.is_valid(), "{output}: {:#?}", program.diagnostics);
        let step = &program.studies.values().next().unwrap().sequence.steps[0];
        let spec = step
            .observations
            .iter()
            .find_map(|request| match request {
                ObservationRequest::Reduction(spec) => Some(spec),
                _ => None,
            })
            .expect("reduction observation");
        assert_eq!(spec.quantity, ObservationQuantity::AcousticIntensity);
        assert_eq!(
            spec.reduction.as_ref().map(|reduction| reduction.spatial),
            Some(SpatialReduction::NormalFlux)
        );
    }
}

// witness: FEA-PUBLIC-LANGUAGE-001/output_expression_forms @
//   shared_infrastructure/mesh_independent/source_check; coverage: partial;
//   boundary: Flux outputs reject volume regions and scalar fields.
#[test]
fn flux_rejects_volume_regions_and_scalar_fields() {
    assert!(!check(
        "output bad = flux(analysis.sound.field.node.acoustic_pressure, through = geometry.part.region.outlet)"
    )
    .is_valid());
}
