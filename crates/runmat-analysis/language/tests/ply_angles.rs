pub mod support;

use runmat_analysis_language::*;
use support::{source_module, TestHost};

/// Plies written on one line, with a zero and a nonzero angle, as the models docs show them.
const SOURCE: &str = r#"
geometry part {
    construct {
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }
    region body : solid { select result; expect exactly_one }
    region skin : face { select result; expect exactly_one }
}
material steel {
    mass_density = 7850 kg/m^3
    elastic { youngs_modulus = 200 GPa; poisson_ratio = 0.3 }
}
model structure {
    assign material.steel to geometry.part.region.body
    orientation axes_zero {
        cartesian {
            origin = [0 m, 0 m, 0 m]
            x_axis = [1, 0, 0]
            y_axis = [0, 1, 0]
        }
    }
    layup skin {
        reference_orientation = orientation.axes_zero
        ply_order = along_surface_normal
        reference_surface = mid_surface
        ply a { material = material.steel; thickness = 0.2 mm; angle = 0 deg }
        ply b { material = material.steel; thickness = 0.2 mm; angle = 45 deg }
        ply c { material = material.steel; thickness = 0.2 mm; angle = -45 deg }
        ply d { material = material.steel; thickness = 0.2 mm; angle = 0 rad }
    }
    assign layup.skin to geometry.part.region.skin
    assign orientation.axes_zero to geometry.part.region.skin
}
"#;

#[test]
fn zero_and_signed_ply_angles_check_and_lower_to_radians() {
    let program = FeaCompiler::new(&TestHost::default()).check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", SOURCE, 1),
        },
    });
    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    let model = program.models.values().next().unwrap();
    let layup = model.layups.first().expect("layup");
    let angles = layup
        .plies
        .iter()
        .map(|ply| ply.angle_rad.canonical_value())
        .collect::<Vec<_>>();
    let quarter = std::f64::consts::FRAC_PI_4;
    assert_eq!(angles.len(), 4);
    for (actual, expected) in angles.iter().zip([0.0, quarter, -quarter, 0.0]) {
        assert!((actual - expected).abs() < 1e-12, "{angles:?}");
    }
}
