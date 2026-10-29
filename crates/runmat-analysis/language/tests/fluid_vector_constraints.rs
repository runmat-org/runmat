pub mod support;

use runmat_analysis_core::{AuthoredConstraintKind, PrescribedValue, SignalShape, Velocity};
use runmat_analysis_language::*;
use support::{source_module, TestHost};

fn check(body: &str) -> CheckedFeaProgram {
    let source = format!(
        r#"
geometry part {{
    construct {{
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }}
    region inlet : face {{ select persistent_id "INLET_ID"; expect exactly_one }}
    region wall : face {{ select persistent_id "WALL_ID"; expect exactly_one }}
}}
model flow {{ {body} }}
"#
    );
    let selection = |name: &str| {
        runmat_geometry_core::PersistentEntityId::new(
            runmat_geometry_core::PersistentEntityKind::Face,
            name.into(),
            vec!["root".into()],
        )
        .analysis_selection_id()
        .unwrap()
    };
    let source = source
        .replace("INLET_ID", &selection("inlet"))
        .replace("WALL_ID", &selection("wall"));
    FeaCompiler::new(&TestHost::default()).check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", &source, 1),
        },
    })
}

#[test]
fn fluid_inline_constraints_preserve_signed_components_and_distinct_regions() {
    let program = check(
        r#"
constraint inlet {
    fluid.velocity { target = geometry.part.region.inlet; value = [-2 m/s, 3 m/s, -4 m/s] }
}
constraint wall {
    fluid.moving_wall { target = geometry.part.region.wall; value = [5 m/s, -6 m/s, 7 m/s] }
}
"#,
    );
    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    let model = program.models.values().next().unwrap();
    let AuthoredConstraintKind::FluidVelocity {
        target: inlet,
        value,
    } = &model.constraints[0].kind
    else {
        panic!("velocity constraint");
    };
    assert_eq!(
        value,
        &PrescribedValue::Constant([-2.0, 3.0, -4.0].map(Velocity::new))
    );
    let AuthoredConstraintKind::FluidMovingWall {
        target: wall,
        velocity,
    } = &model.constraints[1].kind
    else {
        panic!("moving wall");
    };
    assert_eq!(
        velocity,
        &PrescribedValue::Constant([5.0, -6.0, 7.0].map(Velocity::new))
    );
    assert_ne!(inlet.region, wall.region);
}

#[test]
fn fluid_vector_signal_keeps_time_table_and_declared_frame_without_scalarizing() {
    let program = check(
        r#"
signal history : velocity over time {
    samples = [
        { at = 0 s, value = [-1 m/s, 2 m/s, 0 m/s] },
        { at = 1 s, value = [3 m/s, -4 m/s, 5 m/s] },
    ]
    interpolation = linear
    extrapolation = reject
    frame = boundary_normal_tangent
}
constraint inlet {
    fluid.velocity { target = geometry.part.region.inlet; value = signal.history }
}
constraint wall {
    fluid.moving_wall { target = geometry.part.region.wall; value = signal.history }
}
"#,
    );
    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    let model = program.models.values().next().unwrap();
    assert_eq!(model.signals[0].shape(), &SignalShape::Vector3);
    assert_eq!(
        model.signals[0].frame(),
        Some(&runmat_analysis_core::CoordinateFrame::BoundaryNormalTangent)
    );
    let AuthoredConstraintKind::FluidVelocity { value, .. } = &model.constraints[0].kind else {
        panic!("velocity constraint");
    };
    assert_eq!(value.signal(&model.signals), Some(&model.signals[0]));
    assert_eq!(value.study_constant(&model.signals), None);
    let AuthoredConstraintKind::FluidMovingWall { velocity, .. } = &model.constraints[1].kind
    else {
        panic!("moving wall");
    };
    assert_eq!(velocity.signal(&model.signals), Some(&model.signals[0]));
}

#[test]
fn fluid_velocity_rejects_scalar_constants_signals_wrong_dimensions_and_shapes() {
    for body in [
        "constraint inlet { fluid.velocity { target = geometry.part.region.inlet; value = 2 m/s } }",
        "constraint wall { fluid.moving_wall { target = geometry.part.region.wall; value = 2 m/s } }",
        "signal speed : velocity over constant { value = 2 m/s } constraint inlet { fluid.velocity { target = geometry.part.region.inlet; value = signal.speed } }",
        "constraint inlet { fluid.velocity { target = geometry.part.region.inlet; value = [2 m/s, 0 m/s] } }",
        "constraint inlet { fluid.velocity { target = geometry.part.region.inlet; value = [2 m/s, 0 Pa, 0 m/s] } }",
    ] {
        let program = check(body);
        assert!(!program.is_valid(), "scalar/invalid vector accepted: {body}");
    }
}
