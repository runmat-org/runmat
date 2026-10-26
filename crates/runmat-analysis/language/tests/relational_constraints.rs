pub mod support;

use runmat_analysis_language::*;
use support::{source_module, TestHost};

#[test]
fn relational_constraints_lower_every_canonical_family_shape() {
    let source = r#"
geometry part {
    construct {
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }
    region left : face {
        select result
        expect one_or_more
    }
    region right : face {
        select result
        expect one_or_more
    }
}
model periodic_cell {
    constraint prescribed {
        structural.prescribed {
            target = geometry.part.region.left
            frame = global
            ux = 1 mm
            rz = 2 deg
        }
    }
    constraint normal {
        structural.normal_displacement {
            target = geometry.part.region.left
            value = 0 mm
        }
    }
    constraint coupling {
        structural.rigid_coupling {
            target = geometry.part.region.left
            reference_point_id = "control"
            position = [0 m, 0 m, 0 m]
        }
    }
    constraint tie {
        structural.tie {
            primary = geometry.part.region.left
            secondary = geometry.part.region.right
            tolerance = 0.01 mm
        }
    }
    constraint periodic {
        structural.periodic {
            primary = geometry.part.region.left
            secondary = geometry.part.region.right
            translation = [1 m, 0 m, 0 m]
        }
    }
    constraint equation {
        structural.equation {
            terms = [
                { target = geometry.part.region.left, dof = ux, coefficient = 1 },
                { target = geometry.part.region.right, dof = ux, coefficient = -1 },
            ]
            right_hand_side = 0 m
        }
    }
    constraint thermal_jump {
        thermal.affine_jump {
            primary = geometry.part.region.left
            secondary = geometry.part.region.right
            jump = 5 K
        }
    }
    constraint electrical_periodic {
        electrical.periodic {
            primary = geometry.part.region.left
            secondary = geometry.part.region.right
        }
    }
    constraint magnetic_phase {
        magnetic.phase_shifted_periodic {
            primary = geometry.part.region.left
            secondary = geometry.part.region.right
            rotation_origin = [0 m, 0 m, 0 m]
            rotation_axis = [0, 0, 1]
            rotation_angle = 90 deg
            phase = 45 deg
        }
    }
    constraint acoustic_phase {
        acoustic.phase_shifted_periodic {
            primary = geometry.part.region.left
            secondary = geometry.part.region.right
            phase = 30 deg
        }
    }
    constraint fluid_periodic {
        fluid.periodic {
            primary = geometry.part.region.left
            secondary = geometry.part.region.right
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
    assert_eq!(model.constraints.len(), 11);
    assert!(matches!(
        model.constraints[0].kind,
        runmat_analysis_core::AuthoredConstraintKind::StructuralPrescribed(_)
    ));
    assert!(matches!(
        model.constraints[5].kind,
        runmat_analysis_core::AuthoredConstraintKind::StructuralEquation(_)
    ));
    assert!(matches!(
        model.constraints[8].kind,
        runmat_analysis_core::AuthoredConstraintKind::MagneticPhaseShiftedPeriodic(_)
    ));
}

#[test]
fn relational_constraints_reject_incomplete_or_ambiguous_kinematics() {
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
}

model invalid {
    constraint empty_dofs {
        structural.prescribed {
            target = geometry.part.region.wall
            frame = global
        }
    }
    constraint same_pair {
        structural.periodic {
            primary = geometry.part.region.wall
            secondary = geometry.part.region.wall
            translation = [1 m, 0 m, 0 m]
            rotation_angle = 30 deg
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
        diagnostic.message.contains("degree of freedom")
            || diagnostic.message.contains("periodic transform")
    }));
}

#[test]
fn prescribed_constraints_bind_typed_signals_and_base_motion_owns_initial_kinematics() {
    let source = r#"
geometry part {
    construct {
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }
    region support : face {
        select result
        expect one_or_more
    }
}
model driven {
    signal base_acceleration : acceleration over time {
        samples = [
            { at = 0 s, value = [0 m/s^2, 0 m/s^2, 0 m/s^2] },
            { at = 1 s, value = [2 m/s^2, 0 m/s^2, 0 m/s^2] },
        ]
        interpolation = linear
        extrapolation = reject
        frame = global
    }
    signal imposed_displacement : displacement over time {
        samples = [
            { at = 0 s, value = 0 mm },
            { at = 1 s, value = 1 mm },
        ]
        interpolation = linear
        extrapolation = reject
    }
    signal thermal_cycle : temperature over time {
        samples = [
            { at = 0 s, value = 293.15 K },
            { at = 1 s, value = 303.15 K },
        ]
        interpolation = linear
        extrapolation = reject
    }
    constraint moving_support {
        structural.prescribed_motion {
            target = geometry.part.region.support
            translations = [
                {
                    dof = ux,
                    acceleration = signal.base_acceleration,
                    initial_displacement = 0 m,
                    initial_velocity = 0 m/s,
                    reconstruction = piecewise_linear_acceleration_exact,
                },
            ]
        }
    }
    constraint imposed {
        structural.prescribed {
            target = geometry.part.region.support
            frame = global
            uy = signal.imposed_displacement
        }
    }
    constraint prescribed_temperature {
        thermal.temperature {
            target = geometry.part.region.support
            value = signal.thermal_cycle
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
    let runmat_analysis_core::AuthoredConstraintKind::StructuralPrescribedMotion(motion) =
        &model.constraints[0].kind
    else {
        panic!("expected prescribed structural motion");
    };
    assert_eq!(motion.translations.len(), 1);
    let runmat_analysis_core::StructuralKinematicHistory::AccelerationDriven {
        acceleration,
        initial_displacement,
        initial_velocity,
        reconstruction,
    } = &motion.translations[0].history;
    assert_eq!(acceleration.0 .0, "base_acceleration");
    assert_eq!(initial_displacement.canonical_value(), 0.0);
    assert_eq!(initial_velocity.canonical_value(), 0.0);
    assert_eq!(
        *reconstruction,
        runmat_analysis_core::StructuralMotionReconstruction::PiecewiseLinearAccelerationExact
    );
    let runmat_analysis_core::AuthoredConstraintKind::StructuralPrescribed(prescribed) =
        &model.constraints[1].kind
    else {
        panic!("expected prescribed structural degrees of freedom");
    };
    assert!(matches!(
        &prescribed.translations[0].value,
        runmat_analysis_core::PrescribedValue::Signal(id) if id.0 == "imposed_displacement"
    ));
    let runmat_analysis_core::AuthoredConstraintKind::ThermalTemperature { value, .. } =
        &model.constraints[2].kind
    else {
        panic!("expected prescribed temperature");
    };
    assert!(matches!(
        value,
        runmat_analysis_core::PrescribedValue::Signal(id) if id.0 == "thermal_cycle"
    ));
}

#[test]
fn prescribed_motion_rejects_wrong_signal_contract_and_prescribed_quantity() {
    let source = r#"
geometry part {
    construct {
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }
    region support : face {
        select result
        expect one_or_more
    }
}
model invalid_motion {
    signal scalar_acceleration : acceleration over time {
        samples = [
            { at = 0 s, value = 0 m/s^2 },
            { at = 1 s, value = 1 m/s^2 },
        ]
        interpolation = linear
        extrapolation = reject
    }
    constraint moving_support {
        structural.prescribed_motion {
            target = geometry.part.region.support
            translations = [
                {
                    dof = ux,
                    acceleration = signal.scalar_acceleration,
                    initial_displacement = 0 m,
                    initial_velocity = 0 m/s,
                    reconstruction = piecewise_linear_acceleration_exact,
                },
            ]
        }
    }
}
model invalid_prescribed_value {
    signal pressure_history : pressure over constant { value = 1 Pa }
    constraint imposed {
        structural.prescribed {
            target = geometry.part.region.support
            frame = global
            ux = signal.pressure_history
        }
    }
}
model duplicate_motion_dof {
    signal vector_acceleration : acceleration over time {
        samples = [
            { at = 0 s, value = [0 m/s^2, 0 m/s^2, 0 m/s^2] },
            { at = 1 s, value = [1 m/s^2, 0 m/s^2, 0 m/s^2] },
        ]
        interpolation = linear
        extrapolation = reject
        frame = global
    }
    constraint moving_support {
        structural.prescribed_motion {
            target = geometry.part.region.support
            translations = [
                {
                    dof = ux,
                    acceleration = signal.vector_acceleration,
                    initial_displacement = 0 m,
                    initial_velocity = 0 m/s,
                    reconstruction = piecewise_linear_acceleration_exact,
                },
                {
                    dof = ux,
                    acceleration = signal.vector_acceleration,
                    initial_displacement = 0 m,
                    initial_velocity = 0 m/s,
                    reconstruction = piecewise_linear_acceleration_exact,
                },
            ]
        }
    }
}
model nonlinear_waveform_motion {
    signal acceleration_wave : acceleration over time {
        waveform = sine
        amplitude = [1 m/s^2, 0 m/s^2, 0 m/s^2]
        start = 0 s
        frequency = 1 Hz
        phase = 0
        frame = global
    }
    constraint moving_support {
        structural.prescribed_motion {
            target = geometry.part.region.support
            translations = [
                {
                    dof = ux,
                    acceleration = signal.acceleration_wave,
                    initial_displacement = 0 m,
                    initial_velocity = 0 m/s,
                    reconstruction = piecewise_linear_acceleration_exact,
                },
            ]
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
    assert!(program.diagnostics.iter().any(|diagnostic| diagnostic
        .message
        .contains("global vector acceleration signal")));
    assert!(program
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.message.contains("declared scalar signal")));
    assert!(program
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.message.contains("repeats a degree of freedom")));
    assert!(program
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.message.contains("piecewise-linear time history")));
}
