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
