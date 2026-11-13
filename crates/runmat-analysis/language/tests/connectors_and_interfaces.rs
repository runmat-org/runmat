pub mod support;

use runmat_analysis_language::*;
use support::{source_module, TestHost};

#[test]
fn radiation_requires_kelvin_and_dimensionless_physical_emissivity() {
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
model heated {
    load radiation {
        thermal.radiation {
            target = geometry.part.region.wall
            ambient_temperature = AMBIENT
            emissivity = EMISSIVITY
        }
    }
}
"#;
    for (ambient, emissivity, valid) in [
        ("0 K", "0", true),
        ("300 K", "1", true),
        ("-1 K", "0.8", false),
        ("300 K", "-0.1", false),
        ("300 K", "1.1", false),
        ("300 K", "0.8 m", false),
        ("300 m", "0.8", false),
    ] {
        let source = source
            .replace("AMBIENT", ambient)
            .replace("EMISSIVITY", emissivity);
        let host = TestHost::default();
        let program = FeaCompiler::new(&host).check(CheckRequest {
            modules: ModuleGraphRequest {
                source_root: LogicalSourcePath::new("project").unwrap(),
                root: source_module("project/main.fea", &source, 1),
            },
        });
        assert_eq!(
            program.is_valid(),
            valid,
            "{ambient}, {emissivity}: {:#?}",
            program.diagnostics
        );
    }
}

#[test]
fn natural_boundaries_connectors_and_interfaces_lower_with_typed_physics() {
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
    region opposite : face {
        select result
        expect one_or_more
    }
    region anchor : vertex {
        select result
        expect exactly_one
    }
    region attachment : vertex {
        select result
        expect exactly_one
    }
}
model coupled {
    load pressure {
        structural.pressure {
            target = geometry.part.region.wall
            value = 2 MPa
        }
    }
    load heat {
        thermal.heat_flux {
            target = geometry.part.region.wall
            value = 5 W/m^2
        }
    }
    load convection {
        thermal.convection {
            target = geometry.part.region.wall
            ambient_temperature = 20 degC
            coefficient = 10 W/m^2/K
        }
    }
    load insulation {
        magnetic.insulation { target = geometry.part.region.wall }
    }
    load ambient_radiation {
        thermal.radiation {
            target = geometry.part.region.wall
            ambient_temperature = 20 degC
            emissivity = 0.8
        }
    }
    load radiation {
        acoustic.radiation { target = geometry.part.region.wall }
    }
    load impedance {
        acoustic.impedance {
            target = geometry.part.region.wall
            impedance = 400 Pa*s/m
        }
    }
    load outlet {
        fluid.outlet_pressure {
            target = geometry.part.region.wall
            value = 101.325 kPa
        }
    }
    connector spring {
        axial_spring {
            primary = geometry.part.region.anchor
            secondary = geometry.part.region.attachment
            stiffness = 4 kN/m
        }
    }
    connector dashpot {
        axial_dashpot {
            primary = geometry.part.region.anchor
            secondary = geometry.part.region.attachment
            damping = 6 N*s/m
        }
    }
    connector ground_spring {
        isotropic_ground_spring {
            target = geometry.part.region.anchor
            stiffness = 8 kN/m
        }
    }
    connector foundation {
        surface_normal_foundation {
            target = geometry.part.region.wall
            modulus = 10000 kN/m^3
        }
    }
    interface contact {
        contact.coulomb {
            primary = geometry.part.region.wall
            secondary = geometry.part.region.opposite
            coefficient = 0.3
        }
    }
    interface fluid_structure {
        fluid_structure.no_slip {
            primary = geometry.part.region.wall
            secondary = geometry.part.region.opposite
        }
    }
    interface heat_transfer {
        conjugate_heat_transfer.resistance {
            primary = geometry.part.region.wall
            secondary = geometry.part.region.opposite
            resistance = 0.01 m^2*K/W
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
    assert_eq!(model.boundaries.len(), 8);
    assert!(model.boundaries.iter().any(|boundary| matches!(
        &boundary.kind,
        runmat_analysis_core::AuthoredBoundaryKind::ThermalRadiation {
            emissivity: runmat_analysis_core::AuthoredBoundaryCoefficient::Constant(emissivity),
            ..
        } if emissivity.canonical_value() == 0.8
    )));
    assert_eq!(model.connectors.len(), 4);
    assert!(matches!(
        model.connectors[0].kind,
        runmat_analysis_core::AuthoredConnectorKind::AxialSpring { .. }
    ));
    assert!(matches!(
        model.connectors[1].kind,
        runmat_analysis_core::AuthoredConnectorKind::AxialDashpot { .. }
    ));
    assert!(matches!(
        model.connectors[2].kind,
        runmat_analysis_core::AuthoredConnectorKind::IsotropicGroundSpring { .. }
    ));
    assert!(matches!(
        model.connectors[3].kind,
        runmat_analysis_core::AuthoredConnectorKind::SurfaceNormalFoundation { .. }
    ));
    assert_eq!(model.interfaces.len(), 3);
    assert!(matches!(
        model.interfaces[0].kind,
        runmat_analysis_core::AuthoredInterfaceKind::ContactCoulomb { .. }
    ));
}

#[test]
fn connector_language_rejects_legacy_names_and_incompatible_topology() {
    let legacy = r#"
model assembly {
    connector obsolete {
        translational_spring {}
    }
}
"#;
    let program = FeaCompiler::new(&TestHost::default()).check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", legacy, 1),
        },
    });
    assert!(!program.is_valid());
    assert!(program.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == "fea.semantic.model"
            && diagnostic
                .message
                .contains("unsupported connector formulation `translational_spring`")
    }));

    let invalid_topology = r#"
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
model assembly {
    connector invalid_ground_support {
        isotropic_ground_spring {
            target = geometry.part.region.wall
            stiffness = 1 kN/m
        }
    }
}
"#;
    let program = FeaCompiler::new(&TestHost::default()).check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", invalid_topology, 1),
        },
    });
    assert!(!program.is_valid());
    assert!(program.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == "fea.semantic.model"
            && diagnostic.message.contains("incompatible topology")
    }));
}
