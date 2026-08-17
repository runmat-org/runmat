use super::*;
use crate::model::exact_topology_tests::{model, topology};

fn range(start: f64, end: f64) -> ParameterRangeV2 {
    ParameterRangeV2 { start, end }
}

fn registry() -> ExactEvaluatorRegistryV2 {
    ExactEvaluatorRegistryV2 {
        schema_version: EXACT_EVALUATOR_REGISTRY_SCHEMA_VERSION,
        kernel_abi: "occt-v1".into(),
        curves: vec![ExactCurveEvaluatorRecordV2 {
            id: CurveEvaluatorIdV2::new("curve:1").unwrap(),
            implementation: ExactCurveImplementationV2::Portable {
                definition: ExactCurveDefinitionV2::Circle {
                    center_m: [0.0, 0.0, 0.0],
                    x_axis: [1.0, 0.0, 0.0],
                    y_axis: [0.0, 1.0, 0.0],
                    radius_m: 1.0,
                    domain: range(0.0, std::f64::consts::TAU),
                },
            },
        }],
        pcurves: vec![ExactPcurveEvaluatorRecordV2 {
            id: PcurveEvaluatorIdV2::new("pcurve:1").unwrap(),
            implementation: ExactPcurveImplementationV2::Portable {
                definition: ExactPcurveDefinitionV2::Circle {
                    center_uv: [0.0, 0.0],
                    x_axis_uv: [1.0, 0.0],
                    y_axis_uv: [0.0, 1.0],
                    radius_uv: 1.0,
                    domain: range(0.0, std::f64::consts::TAU),
                },
            },
        }],
        surfaces: vec![ExactSurfaceEvaluatorRecordV2 {
            id: SurfaceEvaluatorIdV2::new("surface:1").unwrap(),
            implementation: ExactSurfaceImplementationV2::Portable {
                definition: ExactSurfaceDefinitionV2::Plane {
                    origin_m: [0.0, 0.0, 0.0],
                    u_axis_m_per_parameter: [1.0, 0.0, 0.0],
                    v_axis_m_per_parameter: [0.0, 1.0, 0.0],
                    domains: [range(-2.0, 2.0), range(-2.0, 2.0)],
                },
            },
        }],
        trim_classifiers: vec![ExactTrimClassifierRecordV2 {
            id: TrimClassifierIdV2::new("trim:1").unwrap(),
            implementation: ExactTrimClassifierImplementationV2::OrientedPcurveWinding,
        }],
        mass_properties: vec![ExactMassPropertiesRecordV2 {
            id: MassPropertiesEvaluatorIdV2::new("mass:1").unwrap(),
            implementation: ExactMassPropertiesImplementationV2::KernelValidated {
                properties: BodyMassPropertiesV2 {
                    volume_m3: 1.0,
                    surface_area_m2: 6.0,
                    centroid_m: [0.5, 0.5, 0.5],
                    inertia_about_centroid_m5: [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 0.0, 0.0, 0.0],
                },
                validation_digest: [9; 32],
            },
        }],
    }
}

#[test]
fn exact_registry_fully_binds_topology_and_round_trips() {
    let registry = registry();
    registry.validate_against(&topology(), &model()).unwrap();
    let json = serde_json::to_string(&registry).unwrap();
    let decoded: ExactEvaluatorRegistryV2 = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded, registry);
}

#[test]
fn evaluator_inventory_is_exact_canonical_and_kernel_bound() {
    let mut missing = registry();
    missing.pcurves.clear();
    assert!(missing.validate_against(&topology(), &model()).is_err());

    let mut extra = registry();
    extra.curves.push(ExactCurveEvaluatorRecordV2 {
        id: CurveEvaluatorIdV2::new("curve:extra").unwrap(),
        implementation: extra.curves[0].implementation.clone(),
    });
    assert!(extra.validate_against(&topology(), &model()).is_err());

    let mut noncanonical = registry();
    noncanonical.curves.push(ExactCurveEvaluatorRecordV2 {
        id: CurveEvaluatorIdV2::new("curve:0").unwrap(),
        implementation: noncanonical.curves[0].implementation.clone(),
    });
    let error = noncanonical
        .validate_against(&topology(), &model())
        .unwrap_err();
    assert_eq!(error.field, "curve evaluators");

    let mut wrong_abi = registry();
    wrong_abi.kernel_abi = "other-kernel".into();
    assert!(wrong_abi.validate_against(&topology(), &model()).is_err());
}

#[test]
fn portable_nurbs_curves_pcurves_and_surfaces_are_admitted_without_samples() {
    let mut topology = topology();
    topology.edges[0].is_periodic = false;
    let mut registry = registry();
    registry.curves[0].implementation = ExactCurveImplementationV2::Portable {
        definition: ExactCurveDefinitionV2::Nurbs {
            definition: NurbsCurve3V2 {
                degree: 2,
                knots: vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                control_points_m: vec![[0.0, 0.0, 0.0], [0.5, 1.0, 0.0], [1.0, 0.0, 0.0]],
                weights: vec![1.0, 1.0, 1.0],
                domain: range(0.0, 1.0),
                periodic: false,
            },
        },
    };
    registry.pcurves[0].implementation = ExactPcurveImplementationV2::Portable {
        definition: ExactPcurveDefinitionV2::Nurbs {
            definition: NurbsCurve2V2 {
                degree: 1,
                knots: vec![0.0, 0.0, 1.0, 1.0],
                control_points_uv: vec![[0.0, 0.0], [1.0, 0.0]],
                weights: vec![1.0, 1.0],
                domain: range(0.0, 1.0),
                periodic: false,
            },
        },
    };
    registry.surfaces[0].implementation = ExactSurfaceImplementationV2::Portable {
        definition: ExactSurfaceDefinitionV2::Nurbs {
            definition: NurbsSurface3V2 {
                u_degree: 1,
                v_degree: 1,
                u_knots: vec![0.0, 0.0, 1.0, 1.0],
                v_knots: vec![0.0, 0.0, 1.0, 1.0],
                u_control_count: 2,
                v_control_count: 2,
                control_points_m: vec![
                    [0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                ],
                weights: vec![1.0; 4],
                domains: [range(0.0, 1.0), range(0.0, 1.0)],
                periodic_u: false,
                periodic_v: false,
            },
        },
    };
    registry.validate_against(&topology, &model()).unwrap();

    let ExactCurveImplementationV2::Portable { definition } =
        &mut registry.curves[0].implementation
    else {
        unreachable!()
    };
    let ExactCurveDefinitionV2::Nurbs { definition } = definition else {
        unreachable!()
    };
    definition.knots[3] = -1.0;
    assert!(registry.validate_against(&topology, &model()).is_err());
}

#[test]
fn portable_geometry_and_topology_flags_cannot_disagree() {
    let mut curve_mismatch = topology();
    curve_mismatch.edges[0].is_periodic = false;
    assert!(registry()
        .validate_against(&curve_mismatch, &model())
        .is_err());

    let mut surface_mismatch = topology();
    surface_mismatch.faces[0].periodic_u = true;
    assert!(registry()
        .validate_against(&surface_mismatch, &model())
        .is_err());

    let mut domain_mismatch = registry();
    let ExactPcurveImplementationV2::Portable {
        definition: ExactPcurveDefinitionV2::Circle { domain, .. },
    } = &mut domain_mismatch.pcurves[0].implementation
    else {
        unreachable!()
    };
    domain.end = 1.0;
    let error = domain_mismatch
        .validate_against(&topology(), &model())
        .unwrap_err();
    assert_eq!(error.field, "coedge evaluator domain");
}

#[test]
fn invalid_kernel_and_mass_evidence_fail_closed() {
    let mut kernel = registry();
    kernel.curves[0].implementation = ExactCurveImplementationV2::Kernel {
        reference: KernelEvaluatorRefV2 {
            entity_token: "edge:1".into(),
            representation_digest: [0; 32],
        },
    };
    assert!(kernel.validate_against(&topology(), &model()).is_err());

    let mut mass = registry();
    let ExactMassPropertiesImplementationV2::KernelValidated {
        properties,
        validation_digest,
    } = &mut mass.mass_properties[0].implementation
    else {
        unreachable!()
    };
    properties.volume_m3 = f64::NAN;
    *validation_digest = [0; 32];
    assert!(mass.validate_against(&topology(), &model()).is_err());
}
