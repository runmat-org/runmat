use super::*;
use crate::*;

fn target(dimension: EntityDimension) -> RegionTarget {
    RegionTarget::new(
        "surface",
        dimension,
        ExpectedTopology::GeometryEntity,
        TargetCardinality::OneOrMore,
    )
}

#[test]
fn pec_is_harmonic_electric_face_intent_not_static_or_magnetic_ground() {
    let harmonic = ElectromagneticDomain::Harmonic {
        reference_frequency: Frequency::new(60.0),
    };
    let pec = AnalysisConstraint::Electrical(ElectricalConstraint {
        constraint_id: "pec".into(),
        kind: ElectricalConstraintKind::ElectricPerfectConductor {
            target: target(EntityDimension::Face),
        },
    });
    validate_maxwell_constraints(Some(&harmonic), std::slice::from_ref(&pec)).unwrap();
    let bytes = serde_json::to_vec(&pec).unwrap();
    assert_eq!(
        serde_json::from_slice::<AnalysisConstraint>(&bytes).unwrap(),
        pec
    );
    assert!(String::from_utf8(bytes)
        .unwrap()
        .contains("electric_perfect_conductor"));
    for domain in [
        None,
        Some(ElectromagneticDomain::Static {
            equation: StaticElectricalEquation::Electrostatics,
        }),
        Some(ElectromagneticDomain::Static {
            equation: StaticElectricalEquation::SteadyConduction,
        }),
        Some(ElectromagneticDomain::Magnetostatic {
            formulation: MagnetostaticFormulation::MagneticVectorPotential,
        }),
    ] {
        assert!(validate_maxwell_constraints(domain.as_ref(), std::slice::from_ref(&pec)).is_err());
    }
    let mut volume = pec;
    if let AnalysisConstraint::Electrical(value) = &mut volume {
        value.kind = ElectricalConstraintKind::ElectricPerfectConductor {
            target: target(EntityDimension::Volume),
        };
    }
    assert!(validate_maxwell_constraints(Some(&harmonic), &[volume]).is_err());
}

#[test]
fn harmonic_admits_grounds_and_terminals_but_not_gauges() {
    let domain = ElectromagneticDomain::Harmonic {
        reference_frequency: Frequency::new(60.0),
    };
    for constraint in [
        AnalysisConstraint::Electrical(ElectricalConstraint {
            constraint_id: "terminal".into(),
            kind: ElectricalConstraintKind::PotentialReference {
                target: target(EntityDimension::Face),
            },
        }),
        AnalysisConstraint::Magnetic(MagneticConstraint {
            constraint_id: "far_field".into(),
            kind: MagneticConstraintKind::VectorPotentialGround {
                target: target(EntityDimension::Face),
            },
        }),
    ] {
        assert!(validate_maxwell_constraints(Some(&domain), &[constraint]).is_ok());
    }
    let gauge = AnalysisConstraint::Magnetic(MagneticConstraint {
        constraint_id: "gauge".into(),
        kind: MagneticConstraintKind::CoulombGauge {
            target: target(EntityDimension::Volume),
        },
    });
    assert!(validate_maxwell_constraints(Some(&domain), &[gauge]).is_err());
}
