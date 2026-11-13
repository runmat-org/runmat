//! Domain declarations are explicit model intent, shared by root and included models.
use super::AnalysisModelRealizationError;
use crate::{AuthoredModel, ElectromagneticDomain, ThermalDomain, ThermoMechanicalDomain};

type RealizedDomains = (
    Option<ThermalDomain>,
    Option<ThermoMechanicalDomain>,
    Option<ElectromagneticDomain>,
);

pub(super) fn realize(
    models: &[&AuthoredModel],
) -> Result<RealizedDomains, AnalysisModelRealizationError> {
    let mut thermal = coalesce(
        models.iter().filter_map(|model| model.thermal.as_ref()),
        "thermal",
    )?;
    let mut thermo_mechanical = coalesce(
        models
            .iter()
            .filter_map(|model| model.thermo_mechanical.as_ref()),
        "thermo_mechanical",
    )?;
    // Equal authored zero values must not choose distinct canonical bytes by include order.
    // Preserve authored source intent; normalize only the realized physical reference.
    for reference in thermal
        .iter_mut()
        .map(|domain| &mut domain.reference_temperature)
        .chain(
            thermo_mechanical
                .iter_mut()
                .map(|domain| &mut domain.reference_temperature),
        )
    {
        if reference.canonical_value() == 0.0 {
            *reference = runmat_types::physical::Quantity::new(0.0);
        }
    }
    let electromagnetic = coalesce(
        models
            .iter()
            .filter_map(|model| model.electromagnetic.as_ref()),
        "electromagnetic",
    )?;
    Ok((thermal, thermo_mechanical, electromagnetic))
}

fn coalesce<'a, T: Clone + PartialEq + 'a>(
    mut declarations: impl Iterator<Item = &'a T>,
    domain: &'static str,
) -> Result<Option<T>, AnalysisModelRealizationError> {
    let Some(first) = declarations.next() else {
        return Ok(None);
    };
    if declarations.any(|other| other != first) {
        return Err(AnalysisModelRealizationError::ConflictingDomain(domain));
    }
    Ok(Some(first.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Temperature;

    #[test]
    fn signed_zero_references_realize_identically_in_either_include_order() {
        let mut positive = authored("positive");
        positive.thermal = Some(ThermalDomain {
            reference_temperature: Temperature::new(0.0),
        });
        positive.thermo_mechanical = Some(ThermoMechanicalDomain {
            reference_temperature: Temperature::new(0.0),
        });
        let mut negative = positive.clone();
        negative.model_id = "negative".into();
        negative.thermal.as_mut().unwrap().reference_temperature =
            runmat_types::physical::Quantity::new(-0.0);
        negative
            .thermo_mechanical
            .as_mut()
            .unwrap()
            .reference_temperature = runmat_types::physical::Quantity::new(-0.0);
        let forward = realize(&[&positive, &negative]).unwrap();
        let reverse = realize(&[&negative, &positive]).unwrap();
        assert_eq!(
            serde_json::to_vec(&forward).unwrap(),
            serde_json::to_vec(&reverse).unwrap()
        );
        assert!(!forward
            .0
            .unwrap()
            .reference_temperature
            .canonical_value()
            .is_sign_negative());
        assert!(!reverse
            .1
            .unwrap()
            .reference_temperature
            .canonical_value()
            .is_sign_negative());
        assert!(negative
            .thermal
            .unwrap()
            .reference_temperature
            .canonical_value()
            .is_sign_negative());
    }

    fn authored(id: &str) -> AuthoredModel {
        AuthoredModel {
            module_path: "domains.fea".into(),
            model_id: id.into(),
            thermal: None,
            thermo_mechanical: None,
            electromagnetic: None,
            inputs: vec![],
            includes: vec![],
            sections: vec![],
            layups: vec![],
            orientations: vec![],
            material_assignments: vec![],
            property_assignments: vec![],
            constraints: vec![],
            boundaries: vec![],
            connectors: vec![],
            interfaces: vec![],
            signals: vec![],
            parameterized_signals: vec![],
            excitations: vec![],
            excitation_cases: vec![],
            response_spectra: vec![],
            random_excitations: vec![],
            initial_conditions: vec![],
        }
    }

    #[test]
    fn root_and_included_models_preserve_both_explicit_domains() {
        let mut included = authored("included");
        included.thermal = Some(ThermalDomain {
            reference_temperature: Temperature::new(293.15),
        });
        let mut root = authored("root");
        root.thermal = included.thermal.clone();
        root.thermo_mechanical = Some(ThermoMechanicalDomain {
            reference_temperature: Temperature::new(303.15),
        });
        assert_eq!(
            realize(&[&included, &root]).unwrap(),
            (root.thermal.clone(), root.thermo_mechanical.clone(), None)
        );
        root.thermal = None;
        assert_eq!(realize(&[&included, &root]).unwrap().0, included.thermal);
        root.thermal = Some(ThermalDomain {
            reference_temperature: Temperature::new(273.15),
        });
        assert_eq!(
            realize(&[&included, &root]),
            Err(AnalysisModelRealizationError::ConflictingDomain("thermal"))
        );
        root.thermal = None;
        included.thermo_mechanical = Some(ThermoMechanicalDomain {
            reference_temperature: Temperature::new(273.15),
        });
        assert_eq!(
            realize(&[&included, &root]),
            Err(AnalysisModelRealizationError::ConflictingDomain(
                "thermo_mechanical"
            ))
        );
    }

    #[test]
    fn declarations_coalesce_without_defaults_and_conflicts_are_order_independent() {
        let a = ThermalDomain {
            reference_temperature: Temperature::new(293.15),
        };
        let b = ThermalDomain {
            reference_temperature: Temperature::new(303.15),
        };
        assert_eq!(
            coalesce([&a, &a].into_iter(), "thermal").unwrap(),
            Some(a.clone())
        );
        assert_eq!(
            coalesce::<ThermalDomain>([].into_iter(), "thermal").unwrap(),
            None
        );
        let expected = Err(AnalysisModelRealizationError::ConflictingDomain("thermal"));
        assert_eq!(coalesce([&a, &b].into_iter(), "thermal"), expected);
        assert_eq!(coalesce([&b, &a].into_iter(), "thermal"), expected);
    }

    #[test]
    fn explicit_electrical_equations_coalesce_and_conflicts_fail_closed() {
        use crate::StaticElectricalEquation;
        let mut a = authored("a");
        let mut b = authored("b");
        assert_eq!(realize(&[&a, &b]).unwrap().2, None);
        a.electromagnetic = Some(ElectromagneticDomain::Static {
            equation: StaticElectricalEquation::Electrostatics,
        });
        a.validate().unwrap();
        assert_eq!(realize(&[&a, &b]).unwrap().2, a.electromagnetic);
        b.electromagnetic = a.electromagnetic.clone();
        assert_eq!(realize(&[&a, &b]).unwrap(), realize(&[&b, &a]).unwrap());
        b.electromagnetic = Some(ElectromagneticDomain::Static {
            equation: StaticElectricalEquation::SteadyConduction,
        });
        b.validate().unwrap();
        for models in [[&a, &b], [&b, &a]] {
            assert_eq!(
                realize(&models),
                Err(AnalysisModelRealizationError::ConflictingDomain(
                    "electromagnetic"
                ))
            );
        }
        b.electromagnetic = Some(ElectromagneticDomain::Harmonic {
            reference_frequency: crate::Frequency::new(60.0),
        });
        b.validate().unwrap();
        for models in [[&a, &b], [&b, &a]] {
            assert_eq!(
                realize(&models),
                Err(AnalysisModelRealizationError::ConflictingDomain(
                    "electromagnetic"
                ))
            );
        }
        for hertz in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            b.electromagnetic = Some(ElectromagneticDomain::Harmonic {
                reference_frequency: crate::Frequency::new(hertz),
            });
            assert!(b.validate().is_err());
        }
        b.electromagnetic = Some(ElectromagneticDomain::Magnetostatic {
            formulation: crate::MagnetostaticFormulation::MagneticVectorPotential,
        });
        b.validate().unwrap();
        for models in [[&a, &b], [&b, &a]] {
            assert_eq!(
                realize(&models),
                Err(AnalysisModelRealizationError::ConflictingDomain(
                    "electromagnetic"
                ))
            );
        }
        a.electromagnetic = b.electromagnetic.clone();
        assert_eq!(realize(&[&a, &b]).unwrap().2, b.electromagnetic);
    }
}
