use super::*;

mod excitation_validation;
mod model_validation;

fn valid_model() -> AnalysisModel {
    AnalysisModel {
        model_id: AnalysisModelId("analysis_model_1".to_string()),
        geometry_id: "geo:model_1".to_string(),
        geometry_revision: 1,
        units: runmat_types::physical::CoordinateUnits::canonical(),
        frame: CoordinateFrame::Global,
        materials: vec![MaterialModel {
            fluid: None,
            material_id: "mat_steel".to_string(),
            name: "Steel".to_string(),
            provenance: MaterialProvenance::Authored,
            mass_density_kg_per_m3: Some(runmat_types::physical::Quantity::<
                runmat_types::physical::kind::MassDensity,
                _,
            >::new(7850.0)),
            elasticity: Some(MaterialElasticityModel::IsotropicLinear(
                MaterialMechanicalModel {
                    youngs_modulus_pa: runmat_types::physical::Quantity::<
                        runmat_types::physical::kind::YoungsModulus,
                        _,
                    >::new(200e9),
                    poisson_ratio: runmat_types::physical::Quantity::<
                        runmat_types::physical::kind::PoissonRatio,
                        _,
                    >::new(0.3),
                },
            )),
            damping: None,
            thermal: Some(MaterialThermalModel {
                reference_temperature_k: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Temperature,
                    _,
                >::new(293.15),
                modulus_temp_coeff_per_k: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::InverseTemperature,
                    _,
                >::new(-2.5e-4),
                conductivity_w_per_mk: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::ThermalConductivity,
                    _,
                >::new(45.0),
                specific_heat_j_per_kgk: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::SpecificHeat,
                    _,
                >::new(500.0),
                expansion_coefficient_per_k: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::ThermalExpansion,
                    _,
                >::new(1.2e-5),
            }),
            acoustic: None,
            electrical: None,
            plastic: None,
        }],
        material_assignments: Vec::new(),
        structural: None,
        thermal: None,
        thermo_mechanical: None,
        electro_thermal: None,
        electromagnetic: None,
        cfd: None,
        interfaces: Vec::new(),
        constraints: vec![AnalysisConstraint::Structural(StructuralConstraint {
            constraint_id: "bc_fixed_root".to_string(),
            frame: CoordinateFrame::Global,
            kind: StructuralConstraintKind::Fixed {
                target: RegionTarget::boundary("root"),
            },
        })],
        boundaries: Vec::new(),
        connectors: Vec::new(),
        signals: vec![PhysicalSignal::Force(Signal {
            signal_id: SignalId("tip_force_signal".to_string()),
            shape: SignalShape::Vector3,
            frame: Some(CoordinateFrame::Global),
            representation: SignalRepresentation::Constant(SignalValue::Vector3([
                Force::new(0.0),
                Force::new(-1000.0),
                Force::new(0.0),
            ])),
            precision: SignalPrecision::Real64,
            evidence: SignalApproximationEvidence {
                max_absolute_error: None,
                max_relative_error: None,
                derivative_max_absolute_error: None,
                sample_count: 1,
            },
            provenance: SignalProvenance::Authored,
        })],
        excitations: vec![ExcitationAction::Force(AnalysisExcitation {
            excitation_id: ExcitationId("tip_force".to_string()),
            target: ExcitationTarget::Region(RegionTarget::boundary("tip")),
            frame: CoordinateFrame::Global,
            direction: None,
            distribution: ExcitationDistribution::PreserveResultant,
            kinematics: ExcitationKinematics::FixedFrame,
            signal_id: SignalId("tip_force_signal".to_string()),
            active_step_ids: Vec::new(),
            combination_group: None,
            projection: ProjectionRequirement {
                preserve_resultant: true,
                preserve_moment: false,
                relative_tolerance: 1.0e-10,
            },
        })],
        excitation_cases: vec![ExcitationCase {
            case_id: ExcitationCaseId("primary_case".to_string()),
            action_ids: vec![ExcitationId("tip_force".to_string())],
            activation: ExcitationCaseActivation::Always,
        }],
        initial_conditions: Vec::new(),
        response_spectra: Vec::new(),
        random_excitations: Vec::new(),
        sequence: AnalysisSequence::from_steps(
            "analysis_sequence",
            vec![ProcedureStep::structural_static("step_static")],
        ),
    }
}
