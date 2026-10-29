use runmat_analysis_core::{
    Acceleration, AccelerationSignalId, AnalysisConnector, AnalysisConstraint, AnalysisExcitation,
    AnalysisModel, AnalysisModelId, AnalysisSequence, BeamElementModel, BeamSectionModel,
    BodyForceDensity, ConformalSolidAttachment, ConnectorKind, CoordinateFrame, Direction,
    Displacement, EvidenceConfidence, ExcitationAction, ExcitationCase, ExcitationCaseActivation,
    ExcitationCaseId, ExcitationDistribution, ExcitationId, ExcitationKinematics, ExcitationTarget,
    Force, HomogeneousShellReferenceSurface, HomogeneousShellSectionModel, MaterialAssignment,
    MaterialElasticityModel, MaterialMechanicalModel, MaterialModel, MaterialThermalModel,
    MechanicalDampingModel, Moment, PhysicalSignal, PrescribedStructuralDofs,
    PrescribedStructuralMotion, PrescribedValue, Pressure, ProcedureStep, ProjectionRequirement,
    PropertyRegionTarget, RayleighMassCoefficient, RayleighStiffnessCoefficient, RegionTarget,
    Rotation, RotationDofValue, ShellElementModel, ShellSectionModel, Signal,
    SignalApproximationEvidence, SignalExtrapolation, SignalId, SignalInterpolation, SignalKnot,
    SignalOrdinate, SignalPrecision, SignalProvenance, SignalRepresentation, SignalShape,
    SignalValue, Stiffness, StructuralConstraint, StructuralConstraintKind, StructuralElement,
    StructuralElementKind, StructuralKinematicHistory, StructuralModel,
    StructuralMotionReconstruction, StructuralNode, StructuralRotationDof,
    StructuralTranslationDof, TabulatedSignal, TabulatedSignalDomain, TranslationDofMotion,
    TranslationDofValue, Velocity,
};
#[cfg(test)]
use runmat_analysis_core::{ElectricPotential, Position, ReferencePoint, Temperature, Traction};
use runmat_geometry_core::{PersistentEntityId, PersistentEntityKind};

#[derive(Debug, Clone)]
pub(crate) struct FixtureExcitation {
    pub id: String,
    pub region_id: String,
    pub kind: FixtureExcitationKind,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum FixtureExcitationKind {
    Force([f64; 3]),
    Moment([f64; 3]),
    #[cfg(test)]
    Traction([f64; 3]),
    Pressure(f64),
    BodyForce([f64; 3]),
    #[cfg(test)]
    Wrench {
        force: [f64; 3],
        moment: [f64; 3],
        point_m: [f64; 3],
    },
}

pub(crate) fn replace_fixture_excitations(
    model: &mut AnalysisModel,
    specifications: Vec<FixtureExcitation>,
) {
    model.signals.clear();
    model.excitations.clear();
    model.excitation_cases.clear();
    for specification in specifications {
        append_fixture_excitation(model, specification);
    }
}

pub fn mixed_solid_beam_model(attached_coordinates_m: [f64; 3]) -> AnalysisModel {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.material_assignments.push(MaterialAssignment {
        target: PropertyRegionTarget::beam(runmat_meshing_core::fixtures::fixture_edge("span")),
        expected_material_id: "mat_steel".into(),
        assigned_material_id: "mat_steel".into(),
        confidence: EvidenceConfidence::Verified,
    });
    model.structural = Some(StructuralModel {
        nodes: vec![
            StructuralNode {
                node_id: 101,
                coordinates_m: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new(attached_coordinates_m),
                solid_attachment: Some(ConformalSolidAttachment::GeometryVertex {
                    entity: PersistentEntityId::new(
                        PersistentEntityKind::Vertex,
                        "beam_root".into(),
                        vec!["root".into()],
                    ),
                }),
            },
            StructuralNode {
                node_id: 102,
                coordinates_m: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new([
                    attached_coordinates_m[0] + 1.0,
                    attached_coordinates_m[1],
                    attached_coordinates_m[2],
                ]),
                solid_attachment: None,
            },
        ],
        elements: vec![StructuralElement {
            element_id: "beam_1".into(),
            region_id: runmat_meshing_core::fixtures::fixture_edge("span"),
            kind: StructuralElementKind::Beam(BeamElementModel {
                node_ids: [101, 102],
                section_id: "rect".into(),
                reference_axis: [0.0, 1.0, 0.0],
            }),
        }],
        beam_sections: vec![BeamSectionModel {
            section_id: "rect".into(),
            area_m2: runmat_types::physical::Quantity::<runmat_types::physical::kind::Area, _>::new(
                1.0e-4,
            ),
            iy_m4: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::SecondMomentOfArea,
                _,
            >::new(1.0e-9),
            iz_m4: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::SecondMomentOfArea,
                _,
            >::new(1.0e-9),
            torsion_j_m4: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::SecondMomentOfArea,
                _,
            >::new(1.0e-9),
            outer_fiber_y_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Length,
                _,
            >::new(0.01),
            outer_fiber_z_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Length,
                _,
            >::new(0.01),
            torsion_outer_radius_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Length,
                _,
            >::new(0.01),
        }],
        shell_sections: Vec::new(),
    });
    model.constraints = vec![
        fixed_constraint("fixed_root", "root"),
        AnalysisConstraint::Structural(StructuralConstraint {
            constraint_id: "beam_root_rotation".into(),
            frame: CoordinateFrame::Global,
            kind: StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
                target: RegionTarget::point("node:101"),
                translations: Vec::new(),
                rotations: [
                    StructuralRotationDof::Rx,
                    StructuralRotationDof::Ry,
                    StructuralRotationDof::Rz,
                ]
                .into_iter()
                .map(|dof| RotationDofValue {
                    dof,
                    value: PrescribedValue::Constant(Rotation::new(0.0)),
                })
                .collect(),
            }),
        }),
    ];
    replace_fixture_excitations(
        &mut model,
        vec![
            FixtureExcitation {
                id: "beam_tip_force".into(),
                region_id: runmat_meshing_core::fixtures::fixture_edge("span"),
                kind: FixtureExcitationKind::Force([1_000.0, 0.0, 0.0]),
            },
            FixtureExcitation {
                id: "solid_face_force".into(),
                region_id: "solid_load".into(),
                kind: FixtureExcitationKind::Force([0.0, 250.0, 0.0]),
            },
        ],
    );
    model.connectors = vec![AnalysisConnector {
        connector_id: "mixed_tip_ground_spring".into(),
        kind: ConnectorKind::IsotropicGroundSpring {
            target: RegionTarget::point("node:102"),
            stiffness: Stiffness::new(2_000.0),
        },
    }];
    model
}

pub fn mixed_solid_beam_shell_model(attached_coordinates_m: [f64; 3]) -> AnalysisModel {
    let mut model = mixed_solid_beam_model(attached_coordinates_m);
    model.material_assignments.push(MaterialAssignment {
        target: PropertyRegionTarget::shell(runmat_meshing_core::fixtures::fixture_face("skin")),
        expected_material_id: "mat_steel".into(),
        assigned_material_id: "mat_steel".into(),
        confidence: EvidenceConfidence::Verified,
    });
    let structural = model
        .structural
        .as_mut()
        .expect("mixed solid-beam fixture owns explicit topology");
    structural.nodes.extend([
        StructuralNode {
            node_id: 103,
            coordinates_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new([
                attached_coordinates_m[0],
                attached_coordinates_m[1] + 1.0,
                attached_coordinates_m[2],
            ]),
            solid_attachment: None,
        },
        StructuralNode {
            node_id: 104,
            coordinates_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new([
                attached_coordinates_m[0],
                attached_coordinates_m[1],
                attached_coordinates_m[2] + 1.0,
            ]),
            solid_attachment: None,
        },
    ]);
    structural.elements.push(StructuralElement {
        element_id: "shell_1".into(),
        region_id: runmat_meshing_core::fixtures::fixture_face("skin"),
        kind: StructuralElementKind::Shell(ShellElementModel {
            node_ids: [101, 103, 104],
            section_id: "skin_2mm".into(),
            reference_axis: [0.0, 1.0, 0.0],
        }),
    });
    structural
        .shell_sections
        .push(ShellSectionModel::Homogeneous(
            HomogeneousShellSectionModel {
                section_id: "skin_2mm".into(),
                thickness_m: runmat_types::physical::Length::new(0.002),
                reference_surface: HomogeneousShellReferenceSurface::MidSurface,
                shear_correction: 5.0 / 6.0,
            },
        ));
    append_fixture_excitation(
        &mut model,
        FixtureExcitation {
            id: "shell_skin_force".into(),
            region_id: "skin".into(),
            kind: FixtureExcitationKind::Force([125.0, 0.0, 0.0]),
        },
    );
    model
}

#[cfg(test)]
pub(crate) fn append_fixture_temperature_field(
    model: &mut AnalysisModel,
    id: &str,
    temperature_k: f64,
    target: ExcitationTarget,
) {
    append_fixture_scalar_action(
        model,
        id,
        target,
        Temperature::new(temperature_k),
        PhysicalSignal::Temperature,
        ExcitationAction::TemperatureField,
    );
}

#[cfg(test)]
pub(crate) fn append_fixture_terminal_voltage(
    model: &mut AnalysisModel,
    id: &str,
    voltage_v: f64,
    target: ExcitationTarget,
) {
    append_fixture_scalar_action(
        model,
        id,
        target,
        ElectricPotential::new(voltage_v),
        PhysicalSignal::ElectricPotential,
        ExcitationAction::TerminalVoltage,
    );
}

#[cfg(test)]
fn append_fixture_scalar_action<Q>(
    model: &mut AnalysisModel,
    id: &str,
    target: ExcitationTarget,
    value: Q,
    signal: impl FnOnce(Signal<Q>) -> PhysicalSignal,
    action: impl FnOnce(AnalysisExcitation) -> ExcitationAction,
) {
    let signal_id = SignalId(format!("{id}.signal"));
    let case_id = ExcitationCaseId(format!("{id}.case"));
    model.signals.push(signal(Signal {
        signal_id: signal_id.clone(),
        shape: SignalShape::Scalar,
        frame: None,
        representation: SignalRepresentation::Constant(SignalValue::Scalar(value)),
        precision: SignalPrecision::Real64,
        evidence: exact_signal_evidence(),
        provenance: SignalProvenance::Authored,
    }));
    model.excitations.push(action(fixture_action(
        id,
        target,
        signal_id,
        ExcitationDistribution::Concentrated,
    )));
    model.excitation_cases.push(ExcitationCase {
        case_id,
        action_ids: vec![ExcitationId(id.to_string())],
        activation: ExcitationCaseActivation::Always,
    });
}

fn append_fixture_excitation(model: &mut AnalysisModel, specification: FixtureExcitation) {
    let case_id = ExcitationCaseId(format!("{}.case", specification.id));
    let target = match specification.kind {
        FixtureExcitationKind::Pressure(_) => ExcitationTarget::Region(RegionTarget::boundary(
            fixture_target_selection(&specification.region_id),
        )),
        #[cfg(test)]
        FixtureExcitationKind::Traction(_) => ExcitationTarget::Region(RegionTarget::boundary(
            fixture_target_selection(&specification.region_id),
        )),
        FixtureExcitationKind::BodyForce(_) => ExcitationTarget::Region(RegionTarget::volume(
            fixture_volume_selection(&specification.region_id),
        )),
        _ => ExcitationTarget::Region(RegionTarget::authored(fixture_target_selection(
            &specification.region_id,
        ))),
    };
    let mut action_ids = Vec::new();
    match specification.kind {
        FixtureExcitationKind::Force(value) => append_vector_excitation(
            model,
            &specification.id,
            target,
            value,
            VectorExcitationKind::Force,
        ),
        FixtureExcitationKind::Moment(value) => append_vector_excitation(
            model,
            &specification.id,
            target,
            value,
            VectorExcitationKind::Moment,
        ),
        #[cfg(test)]
        FixtureExcitationKind::Traction(value) => append_vector_excitation(
            model,
            &specification.id,
            target,
            value,
            VectorExcitationKind::Traction,
        ),
        FixtureExcitationKind::Pressure(value) => append_scalar_excitation(
            model,
            &specification.id,
            target,
            Pressure::new(value),
            PhysicalSignal::Pressure,
            ExcitationAction::Pressure,
        ),
        FixtureExcitationKind::BodyForce(value) => append_vector_excitation(
            model,
            &specification.id,
            target,
            value,
            VectorExcitationKind::BodyForce,
        ),
        #[cfg(test)]
        FixtureExcitationKind::Wrench {
            force,
            moment,
            point_m,
        } => {
            let target = ExcitationTarget::ReferencePoint(ReferencePoint {
                reference_point_id: specification.region_id,
                position: Position::new(point_m),
            });
            let force_id = specification.id.clone();
            let moment_id = format!("{}.moment", specification.id);
            append_vector_excitation(
                model,
                &force_id,
                target.clone(),
                force,
                VectorExcitationKind::Force,
            );
            append_vector_excitation(
                model,
                &moment_id,
                target,
                moment,
                VectorExcitationKind::Moment,
            );
            action_ids.extend([ExcitationId(force_id), ExcitationId(moment_id)]);
        }
    }
    if action_ids.is_empty() {
        action_ids.push(ExcitationId(specification.id));
    }
    model.excitation_cases.push(ExcitationCase {
        case_id,
        action_ids,
        activation: ExcitationCaseActivation::Always,
    });
}

#[derive(Debug, Clone, Copy)]
enum VectorExcitationKind {
    Force,
    Moment,
    #[cfg(test)]
    Traction,
    BodyForce,
}

fn append_vector_excitation(
    model: &mut AnalysisModel,
    id: &str,
    target: ExcitationTarget,
    values: [f64; 3],
    kind: VectorExcitationKind,
) {
    let signal_id = SignalId(format!("{id}.signal"));
    let (signal, distribution) = match kind {
        VectorExcitationKind::Force => (
            PhysicalSignal::Force(fixture_vector_signal(signal_id.clone(), values)),
            ExcitationDistribution::PreserveResultant,
        ),
        VectorExcitationKind::Moment => (
            PhysicalSignal::Moment(fixture_vector_signal(signal_id.clone(), values)),
            ExcitationDistribution::PreserveResultant,
        ),
        #[cfg(test)]
        VectorExcitationKind::Traction => (
            PhysicalSignal::Traction(fixture_vector_signal(signal_id.clone(), values)),
            ExcitationDistribution::IntensityPerArea,
        ),
        VectorExcitationKind::BodyForce => (
            PhysicalSignal::BodyForceDensity(fixture_vector_signal(signal_id.clone(), values)),
            ExcitationDistribution::IntensityPerVolume,
        ),
    };
    model.signals.push(signal);
    let action = fixture_action(id, target, signal_id, distribution);
    model.excitations.push(match kind {
        VectorExcitationKind::Force => ExcitationAction::Force(action),
        VectorExcitationKind::Moment => ExcitationAction::Moment(action),
        #[cfg(test)]
        VectorExcitationKind::Traction => ExcitationAction::Traction(action),
        VectorExcitationKind::BodyForce => ExcitationAction::BodyForce(action),
    });
}

fn fixture_vector_signal<Q: Copy + FromFixtureComponent>(
    signal_id: SignalId,
    values: [f64; 3],
) -> Signal<Q> {
    Signal {
        signal_id: signal_id.clone(),
        shape: SignalShape::Vector3,
        frame: Some(CoordinateFrame::Global),
        representation: SignalRepresentation::Constant(SignalValue::Vector3(
            values.map(Q::from_component),
        )),
        precision: SignalPrecision::Real64,
        evidence: exact_signal_evidence(),
        provenance: SignalProvenance::Authored,
    }
}

fn append_scalar_excitation<Q>(
    model: &mut AnalysisModel,
    id: &str,
    target: ExcitationTarget,
    value: Q,
    signal_kind: impl FnOnce(Signal<Q>) -> PhysicalSignal,
    action_kind: impl FnOnce(AnalysisExcitation) -> ExcitationAction,
) {
    let signal_id = SignalId(format!("{id}.signal"));
    model.signals.push(signal_kind(Signal {
        signal_id: signal_id.clone(),
        shape: SignalShape::Scalar,
        frame: None,
        representation: SignalRepresentation::Constant(SignalValue::Scalar(value)),
        precision: SignalPrecision::Real64,
        evidence: exact_signal_evidence(),
        provenance: SignalProvenance::Authored,
    }));
    model.excitations.push(action_kind(fixture_action(
        id,
        target,
        signal_id,
        ExcitationDistribution::IntensityPerArea,
    )));
}

fn fixture_action(
    id: &str,
    target: ExcitationTarget,
    signal_id: SignalId,
    distribution: ExcitationDistribution,
) -> AnalysisExcitation {
    AnalysisExcitation {
        excitation_id: ExcitationId(id.to_string()),
        target,
        frame: CoordinateFrame::Global,
        direction: Some(Direction {
            vector: [1.0, 0.0, 0.0],
        }),
        distribution,
        kinematics: ExcitationKinematics::FixedFrame,
        signal_id,
        active_step_ids: Vec::new(),
        combination_group: None,
        projection: ProjectionRequirement {
            preserve_resultant: true,
            preserve_moment: true,
            relative_tolerance: 1.0e-12,
        },
    }
}

fn exact_signal_evidence() -> SignalApproximationEvidence {
    SignalApproximationEvidence {
        max_absolute_error: Some(0.0),
        max_relative_error: Some(0.0),
        derivative_max_absolute_error: None,
        sample_count: 1,
    }
}

trait FromFixtureComponent {
    fn from_component(value: f64) -> Self;
}

impl FromFixtureComponent for Force {
    fn from_component(value: f64) -> Self {
        Self::new(value)
    }
}

impl FromFixtureComponent for Moment {
    fn from_component(value: f64) -> Self {
        Self::new(value)
    }
}

#[cfg(test)]
impl FromFixtureComponent for Traction {
    fn from_component(value: f64) -> Self {
        Self::new(value)
    }
}

impl FromFixtureComponent for BodyForceDensity {
    fn from_component(value: f64) -> Self {
        Self::new(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixtureId {
    CantileverLinearStatic,
    CantileverLoadSweep,
    CantileverLargeLoadSweep,
    StructuralAxialBarReference,
    StructuralBeamBendingReference,
    StructuralBeamCantileverEndMomentReference,
    StructuralBeamTorsionReference,
    StructuralBeamForceAndMomentReference,
    StructuralShellPlateMomentReference,
    StructuralInvalidMomentWithoutRotationalDofs,
    ModalLarge,
    TransientLong,
    TransientShock,
    StructuralTransientTet4Reference,
    StructuralTransientTet4SupportMotionReference,
    NonlinearAssembly,
    NonlinearAssemblyStress,
    NonlinearSofteningBenchmark,
    NonlinearLoadPathMix,
    NonlinearContactFrictionlessReference,
    NonlinearContactFrictionlessReferenceComplex,
    NonlinearPlasticHardeningReference,
    NonlinearPlasticHardeningReferenceComplex,
    ThermoMechanicalKickoff,
    ThermoGradientBenign,
    ThermoGradientPathological,
    ThermoRampSmooth,
    ThermoShockOscillatory,
    ElectroThermalJouleBenign,
    ElectroThermalJoulePathological,
    MultiMaterialAssembly,
    MissingMaterials,
}

/// Canonical face selector for a fixture boundary name; `node:` point ids and canonical
/// selectors pass through unchanged.
pub(crate) fn fixture_target_selection(name: &str) -> String {
    runmat_meshing_core::fixtures::fixture_boundary_selection(name)
}

/// Volume selector for a fixture region name; canonical selectors pass through unchanged.
fn fixture_volume_selection(name: &str) -> String {
    if name.starts_with("persistent-entity:sha256:") {
        name.to_owned()
    } else {
        runmat_meshing_core::fixtures::fixture_region(name)
    }
}

fn fixed_constraint(constraint_id: &str, region_id: &str) -> AnalysisConstraint {
    AnalysisConstraint::Structural(StructuralConstraint {
        constraint_id: constraint_id.to_string(),
        frame: CoordinateFrame::Global,
        kind: StructuralConstraintKind::Fixed {
            target: RegionTarget::authored(fixture_target_selection(region_id)),
        },
    })
}

fn prescribed_zero_translations(constraint_id: &str, region_id: &str) -> AnalysisConstraint {
    AnalysisConstraint::Structural(StructuralConstraint {
        constraint_id: constraint_id.to_string(),
        frame: CoordinateFrame::Global,
        kind: StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
            target: RegionTarget::authored(fixture_target_selection(region_id)),
            translations: [
                StructuralTranslationDof::Ux,
                StructuralTranslationDof::Uy,
                StructuralTranslationDof::Uz,
            ]
            .into_iter()
            .map(|dof| TranslationDofValue {
                dof,
                value: PrescribedValue::Constant(Displacement::new(0.0)),
            })
            .collect(),
            rotations: Vec::new(),
        }),
    })
}

pub fn fixture_model(fixture: FixtureId) -> AnalysisModel {
    match fixture {
        FixtureId::CantileverLinearStatic => cantilever_linear_static(),
        FixtureId::CantileverLoadSweep => cantilever_load_sweep(),
        FixtureId::CantileverLargeLoadSweep => cantilever_large_load_sweep(),
        FixtureId::StructuralAxialBarReference => structural_axial_bar_reference(),
        FixtureId::StructuralBeamBendingReference => structural_beam_bending_reference(),
        FixtureId::StructuralBeamCantileverEndMomentReference => {
            structural_beam_cantilever_end_moment_reference()
        }
        FixtureId::StructuralBeamTorsionReference => structural_beam_torsion_reference(),
        FixtureId::StructuralBeamForceAndMomentReference => {
            structural_beam_force_and_moment_reference()
        }
        FixtureId::StructuralShellPlateMomentReference => structural_shell_plate_moment_reference(),
        FixtureId::StructuralInvalidMomentWithoutRotationalDofs => {
            structural_invalid_moment_without_rotational_dofs()
        }
        FixtureId::ModalLarge => modal_large_fixture(),
        FixtureId::TransientLong => transient_long_fixture(),
        FixtureId::TransientShock => transient_shock_fixture(),
        FixtureId::StructuralTransientTet4Reference => structural_transient_tet4_reference(),
        FixtureId::StructuralTransientTet4SupportMotionReference => {
            structural_transient_tet4_support_motion_reference()
        }
        FixtureId::NonlinearAssembly => nonlinear_assembly_fixture(),
        FixtureId::NonlinearAssemblyStress => nonlinear_assembly_stress_fixture(),
        FixtureId::NonlinearSofteningBenchmark => nonlinear_softening_benchmark_fixture(),
        FixtureId::NonlinearLoadPathMix => nonlinear_load_path_mix_fixture(),
        FixtureId::NonlinearContactFrictionlessReference => {
            nonlinear_contact_frictionless_reference_fixture()
        }
        FixtureId::NonlinearContactFrictionlessReferenceComplex => {
            nonlinear_contact_frictionless_reference_complex_fixture()
        }
        FixtureId::NonlinearPlasticHardeningReference => {
            nonlinear_plastic_hardening_reference_fixture()
        }
        FixtureId::NonlinearPlasticHardeningReferenceComplex => {
            nonlinear_plastic_hardening_reference_complex_fixture()
        }
        FixtureId::ThermoMechanicalKickoff => thermo_mechanical_kickoff_fixture(),
        FixtureId::ThermoGradientBenign => thermo_gradient_benign_fixture(),
        FixtureId::ThermoGradientPathological => thermo_gradient_pathological_fixture(),
        FixtureId::ThermoRampSmooth => thermo_ramp_smooth_fixture(),
        FixtureId::ThermoShockOscillatory => thermo_shock_oscillatory_fixture(),
        FixtureId::ElectroThermalJouleBenign => electro_thermal_joule_benign_fixture(),
        FixtureId::ElectroThermalJoulePathological => electro_thermal_joule_pathological_fixture(),
        FixtureId::MultiMaterialAssembly => multi_material_assembly(),
        FixtureId::MissingMaterials => missing_materials(),
    }
}

fn isotropic_material(
    material_id: &str,
    name: &str,
    youngs_modulus_pa: f64,
    poisson_ratio: f64,
    mass_density_kg_per_m3: f64,
    modulus_temp_coeff_per_k: f64,
) -> MaterialModel {
    MaterialModel {
        fluid: None,
        material_id: material_id.to_string(),
        name: name.to_string(),
        provenance: runmat_analysis_core::MaterialProvenance::Authored,
        mass_density_kg_per_m3: Some(runmat_types::physical::MassDensity::new(
            mass_density_kg_per_m3,
        )),
        elasticity: Some(MaterialElasticityModel::IsotropicLinear(
            MaterialMechanicalModel {
                youngs_modulus_pa: runmat_types::physical::YoungsModulus::new(youngs_modulus_pa),
                poisson_ratio: runmat_types::physical::PoissonRatio::new(poisson_ratio),
            },
        )),
        damping: None,
        thermal: Some(MaterialThermalModel {
            reference_temperature_k: runmat_types::physical::Temperature::new(293.15),
            modulus_temp_coeff_per_k: runmat_types::physical::InverseTemperature::new(
                modulus_temp_coeff_per_k,
            ),
            conductivity_w_per_mk: runmat_types::physical::ThermalConductivity::new(45.0),
            specific_heat_j_per_kgk: runmat_types::physical::SpecificHeat::new(500.0),
            expansion_coefficient_per_k: runmat_types::physical::ThermalExpansion::new(1.2e-5),
        }),
        acoustic: None,
        electrical: None,
        plastic: None,
    }
}

fn cantilever_linear_static() -> AnalysisModel {
    let mut model = AnalysisModel {
        model_id: AnalysisModelId("cantilever".to_string()),
        geometry_id: "geo:cantilever".to_string(),
        geometry_revision: 1,
        units: runmat_types::physical::CoordinateUnits::canonical(),
        frame: CoordinateFrame::Global,
        materials: vec![isotropic_material(
            "mat_steel",
            "Steel",
            200e9,
            0.3,
            7850.0,
            -2.5e-4,
        )],
        material_assignments: vec![MaterialAssignment {
            target: PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_region(
                "region",
            )),
            expected_material_id: "mat_steel".to_string(),
            assigned_material_id: "mat_steel".to_string(),
            confidence: EvidenceConfidence::Verified,
        }],
        structural: None,
        thermal: None,
        thermo_mechanical: None,
        electro_thermal: None,
        electromagnetic: None,
        cfd: None,
        interfaces: Vec::new(),
        constraints: vec![fixed_constraint("bc_root", "root")],
        boundaries: Vec::new(),
        connectors: Vec::new(),
        signals: Vec::new(),
        excitations: Vec::new(),
        excitation_cases: Vec::new(),
        initial_conditions: Vec::new(),
        response_spectra: Vec::new(),
        random_excitations: Vec::new(),
        sequence: AnalysisSequence::from_steps(
            "cantilever_sequence",
            vec![ProcedureStep::structural_static("static_1")],
        ),
    };
    replace_fixture_excitations(
        &mut model,
        vec![FixtureExcitation {
            id: "tip_load".into(),
            region_id: "tip".into(),
            kind: FixtureExcitationKind::Force([0.0, -1000.0, 0.0]),
        }],
    );
    model
}

fn cantilever_load_sweep() -> AnalysisModel {
    let mut model = cantilever_linear_static();
    model.model_id = AnalysisModelId("cantilever_load_sweep".to_string());
    let excitations = (0..128)
        .map(|i| {
            let scale = 1.0 + (i as f64) * 0.01;
            FixtureExcitation {
                id: format!("tip_load_{i}"),
                region_id: format!("tip_{i}"),
                kind: FixtureExcitationKind::Force([0.0, -1000.0 * scale, 0.0]),
            }
        })
        .collect();
    replace_fixture_excitations(&mut model, excitations);
    model
}

fn cantilever_large_load_sweep() -> AnalysisModel {
    let mut model = cantilever_linear_static();
    model.model_id = AnalysisModelId("cantilever_large_load_sweep".to_string());
    let excitations = (0..512)
        .map(|i| {
            let scale = 1.0 + (i as f64) * 0.005;
            FixtureExcitation {
                id: format!("tip_load_large_{i}"),
                region_id: format!("tip_large_{i}"),
                kind: FixtureExcitationKind::Force([0.0, -800.0 * scale, 0.0]),
            }
        })
        .collect();
    replace_fixture_excitations(&mut model, excitations);
    model
}

fn structural_axial_bar_reference() -> AnalysisModel {
    let mut model = cantilever_linear_static();
    model.model_id = AnalysisModelId("structural_axial_bar_reference".to_string());
    model.geometry_id = "geo:structural_axial_bar".to_string();
    let excitations = (0..12)
        .map(|i| FixtureExcitation {
            id: format!("axial_bar_tension_{i}"),
            region_id: format!("bar_station_{i}"),
            kind: FixtureExcitationKind::Force([2_000.0, 0.0, 0.0]),
        })
        .collect();
    replace_fixture_excitations(&mut model, excitations);
    model.material_assignments = vec![MaterialAssignment {
        target: PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_region(
            "bar_span",
        )),
        expected_material_id: "mat_steel".to_string(),
        assigned_material_id: "mat_steel".to_string(),
        confidence: EvidenceConfidence::Verified,
    }];
    model
}

fn structural_beam_bending_reference() -> AnalysisModel {
    let mut model = cantilever_linear_static();
    model.model_id = AnalysisModelId("structural_beam_bending_reference".to_string());
    model.geometry_id = "geo:structural_beam_bending".to_string();
    let excitations = (0..12)
        .map(|i| {
            let span_fraction = (i + 1) as f64 / 12.0;
            FixtureExcitation {
                id: format!("beam_bending_station_{i}"),
                region_id: format!("beam_station_{i}"),
                kind: FixtureExcitationKind::Force([0.0, -500.0 * span_fraction, 0.0]),
            }
        })
        .collect();
    replace_fixture_excitations(&mut model, excitations);
    model.material_assignments = vec![MaterialAssignment {
        target: PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_region(
            "beam_span",
        )),
        expected_material_id: "mat_steel".to_string(),
        assigned_material_id: "mat_steel".to_string(),
        confidence: EvidenceConfidence::Verified,
    }];
    model
}

fn structural_beam_cantilever_end_moment_reference() -> AnalysisModel {
    structural_beam_reference_model(
        "structural_beam_cantilever_end_moment_reference",
        "geo:structural_beam_end_moment",
        vec![FixtureExcitation {
            id: "tip_moment_z".into(),
            region_id: "node:2".into(),
            kind: FixtureExcitationKind::Moment([0.0, 0.0, 125.0]),
        }],
    )
}

fn structural_beam_torsion_reference() -> AnalysisModel {
    structural_beam_reference_model(
        "structural_beam_torsion_reference",
        "geo:structural_beam_torsion",
        vec![FixtureExcitation {
            id: "tip_torque_x".into(),
            region_id: "node:2".into(),
            kind: FixtureExcitationKind::Moment([80.0, 0.0, 0.0]),
        }],
    )
}

fn structural_beam_force_and_moment_reference() -> AnalysisModel {
    structural_beam_reference_model(
        "structural_beam_force_and_moment_reference",
        "geo:structural_beam_force_and_moment",
        vec![
            FixtureExcitation {
                id: "tip_force_y".into(),
                region_id: "node:2".into(),
                kind: FixtureExcitationKind::Force([0.0, 500.0, 0.0]),
            },
            FixtureExcitation {
                id: "tip_moment_z".into(),
                region_id: "node:2".into(),
                kind: FixtureExcitationKind::Moment([0.0, 0.0, 90.0]),
            },
        ],
    )
}

fn structural_invalid_moment_without_rotational_dofs() -> AnalysisModel {
    let mut model = cantilever_linear_static();
    model.model_id = AnalysisModelId("structural_invalid_moment_without_rotational_dofs".into());
    model.geometry_id = "geo:structural_invalid_moment_without_rotational_dofs".to_string();
    replace_fixture_excitations(
        &mut model,
        vec![FixtureExcitation {
            id: "solid_tip_moment".into(),
            region_id: "tip".into(),
            kind: FixtureExcitationKind::Moment([0.0, 0.0, 125.0]),
        }],
    );
    model
}

fn structural_shell_plate_moment_reference() -> AnalysisModel {
    let mut model = cantilever_linear_static();
    model.model_id = AnalysisModelId("structural_shell_plate_moment_reference".to_string());
    model.geometry_id = "geo:structural_shell_plate_moment".to_string();
    model.constraints = vec![fixed_constraint("fixed_corner", "node:1")];
    replace_fixture_excitations(
        &mut model,
        vec![FixtureExcitation {
            id: "free_corner_moment_y".into(),
            region_id: "node:3".into(),
            kind: FixtureExcitationKind::Moment([0.0, 10.0, 0.0]),
        }],
    );
    model.material_assignments = vec![MaterialAssignment {
        target: PropertyRegionTarget::shell(runmat_meshing_core::fixtures::fixture_face(
            "shell_panel",
        )),
        expected_material_id: "mat_steel".to_string(),
        assigned_material_id: "mat_steel".to_string(),
        confidence: EvidenceConfidence::Verified,
    }];
    model.structural = Some(StructuralModel {
        nodes: vec![
            StructuralNode {
                node_id: 1,
                coordinates_m: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new([0.0, 0.0, 0.0]),
                solid_attachment: None,
            },
            StructuralNode {
                node_id: 2,
                coordinates_m: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new([1.0, 0.0, 0.0]),
                solid_attachment: None,
            },
            StructuralNode {
                node_id: 3,
                coordinates_m: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new([0.0, 1.0, 0.0]),
                solid_attachment: None,
            },
        ],
        elements: vec![StructuralElement {
            element_id: "shell_1".to_string(),
            region_id: runmat_meshing_core::fixtures::fixture_face("shell_panel"),
            kind: StructuralElementKind::Shell(ShellElementModel {
                node_ids: [1, 2, 3],
                section_id: "panel_2mm".to_string(),
                reference_axis: [1.0, 0.0, 0.0],
            }),
        }],
        beam_sections: Vec::new(),
        shell_sections: vec![ShellSectionModel::Homogeneous(
            HomogeneousShellSectionModel {
                section_id: "panel_2mm".to_string(),
                thickness_m: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Length,
                    _,
                >::new(0.002),
                reference_surface: HomogeneousShellReferenceSurface::MidSurface,
                shear_correction: 5.0 / 6.0,
            },
        )],
    });
    model
}

fn structural_beam_reference_model(
    model_id: &str,
    geometry_id: &str,
    excitations: Vec<FixtureExcitation>,
) -> AnalysisModel {
    let mut model = cantilever_linear_static();
    model.model_id = AnalysisModelId(model_id.to_string());
    model.geometry_id = geometry_id.to_string();
    model.constraints = vec![fixed_constraint("fixed_root", "node:1")];
    replace_fixture_excitations(&mut model, excitations);
    model.material_assignments = vec![MaterialAssignment {
        target: PropertyRegionTarget::beam(runmat_meshing_core::fixtures::fixture_edge(
            "beam_span",
        )),
        expected_material_id: "mat_steel".to_string(),
        assigned_material_id: "mat_steel".to_string(),
        confidence: EvidenceConfidence::Verified,
    }];
    model.structural = Some(StructuralModel {
        nodes: vec![
            StructuralNode {
                node_id: 1,
                coordinates_m: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new([0.0, 0.0, 0.0]),
                solid_attachment: None,
            },
            StructuralNode {
                node_id: 2,
                coordinates_m: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new([1.0, 0.0, 0.0]),
                solid_attachment: None,
            },
        ],
        elements: vec![StructuralElement {
            element_id: "beam_1".to_string(),
            region_id: runmat_meshing_core::fixtures::fixture_edge("beam_span"),
            kind: StructuralElementKind::Beam(BeamElementModel {
                node_ids: [1, 2],
                section_id: "rect".to_string(),
                reference_axis: [0.0, 1.0, 0.0],
            }),
        }],
        beam_sections: vec![BeamSectionModel {
            section_id: "rect".to_string(),
            area_m2: runmat_types::physical::Quantity::<runmat_types::physical::kind::Area, _>::new(
                2.0e-4,
            ),
            iy_m4: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::SecondMomentOfArea,
                _,
            >::new(1.6e-9),
            iz_m4: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::SecondMomentOfArea,
                _,
            >::new(6.4e-9),
            torsion_j_m4: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::SecondMomentOfArea,
                _,
            >::new(2.4e-9),
            outer_fiber_y_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Length,
                _,
            >::new(0.01),
            outer_fiber_z_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Length,
                _,
            >::new(0.005),
            torsion_outer_radius_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Length,
                _,
            >::new(0.011_180_339_887_498_949),
        }],
        shell_sections: Vec::new(),
    });
    model
}

fn modal_large_fixture() -> AnalysisModel {
    let mut model = cantilever_large_load_sweep();
    model.model_id = AnalysisModelId("modal_large_fixture".to_string());
    model.sequence.steps = vec![ProcedureStep::structural_modal("modal_large_1".to_string())];
    model
}

fn transient_long_fixture() -> AnalysisModel {
    let mut model = cantilever_load_sweep();
    model.model_id = AnalysisModelId("transient_long_fixture".to_string());
    model.sequence.steps = vec![ProcedureStep::structural_transient(
        "transient_long_1".to_string(),
    )];
    model
}

fn transient_shock_fixture() -> AnalysisModel {
    let mut model = cantilever_large_load_sweep();
    model.model_id = AnalysisModelId("transient_shock_fixture".to_string());
    model.constraints.push(prescribed_zero_translations(
        "bc_mid_prescribed",
        "mid_support",
    ));
    let excitations = (0..256)
        .map(|i| {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            let scale = 1.0 + (i as f64) * 0.01;
            FixtureExcitation {
                id: format!("shock_load_{i}"),
                region_id: format!("shock_region_{i}"),
                kind: FixtureExcitationKind::Force([50.0 * scale, sign * -1500.0 * scale, 0.0]),
            }
        })
        .collect();
    replace_fixture_excitations(&mut model, excitations);
    model.sequence.steps = vec![ProcedureStep::structural_transient(
        "transient_shock_1".to_string(),
    )];
    model
}

fn structural_transient_tet4_reference() -> AnalysisModel {
    let mut model = cantilever_linear_static();
    model.model_id = AnalysisModelId("structural_transient_tet4_reference".into());
    model.geometry_id = "geo:structural_transient_tet4_reference".into();
    let MaterialElasticityModel::IsotropicLinear(mechanical) = model.materials[0]
        .elasticity
        .as_mut()
        .expect("cantilever fixture declares linear elasticity")
    else {
        unreachable!("cantilever fixture declares isotropic linear elasticity")
    };
    mechanical.youngs_modulus_pa = runmat_types::physical::YoungsModulus::new(1.0e5);
    mechanical.poisson_ratio = runmat_types::physical::PoissonRatio::new(0.25);
    model.materials[0].mass_density_kg_per_m3 =
        Some(runmat_types::physical::MassDensity::new(1_000.0));
    model.materials[0].damping = Some(MechanicalDampingModel::Rayleigh {
        mass_coefficient: RayleighMassCoefficient::new(0.2),
        stiffness_coefficient: RayleighStiffnessCoefficient::new(2.0e-4),
    });
    model.material_assignments = vec![MaterialAssignment {
        target: PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_region(
            "dynamic_volume",
        )),
        expected_material_id: "mat_steel".into(),
        assigned_material_id: "mat_steel".into(),
        confidence: EvidenceConfidence::Verified,
    }];
    model.constraints = vec![
        fixed_constraint("fixed_root", "root"),
        AnalysisConstraint::Structural(StructuralConstraint {
            constraint_id: "free_vertex_y_support".into(),
            frame: CoordinateFrame::Global,
            kind: StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
                target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                    "free_vertex_y_support",
                )),
                translations: vec![TranslationDofValue {
                    dof: StructuralTranslationDof::Uy,
                    value: PrescribedValue::Constant(Displacement::new(0.0)),
                }],
                rotations: Vec::new(),
            }),
        }),
        AnalysisConstraint::Structural(StructuralConstraint {
            constraint_id: "free_vertex_z_support".into(),
            frame: CoordinateFrame::Global,
            kind: StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
                target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                    "free_vertex_z_support",
                )),
                translations: vec![TranslationDofValue {
                    dof: StructuralTranslationDof::Uz,
                    value: PrescribedValue::Constant(Displacement::new(0.0)),
                }],
                rotations: Vec::new(),
            }),
        }),
    ];
    replace_fixture_excitations(
        &mut model,
        vec![FixtureExcitation {
            id: "dynamic_force_x".into(),
            region_id: "dynamic_force".into(),
            kind: FixtureExcitationKind::Force([1.0, 0.0, 0.0]),
        }],
    );
    model.sequence = AnalysisSequence::from_steps(
        "structural_transient_tet4_reference_sequence",
        vec![ProcedureStep::structural_transient("transient")],
    );
    model
}

fn structural_transient_tet4_support_motion_reference() -> AnalysisModel {
    let mut model = structural_transient_tet4_reference();
    model.model_id = AnalysisModelId("structural_transient_tet4_support_motion_reference".into());
    model.geometry_id = "geo:structural_transient_tet4_support_motion_reference".into();
    replace_fixture_excitations(&mut model, Vec::new());
    let acceleration_id = SignalId("root-acceleration".into());
    let ordinate = |x: f64| {
        SignalOrdinate::Real(SignalValue::Vector3([
            Acceleration::new(x),
            Acceleration::new(0.0),
            Acceleration::new(0.0),
        ]))
    };
    model.signals.push(PhysicalSignal::Acceleration(Signal {
        signal_id: acceleration_id.clone(),
        shape: SignalShape::Vector3,
        frame: Some(CoordinateFrame::Global),
        representation: SignalRepresentation::Tabulated(TabulatedSignal {
            domain: TabulatedSignalDomain::TimeSeconds,
            knots: vec![
                SignalKnot::continuous(0.0, ordinate(1.0)),
                SignalKnot::continuous(0.5, ordinate(-1.0)),
                SignalKnot::continuous(1.0, ordinate(1.0)),
            ],
            interpolation: SignalInterpolation::Linear,
            extrapolation: SignalExtrapolation::Reject,
        }),
        precision: SignalPrecision::Real64,
        evidence: SignalApproximationEvidence {
            max_absolute_error: Some(0.0),
            max_relative_error: Some(0.0),
            derivative_max_absolute_error: None,
            sample_count: 3,
        },
        provenance: SignalProvenance::Authored,
    }));
    model.constraints[0] = AnalysisConstraint::Structural(StructuralConstraint {
        constraint_id: "moving_root_x".into(),
        frame: CoordinateFrame::Global,
        kind: StructuralConstraintKind::PrescribedMotion(PrescribedStructuralMotion {
            target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("root")),
            translations: vec![TranslationDofMotion {
                dof: StructuralTranslationDof::Ux,
                history: StructuralKinematicHistory::AccelerationDriven {
                    acceleration: AccelerationSignalId(acceleration_id),
                    initial_displacement: Displacement::new(0.0),
                    initial_velocity: Velocity::new(0.0),
                    reconstruction:
                        StructuralMotionReconstruction::PiecewiseLinearAccelerationExact,
                },
            }],
        }),
    });
    model
        .constraints
        .push(AnalysisConstraint::Structural(StructuralConstraint {
            constraint_id: "fixed_root_yz".into(),
            frame: CoordinateFrame::Global,
            kind: StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
                target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("root")),
                translations: [StructuralTranslationDof::Uy, StructuralTranslationDof::Uz]
                    .into_iter()
                    .map(|dof| TranslationDofValue {
                        dof,
                        value: PrescribedValue::Constant(Displacement::new(0.0)),
                    })
                    .collect(),
                rotations: Vec::new(),
            }),
        }));
    model
}

fn nonlinear_assembly_fixture() -> AnalysisModel {
    let mut model = transient_shock_fixture();
    model.model_id = AnalysisModelId("nonlinear_assembly_fixture".to_string());
    model.sequence.steps = vec![ProcedureStep::structural_nonlinear(
        "nonlinear_assembly_1".to_string(),
    )];
    model
}

fn nonlinear_assembly_stress_fixture() -> AnalysisModel {
    let mut model = nonlinear_assembly_fixture();
    model.model_id = AnalysisModelId("nonlinear_assembly_stress_fixture".to_string());
    model.constraints.push(prescribed_zero_translations(
        "bc_stress_mid_support",
        "mid_support_stress",
    ));
    let excitations = (0..640)
        .map(|i| {
            let phase = if i % 3 == 0 { -1.0 } else { 1.0 };
            let scale = 1.0 + (i as f64) * 0.003;
            FixtureExcitation {
                id: format!("nonlinear_stress_load_{i}"),
                region_id: format!("nonlinear_stress_region_{}", i % 48),
                kind: FixtureExcitationKind::Force([
                    75.0 * scale,
                    phase * -1800.0 * scale,
                    20.0 * scale,
                ]),
            }
        })
        .collect();
    replace_fixture_excitations(&mut model, excitations);
    model.sequence.steps = vec![ProcedureStep::structural_nonlinear(
        "nonlinear_stress_1".to_string(),
    )];
    model
}

fn nonlinear_softening_benchmark_fixture() -> AnalysisModel {
    let mut model = nonlinear_assembly_stress_fixture();
    model.model_id = AnalysisModelId("nonlinear_softening_benchmark_fixture".to_string());
    model.materials = vec![
        isotropic_material(
            "mat_soft_polymer",
            "Soft Polymer",
            1.4e9,
            0.39,
            1200.0,
            -1.2e-3,
        ),
        isotropic_material("mat_aluminum", "Aluminum", 69e9, 0.33, 2700.0, -3.6e-4),
    ];
    model.material_assignments = vec![
        MaterialAssignment {
            target: PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_region(
                "nonlinear_soft_region_root",
            )),
            expected_material_id: "mat_soft_polymer".to_string(),
            assigned_material_id: "mat_soft_polymer".to_string(),
            confidence: EvidenceConfidence::Verified,
        },
        MaterialAssignment {
            target: PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_region(
                "nonlinear_soft_region_tip",
            )),
            expected_material_id: "mat_aluminum".to_string(),
            assigned_material_id: "mat_aluminum".to_string(),
            confidence: EvidenceConfidence::Verified,
        },
    ];
    let excitations = (0..720)
        .map(|i| {
            let phase = if i % 4 == 0 { -1.0 } else { 1.0 };
            let drift = 1.0 + (i as f64) * 0.0025;
            FixtureExcitation {
                id: format!("nonlinear_softening_load_{i}"),
                region_id: format!("nonlinear_softening_region_{}", i % 64),
                kind: FixtureExcitationKind::Force([
                    65.0 * drift,
                    phase * -2100.0 * drift,
                    28.0 * drift,
                ]),
            }
        })
        .collect();
    replace_fixture_excitations(&mut model, excitations);
    model.sequence.steps = vec![ProcedureStep::structural_nonlinear(
        "nonlinear_softening_1".to_string(),
    )];
    model
}

fn nonlinear_load_path_mix_fixture() -> AnalysisModel {
    let mut model = multi_material_assembly();
    model.model_id = AnalysisModelId("nonlinear_load_path_mix_fixture".to_string());
    model.constraints.push(prescribed_zero_translations(
        "bc_mix_path_support",
        "mix_path_support",
    ));
    let excitations = (0..480)
        .map(|i| {
            let scale = 1.0 + (i as f64) * 0.0035;
            if i % 3 == 0 {
                FixtureExcitation {
                    id: format!("mix_force_{i}"),
                    region_id: format!("mix_force_region_{}", i % 36),
                    kind: FixtureExcitationKind::Force([
                        90.0 * scale,
                        -1300.0 * scale,
                        40.0 * scale,
                    ]),
                }
            } else if i % 3 == 1 {
                FixtureExcitation {
                    id: format!("mix_pressure_{i}"),
                    region_id: format!("mix_pressure_region_{}", i % 24),
                    kind: FixtureExcitationKind::Pressure(9.0e5 * scale),
                }
            } else {
                let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                FixtureExcitation {
                    id: format!("mix_body_{i}"),
                    region_id: format!("mix_body_region_{}", i % 18),
                    kind: FixtureExcitationKind::BodyForce([
                        0.35 * scale,
                        sign * -9.81 * scale,
                        0.12 * scale,
                    ]),
                }
            }
        })
        .collect();
    replace_fixture_excitations(&mut model, excitations);
    model.sequence.steps = vec![ProcedureStep::structural_nonlinear(
        "nonlinear_mix_1".to_string(),
    )];
    model
}

fn nonlinear_contact_frictionless_reference_fixture() -> AnalysisModel {
    let mut model = nonlinear_load_path_mix_fixture();
    model.model_id =
        AnalysisModelId("nonlinear_contact_frictionless_reference_fixture".to_string());
    model.material_assignments = vec![
        MaterialAssignment {
            target: PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_region(
                "tip_steel",
            )),
            expected_material_id: "mat_steel".to_string(),
            assigned_material_id: "mat_steel".to_string(),
            confidence: EvidenceConfidence::Verified,
        },
        MaterialAssignment {
            target: PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_region(
                "mid_aluminum",
            )),
            expected_material_id: "mat_aluminum".to_string(),
            assigned_material_id: "mat_aluminum".to_string(),
            confidence: EvidenceConfidence::Verified,
        },
        MaterialAssignment {
            target: PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_region(
                "polymer_segment",
            )),
            expected_material_id: "mat_polymer".to_string(),
            assigned_material_id: "mat_polymer".to_string(),
            confidence: EvidenceConfidence::Verified,
        },
    ];
    model
}

fn nonlinear_contact_frictionless_reference_complex_fixture() -> AnalysisModel {
    let mut model = nonlinear_contact_frictionless_reference_fixture();
    model.model_id =
        AnalysisModelId("nonlinear_contact_frictionless_reference_complex_fixture".to_string());
    let excitations = (0..560)
        .map(|i| {
            let scale = 1.0 + (i as f64) * 0.004;
            if i % 2 == 0 {
                FixtureExcitation {
                    id: format!("contact_ref_force_{i}"),
                    region_id: format!("contact_ref_force_region_{}", i % 32),
                    kind: FixtureExcitationKind::Force([
                        80.0 * scale,
                        -1400.0 * scale,
                        32.0 * scale,
                    ]),
                }
            } else {
                FixtureExcitation {
                    id: format!("contact_ref_pressure_{i}"),
                    region_id: format!("contact_ref_pressure_region_{}", i % 28),
                    kind: FixtureExcitationKind::Pressure(1.05e6 * scale),
                }
            }
        })
        .collect();
    replace_fixture_excitations(&mut model, excitations);
    model
}

fn nonlinear_plastic_hardening_reference_fixture() -> AnalysisModel {
    let mut model = nonlinear_load_path_mix_fixture();
    model.model_id = AnalysisModelId("nonlinear_plastic_hardening_reference_fixture".to_string());
    model.material_assignments = vec![
        MaterialAssignment {
            target: PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_region(
                "tip_steel",
            )),
            expected_material_id: "mat_steel".to_string(),
            assigned_material_id: "mat_steel".to_string(),
            confidence: EvidenceConfidence::Verified,
        },
        MaterialAssignment {
            target: PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_region(
                "mid_aluminum",
            )),
            expected_material_id: "mat_aluminum".to_string(),
            assigned_material_id: "mat_aluminum".to_string(),
            confidence: EvidenceConfidence::Verified,
        },
        MaterialAssignment {
            target: PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_region(
                "polymer_segment",
            )),
            expected_material_id: "mat_polymer".to_string(),
            assigned_material_id: "mat_polymer".to_string(),
            confidence: EvidenceConfidence::Verified,
        },
    ];
    model
}

fn nonlinear_plastic_hardening_reference_complex_fixture() -> AnalysisModel {
    let mut model = nonlinear_plastic_hardening_reference_fixture();
    model.model_id =
        AnalysisModelId("nonlinear_plastic_hardening_reference_complex_fixture".to_string());
    let excitations = (0..620)
        .map(|i| {
            let scale = 1.0 + (i as f64) * 0.0045;
            if i % 3 == 0 {
                FixtureExcitation {
                    id: format!("plastic_ref_force_{i}"),
                    region_id: format!("plastic_ref_force_region_{}", i % 36),
                    kind: FixtureExcitationKind::Force([
                        85.0 * scale,
                        -1450.0 * scale,
                        28.0 * scale,
                    ]),
                }
            } else if i % 3 == 1 {
                FixtureExcitation {
                    id: format!("plastic_ref_pressure_{i}"),
                    region_id: format!("plastic_ref_pressure_region_{}", i % 30),
                    kind: FixtureExcitationKind::Pressure(1.1e6 * scale),
                }
            } else {
                FixtureExcitation {
                    id: format!("plastic_ref_body_{i}"),
                    region_id: format!("plastic_ref_body_region_{}", i % 22),
                    kind: FixtureExcitationKind::BodyForce([
                        0.28 * scale,
                        -9.81 * scale,
                        0.09 * scale,
                    ]),
                }
            }
        })
        .collect();
    replace_fixture_excitations(&mut model, excitations);
    model
}

fn thermo_mechanical_kickoff_fixture() -> AnalysisModel {
    let mut model = multi_material_assembly();
    model.model_id = AnalysisModelId("thermo_mechanical_kickoff_fixture".to_string());
    let excitations = (0..240)
        .map(|i| {
            let scale = 1.0 + (i as f64) * 0.004;
            FixtureExcitation {
                id: format!("thermo_mech_force_{i}"),
                region_id: format!("thermo_mech_region_{}", i % 24),
                kind: FixtureExcitationKind::Force([35.0 * scale, -900.0 * scale, 12.0 * scale]),
            }
        })
        .collect();
    replace_fixture_excitations(&mut model, excitations);
    model.sequence.steps = vec![ProcedureStep::coupled(
        "thermo_mech_transient_1",
        runmat_analysis_core::CoupledPhysicsFamily::Thermomechanical,
        runmat_analysis_core::AnalysisProcedure::TransientResponse,
    )];
    model
}

fn thermo_gradient_benign_fixture() -> AnalysisModel {
    let mut model = multi_material_assembly();
    model.model_id = AnalysisModelId("thermo_gradient_benign_fixture".to_string());
    model.material_assignments = vec![
        MaterialAssignment {
            target: PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_region(
                "tip_steel",
            )),
            expected_material_id: "mat_steel".to_string(),
            assigned_material_id: "mat_steel".to_string(),
            confidence: EvidenceConfidence::Verified,
        },
        MaterialAssignment {
            target: PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_region(
                "mid_aluminum",
            )),
            expected_material_id: "mat_aluminum".to_string(),
            assigned_material_id: "mat_aluminum".to_string(),
            confidence: EvidenceConfidence::Verified,
        },
        MaterialAssignment {
            target: PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_region(
                "polymer_segment",
            )),
            expected_material_id: "mat_polymer".to_string(),
            assigned_material_id: "mat_polymer".to_string(),
            confidence: EvidenceConfidence::Probable,
        },
    ];
    let excitations = (0..260)
        .map(|i| {
            let scale = 1.0 + (i as f64) * 0.003;
            FixtureExcitation {
                id: format!("thermo_grad_benign_load_{i}"),
                region_id: format!("thermo_grad_benign_region_{}", i % 28),
                kind: FixtureExcitationKind::Force([30.0 * scale, -850.0 * scale, 14.0 * scale]),
            }
        })
        .collect();
    replace_fixture_excitations(&mut model, excitations);
    model.sequence.steps = vec![ProcedureStep::structural_transient(
        "thermo_grad_benign_transient_1".to_string(),
    )];
    model
}

fn thermo_gradient_pathological_fixture() -> AnalysisModel {
    let mut model = multi_material_assembly();
    model.model_id = AnalysisModelId("thermo_gradient_pathological_fixture".to_string());
    model.material_assignments = vec![
        MaterialAssignment {
            target: PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_region(
                "tip_steel",
            )),
            expected_material_id: "mat_steel".to_string(),
            assigned_material_id: "mat_polymer".to_string(),
            confidence: EvidenceConfidence::Verified,
        },
        MaterialAssignment {
            target: PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_region(
                "mid_aluminum",
            )),
            expected_material_id: "mat_aluminum".to_string(),
            assigned_material_id: "mat_polymer".to_string(),
            confidence: EvidenceConfidence::Verified,
        },
        MaterialAssignment {
            target: PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_region(
                "polymer_segment",
            )),
            expected_material_id: "mat_polymer".to_string(),
            assigned_material_id: "mat_steel".to_string(),
            confidence: EvidenceConfidence::Inferred,
        },
    ];
    let excitations = (0..320)
        .map(|i| {
            let scale = 1.0 + (i as f64) * 0.0038;
            FixtureExcitation {
                id: format!("thermo_grad_pathological_load_{i}"),
                region_id: format!("thermo_grad_pathological_region_{}", i % 32),
                kind: FixtureExcitationKind::Force([52.0 * scale, -1150.0 * scale, 26.0 * scale]),
            }
        })
        .collect();
    replace_fixture_excitations(&mut model, excitations);
    model.sequence.steps = vec![ProcedureStep::structural_transient(
        "thermo_grad_pathological_transient_1".to_string(),
    )];
    model
}

fn thermo_ramp_smooth_fixture() -> AnalysisModel {
    let mut model = thermo_gradient_benign_fixture();
    model.model_id = AnalysisModelId("thermo_ramp_smooth_fixture".to_string());
    let excitations = (0..280)
        .map(|i| {
            let scale = 0.6 + (i as f64) * 0.0025;
            FixtureExcitation {
                id: format!("thermo_ramp_smooth_load_{i}"),
                region_id: format!("thermo_ramp_smooth_region_{}", i % 30),
                kind: FixtureExcitationKind::Force([24.0 * scale, -760.0 * scale, 10.0 * scale]),
            }
        })
        .collect();
    replace_fixture_excitations(&mut model, excitations);
    model
}

fn thermo_shock_oscillatory_fixture() -> AnalysisModel {
    let mut model = thermo_gradient_pathological_fixture();
    model.model_id = AnalysisModelId("thermo_shock_oscillatory_fixture".to_string());
    let excitations = (0..360)
        .map(|i| {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            let scale = 1.0 + (i as f64) * 0.004;
            FixtureExcitation {
                id: format!("thermo_shock_osc_load_{i}"),
                region_id: format!("thermo_shock_osc_region_{}", i % 36),
                kind: FixtureExcitationKind::Force([
                    sign * 48.0 * scale,
                    sign * -1320.0 * scale,
                    sign * 28.0 * scale,
                ]),
            }
        })
        .collect();
    replace_fixture_excitations(&mut model, excitations);
    model
}

fn electro_thermal_joule_benign_fixture() -> AnalysisModel {
    let mut model = transient_long_fixture();
    model.model_id = AnalysisModelId("electro_thermal_joule_benign_fixture".to_string());
    model
}

fn electro_thermal_joule_pathological_fixture() -> AnalysisModel {
    let mut model = transient_shock_fixture();
    model.model_id = AnalysisModelId("electro_thermal_joule_pathological_fixture".to_string());
    model
}

fn multi_material_assembly() -> AnalysisModel {
    let mut model = cantilever_linear_static();
    model.model_id = AnalysisModelId("multi_material_assembly".to_string());
    model.materials = vec![
        isotropic_material("mat_steel", "Steel", 200e9, 0.3, 7850.0, -2.5e-4),
        isotropic_material("mat_aluminum", "Aluminum", 69e9, 0.33, 2700.0, -3.6e-4),
        isotropic_material("mat_polymer", "Polymer", 3.2e9, 0.37, 1150.0, -8.0e-4),
    ];

    model.constraints = vec![
        fixed_constraint("bc_root", "root"),
        prescribed_zero_translations("bc_interface", "interface"),
    ];

    replace_fixture_excitations(
        &mut model,
        vec![
            FixtureExcitation {
                id: "load_tip_force".into(),
                region_id: "tip_steel".into(),
                kind: FixtureExcitationKind::Force([0.0, -1200.0, 0.0]),
            },
            FixtureExcitation {
                id: "load_mid_pressure".into(),
                region_id: "mid_aluminum".into(),
                kind: FixtureExcitationKind::Pressure(8.5e5),
            },
            FixtureExcitation {
                id: "load_body".into(),
                region_id: "polymer_segment".into(),
                kind: FixtureExcitationKind::BodyForce([0.0, -9.81, 0.0]),
            },
        ],
    );

    model.material_assignments = vec![
        MaterialAssignment {
            target: PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_region(
                "tip_steel",
            )),
            expected_material_id: "mat_steel".to_string(),
            assigned_material_id: "mat_steel".to_string(),
            confidence: EvidenceConfidence::Verified,
        },
        MaterialAssignment {
            target: PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_region(
                "mid_aluminum",
            )),
            expected_material_id: "mat_aluminum".to_string(),
            assigned_material_id: "mat_polymer".to_string(),
            confidence: EvidenceConfidence::Inferred,
        },
        MaterialAssignment {
            target: PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_region(
                "polymer_segment",
            )),
            expected_material_id: "mat_polymer".to_string(),
            assigned_material_id: "mat_polymer".to_string(),
            confidence: EvidenceConfidence::Probable,
        },
    ];

    model
}

fn missing_materials() -> AnalysisModel {
    let mut model = cantilever_linear_static();
    model.model_id = AnalysisModelId("missing_materials".to_string());
    model.materials.clear();
    model
}
