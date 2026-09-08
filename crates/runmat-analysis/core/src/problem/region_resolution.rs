//! Resolution of importer-authored boundary aliases into persistent geometry identities.

use std::collections::BTreeMap;

use runmat_geometry_core::{PersistentEntityId, PersistentEntityKind};

use super::{
    excitation::ExcitationTarget,
    model::AnalysisModel,
    target::{EntityDimension, ExpectedTopology, RegionTarget},
};
use crate::ObservationTarget;

/// A validated mapping from geometry-importer boundary aliases to persistent face identities.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BoundaryRegionAliases {
    aliases: BTreeMap<String, PersistentEntityId>,
}

impl BoundaryRegionAliases {
    pub fn new(
        aliases: BTreeMap<String, PersistentEntityId>,
    ) -> Result<Self, BoundaryRegionResolutionError> {
        for (alias, identity) in &aliases {
            if !is_importer_face_alias(alias) {
                return Err(BoundaryRegionResolutionError::InvalidAlias {
                    alias: alias.clone(),
                });
            }
            identity.validate().map_err(|error| {
                BoundaryRegionResolutionError::InvalidPersistentIdentity {
                    alias: alias.clone(),
                    reason: error.to_string(),
                }
            })?;
            if identity.kind != PersistentEntityKind::Face {
                return Err(BoundaryRegionResolutionError::NonFaceIdentity {
                    alias: alias.clone(),
                    kind: identity.kind,
                });
            }
        }
        Ok(Self { aliases })
    }

    pub fn get(&self, alias: &str) -> Option<&PersistentEntityId> {
        self.aliases.get(alias)
    }

    pub fn is_empty(&self) -> bool {
        self.aliases.is_empty()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum BoundaryRegionResolutionError {
    #[error("boundary region alias `{alias}` is invalid")]
    InvalidAlias { alias: String },
    #[error("boundary region alias `{alias}` has an invalid persistent identity: {reason}")]
    InvalidPersistentIdentity { alias: String, reason: String },
    #[error("boundary region alias `{alias}` resolves to {kind:?}, not a face")]
    NonFaceIdentity {
        alias: String,
        kind: PersistentEntityKind,
    },
    #[error("importer boundary alias `{alias}` has no persistent face resolution")]
    UnresolvedImporterAlias { alias: String },
}

/// Returns a model with every mapped boundary target rewritten to its canonical persistent
/// selection identity.
///
/// The input model is never partially changed. Importer aliases use the canonical `face_######`
/// spelling and fail closed when absent from `aliases`. Other authored or already-persistent region
/// identifiers remain unchanged when they do not occur in the mapping.
pub fn resolve_boundary_region_aliases(
    model: &AnalysisModel,
    aliases: &BoundaryRegionAliases,
) -> Result<AnalysisModel, BoundaryRegionResolutionError> {
    let mut resolved = model.clone();
    visit_boundary_targets(&mut resolved, |target| {
        if target.entity_dimension != EntityDimension::Face
            || target.expected_topology != ExpectedTopology::BoundaryFace
            || !is_importer_face_alias(&target.region_id)
        {
            return Ok(());
        }
        if let Some(identity) = aliases.get(&target.region_id) {
            target.region_id = identity.analysis_selection_id().map_err(|error| {
                BoundaryRegionResolutionError::InvalidPersistentIdentity {
                    alias: target.region_id.clone(),
                    reason: error.to_string(),
                }
            })?;
            Ok(())
        } else {
            Err(BoundaryRegionResolutionError::UnresolvedImporterAlias {
                alias: target.region_id.clone(),
            })
        }
    })?;
    Ok(resolved)
}

fn visit_boundary_targets(
    model: &mut AnalysisModel,
    mut visit: impl FnMut(&mut RegionTarget) -> Result<(), BoundaryRegionResolutionError>,
) -> Result<(), BoundaryRegionResolutionError> {
    for constraint in &mut model.constraints {
        let mut failure = None;
        constraint.visit_targets_mut(|target| {
            if failure.is_none() {
                failure = visit(target).err();
            }
        });
        if let Some(error) = failure {
            return Err(error);
        }
    }
    for boundary in &mut model.boundaries {
        visit(&mut boundary.target)?;
    }
    for connector in &mut model.connectors {
        visit(&mut connector.primary)?;
        if let Some(secondary) = &mut connector.secondary {
            visit(secondary)?;
        }
    }
    for interface in &mut model.interfaces {
        visit(&mut interface.primary)?;
        visit(&mut interface.secondary)?;
    }
    for action in &mut model.excitations {
        if let ExcitationTarget::Region(target) = &mut action.excitation_mut().target {
            visit(target)?;
        }
    }
    for condition in &mut model.initial_conditions {
        if let Some(target) = condition.region_target_mut() {
            visit(target)?;
        }
    }
    for step in &mut model.sequence.steps {
        for observation in &mut step.observations {
            let Some(spec) = observation.spec_mut() else {
                continue;
            };
            if let ObservationTarget::Spatial(ExcitationTarget::Region(target)) = &mut spec.target {
                visit(target)?;
            }
        }
    }
    Ok(())
}

fn is_importer_face_alias(region_id: &str) -> bool {
    let Some(ordinal) = region_id.strip_prefix("face_") else {
        return false;
    };
    ordinal.len() == 6 && ordinal.bytes().all(|byte| byte.is_ascii_digit()) && ordinal != "000000"
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AnalysisBoundary, AnalysisConnector, AnalysisConstraint, AnalysisExcitation,
        AnalysisInterface, AnalysisInterfaceKind, AnalysisModelId, AnalysisSequence,
        BoundaryOperator, CompressionPolicy, ConnectorKind, ContactFriction, ContactInterfaceModel,
        CoordinateFrame, ExcitationAction, ExcitationCaseId, ExcitationDistribution, ExcitationId,
        ExcitationKinematics, FieldLocation, InitialCondition, InitialConditionTarget,
        InitialConditionValue, InitialValue, ObservationComponent, ObservationCostEstimate,
        ObservationId, ObservationPurpose, ObservationQuantity, ObservationRequest,
        ObservationSpec, ObservationStoragePolicy, ObservationValueShape, PartialAccessPolicy,
        ProjectionRequirement, RecoveryPolicy, RegionTarget, RetentionPolicy, SamplingPolicy,
        SignConvention, SignalId, SignalPrecision, SpatialReduction, StructuralConstraint,
        StructuralConstraintKind,
    };
    use runmat_geometry_core::UnitSystem;

    fn face(id: &str) -> PersistentEntityId {
        PersistentEntityId {
            kind: PersistentEntityKind::Face,
            source_topology_id: id.into(),
            assembly_path: vec!["root".into()],
        }
    }

    fn selection(id: &str) -> String {
        face(id).analysis_selection_id().unwrap()
    }

    fn model_with_all_boundary_targets() -> AnalysisModel {
        let mut sequence = AnalysisSequence::from_steps("sequence", Vec::new());
        sequence
            .steps
            .push(crate::ProcedureStep::structural_static("step"));
        sequence.steps[0]
            .observations
            .push(ObservationRequest::Field(ObservationSpec {
                observation_id: ObservationId("observation".into()),
                quantity: ObservationQuantity::Displacement,
                value_shape: ObservationValueShape::Vector3,
                target: ObservationTarget::Spatial(ExcitationTarget::Region(
                    RegionTarget::boundary("face_000008"),
                )),
                source_location: FieldLocation::Node,
                output_location: FieldLocation::Node,
                recovery: RecoveryPolicy::Native,
                component: ObservationComponent::All,
                frame: CoordinateFrame::Global,
                sign_convention: SignConvention::Native,
                complex_representation: crate::ComplexRepresentation::Real,
                sampling: SamplingPolicy::Final,
                reduction: Some(crate::ReductionPolicy {
                    spatial: SpatialReduction::None,
                    series: crate::SeriesReduction::None,
                    extrema_ties: crate::ExtremaTiePolicy::FirstCanonical,
                }),
                storage: ObservationStoragePolicy {
                    precision: SignalPrecision::Real64,
                    compression: CompressionPolicy::None,
                    retention: RetentionPolicy::Canonical,
                    partial_access: PartialAccessPolicy::ManifestAndChunks,
                },
                purpose: ObservationPurpose::Visualization,
                cost: ObservationCostEstimate {
                    operation_count: 0,
                    peak_resident_bytes: 0,
                    stored_bytes: 0,
                    transfer_bytes: 0,
                    collective_count: 0,
                },
            }));
        AnalysisModel {
            model_id: AnalysisModelId("model".into()),
            geometry_id: "geometry".into(),
            geometry_revision: 1,
            units: UnitSystem::Meter,
            frame: CoordinateFrame::Global,
            materials: Vec::new(),
            material_assignments: Vec::new(),
            structural: None,
            thermal: None,
            thermo_mechanical: None,
            electro_thermal: None,
            electromagnetic: None,
            cfd: None,
            interfaces: vec![AnalysisInterface {
                interface_id: "interface".into(),
                primary: RegionTarget::boundary("face_000005"),
                secondary: RegionTarget::boundary("face_000006"),
                kind: AnalysisInterfaceKind::Contact(ContactInterfaceModel {
                    friction: ContactFriction::Frictionless,
                }),
            }],
            constraints: vec![AnalysisConstraint::Structural(StructuralConstraint {
                constraint_id: "constraint".into(),
                frame: CoordinateFrame::Global,
                kind: StructuralConstraintKind::Fixed {
                    target: RegionTarget::boundary("face_000001"),
                },
            })],
            boundaries: vec![AnalysisBoundary {
                boundary_id: "boundary".into(),
                target: RegionTarget::boundary("face_000002"),
                operator: BoundaryOperator::MagneticInsulation,
            }],
            connectors: vec![AnalysisConnector {
                connector_id: "connector".into(),
                primary: RegionTarget::boundary("face_000003"),
                secondary: Some(RegionTarget::boundary("face_000004")),
                kind: ConnectorKind::TranslationalSpring {
                    stiffness: crate::Stiffness::new(1.0),
                },
            }],
            signals: Vec::new(),
            excitations: vec![ExcitationAction::Force(AnalysisExcitation {
                excitation_id: ExcitationId("force".into()),
                case_id: ExcitationCaseId("case".into()),
                target: ExcitationTarget::Region(RegionTarget::boundary("face_000007")),
                frame: CoordinateFrame::Global,
                direction: None,
                distribution: ExcitationDistribution::PreserveResultant,
                kinematics: ExcitationKinematics::FixedFrame,
                signal_id: SignalId("signal".into()),
                active_step_ids: Vec::new(),
                combination_group: None,
                projection: ProjectionRequirement {
                    preserve_resultant: true,
                    preserve_moment: false,
                    relative_tolerance: 1.0e-10,
                },
            })],
            excitation_cases: Vec::new(),
            initial_conditions: vec![InitialCondition::Temperature(InitialConditionValue {
                initial_condition_id: "initial".into(),
                target: InitialConditionTarget::Region(RegionTarget::boundary("face_000009")),
                frame: CoordinateFrame::Global,
                value: InitialValue::Uniform(crate::Temperature::new(300.0)),
                applies_before_step_id: "step".into(),
            })],
            response_spectra: Vec::new(),
            sequence,
        }
    }

    #[test]
    fn resolves_every_boundary_target_category_without_mutating_the_input() {
        let model = model_with_all_boundary_targets();
        let aliases = BoundaryRegionAliases::new(
            (1..=9)
                .map(|index| {
                    (
                        format!("face_{index:06}"),
                        face(&format!("persistent-{index}")),
                    )
                })
                .collect(),
        )
        .unwrap();

        let resolved = resolve_boundary_region_aliases(&model, &aliases).unwrap();

        assert_eq!(
            model.constraints[0].target().unwrap().region_id,
            "face_000001"
        );
        assert_eq!(
            resolved.constraints[0].target().unwrap().region_id,
            selection("persistent-1")
        );
        assert_eq!(
            resolved.boundaries[0].target.region_id,
            selection("persistent-2")
        );
        assert_eq!(
            resolved.connectors[0].primary.region_id,
            selection("persistent-3")
        );
        assert_eq!(
            resolved.connectors[0].secondary.as_ref().unwrap().region_id,
            selection("persistent-4")
        );
        assert_eq!(
            resolved.interfaces[0].primary.region_id,
            selection("persistent-5")
        );
        assert_eq!(
            resolved.interfaces[0].secondary.region_id,
            selection("persistent-6")
        );
        assert!(matches!(
            &resolved.excitations[0].excitation().target,
            ExcitationTarget::Region(target) if target.region_id == selection("persistent-7")
        ));
        assert!(matches!(
            resolved.sequence.steps[0].observations[0].spec().map(|spec| &spec.target),
            Some(ObservationTarget::Spatial(ExcitationTarget::Region(target)))
                if target.region_id == selection("persistent-8")
        ));
        assert!(matches!(
            &resolved.initial_conditions[0],
            InitialCondition::Temperature(value)
                if matches!(&value.target, InitialConditionTarget::Region(target)
                    if target.region_id == selection("persistent-9"))
        ));
    }

    #[test]
    fn missing_importer_alias_fails_closed_and_authored_ids_remain_intact() {
        let mut model = model_with_all_boundary_targets();
        model.boundaries[0].target.region_id = "authored-wall".into();
        let aliases = BoundaryRegionAliases::new(BTreeMap::from([(
            "face_000001".into(),
            face("persistent-1"),
        )]))
        .unwrap();

        let error = resolve_boundary_region_aliases(&model, &aliases).unwrap_err();
        assert_eq!(
            error,
            BoundaryRegionResolutionError::UnresolvedImporterAlias {
                alias: "face_000003".into()
            }
        );
        assert_eq!(model.boundaries[0].target.region_id, "authored-wall");
        assert_eq!(
            model.constraints[0].target().unwrap().region_id,
            "face_000001"
        );
    }

    #[test]
    fn noncanonical_face_like_ids_are_not_treated_as_importer_aliases() {
        let mut model = model_with_all_boundary_targets();
        model.constraints[0].visit_targets_mut(|target| target.region_id = "face_17".into());
        model.boundaries.clear();
        model.connectors.clear();
        model.interfaces.clear();
        model.excitations.clear();
        model.initial_conditions.clear();
        model.sequence.steps[0].observations.clear();

        let resolved = resolve_boundary_region_aliases(
            &model,
            &BoundaryRegionAliases::new(BTreeMap::new()).unwrap(),
        )
        .unwrap();
        assert_eq!(
            resolved.constraints[0].target().unwrap().region_id,
            "face_17"
        );
    }

    #[test]
    fn importer_alias_grammar_is_one_based_and_exact() {
        for alias in [
            "face_000000",
            "face_17",
            "face_0000010",
            "face_00001a",
            "Face_000001",
            " face_000001",
            "face_000001 ",
            "authored-wall",
        ] {
            assert!(matches!(
                BoundaryRegionAliases::new(BTreeMap::from([(alias.into(), face("persistent"))])),
                Err(BoundaryRegionResolutionError::InvalidAlias { .. })
            ));
        }
        BoundaryRegionAliases::new(BTreeMap::from([("face_999999".into(), face("persistent"))]))
            .unwrap();
    }

    #[test]
    fn map_cannot_rewrite_authored_ids_and_non_boundary_targets_are_not_aliases() {
        let mut model = model_with_all_boundary_targets();
        model.constraints[0]
            .visit_targets_mut(|target| *target = RegionTarget::volume("face_000001"));
        model.boundaries[0].target = RegionTarget::point("face_000002");
        model.connectors.clear();
        model.interfaces.clear();
        model.excitations.clear();
        model.initial_conditions.clear();
        model.sequence.steps[0].observations.clear();

        let resolved = resolve_boundary_region_aliases(
            &model,
            &BoundaryRegionAliases::new(BTreeMap::from([
                ("face_000001".into(), face("persistent-1")),
                ("face_000002".into(), face("persistent-2")),
            ]))
            .unwrap(),
        )
        .unwrap();
        let target = resolved.constraints[0].target().unwrap();
        assert_eq!(target.region_id, "face_000001");
        assert_eq!(target.entity_dimension, EntityDimension::Volume);
        assert_eq!(resolved.boundaries[0].target.region_id, "face_000002");
        assert_eq!(
            resolved.boundaries[0].target.expected_topology,
            ExpectedTopology::MeshNode
        );

        assert!(matches!(
            BoundaryRegionAliases::new(BTreeMap::from([(
                "authored-wall".into(),
                face("persistent")
            )])),
            Err(BoundaryRegionResolutionError::InvalidAlias { .. })
        ));
    }

    #[test]
    fn alias_inventory_rejects_non_face_and_invalid_identities() {
        let non_face = PersistentEntityId {
            kind: PersistentEntityKind::Solid,
            source_topology_id: "solid".into(),
            assembly_path: Vec::new(),
        };
        assert!(matches!(
            BoundaryRegionAliases::new(BTreeMap::from([("face_000001".into(), non_face)])),
            Err(BoundaryRegionResolutionError::NonFaceIdentity { .. })
        ));
        assert!(matches!(
            BoundaryRegionAliases::new(BTreeMap::from([("face_000001".into(), face(""))])),
            Err(BoundaryRegionResolutionError::InvalidPersistentIdentity { .. })
        ));
    }
}
