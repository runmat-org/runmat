use std::collections::BTreeMap;

use runmat_geometry_core::GeometryRegionRole;

use crate::{
    AuthoredExtremaTiePolicy, AuthoredFieldDomain, AuthoredFieldProjection, AuthoredOutput,
    AuthoredOutputExpression, AuthoredPrincipalSelection, AuthoredRegionSource,
    AuthoredSpatialReduction, AuthoredVectorComponent, ComplexRepresentation, CompressionPolicy,
    CoordinateFrame, EntityDimension, ExcitationTarget, ExpectedTopology, ExtremaTiePolicy,
    FieldLocation, ObservationComponent, ObservationCostPolicy, ObservationId, ObservationPurpose,
    ObservationQuantity, ObservationRequest, ObservationSpec, ObservationStoragePolicy,
    ObservationTarget, ObservationValueShape, PartialAccessPolicy, ProcedureStep, ProcedureStepId,
    RecoveryPolicy, ReductionPolicy, RegionTarget, RetentionPolicy, SamplingPolicy,
    SeriesReduction, SignConvention, SignalPrecision, SpatialReduction, TargetCardinality,
};

use super::authored_output::field_domain;

#[derive(Debug, Clone, PartialEq)]
pub struct LoweredAuthoredObservation {
    pub step_id: ProcedureStepId,
    pub request: ObservationRequest,
}

/// Lower the checked authored output graph into the canonical executable observation contract.
///
/// This pass remains topology independent. Exact selected entities, storage sizes, transfer work,
/// and collective counts are resolved later against the admitted mesh and field catalog.
pub fn lower_authored_observations(
    outputs: &[AuthoredOutput],
    steps: &[ProcedureStep],
) -> Result<Vec<LoweredAuthoredObservation>, AuthoredObservationError> {
    let steps = steps
        .iter()
        .map(|step| (step.step_id.0.as_str(), step))
        .collect::<BTreeMap<_, _>>();
    let mut lowered = BTreeMap::<String, ObservationDraft>::new();
    let mut requests = Vec::with_capacity(outputs.len());
    for output in outputs {
        let mut draft = lower_expression(&output.expression, &steps, &lowered)?;
        draft.spec.observation_id = ObservationId(output.output_id.clone());
        let request = match draft.kind {
            ObservationKind::Field => ObservationRequest::Field(draft.spec.clone()),
            ObservationKind::Reduction => ObservationRequest::Reduction(draft.spec.clone()),
        };
        requests.push(LoweredAuthoredObservation {
            step_id: ProcedureStepId(draft.step_id.clone()),
            request,
        });
        lowered.insert(output.output_id.clone(), draft);
    }
    Ok(requests)
}

#[derive(Clone)]
struct ObservationDraft {
    step_id: String,
    spec: ObservationSpec,
    kind: ObservationKind,
}

#[derive(Clone, Copy)]
enum ObservationKind {
    Field,
    Reduction,
}

fn lower_expression(
    expression: &AuthoredOutputExpression,
    steps: &BTreeMap<&str, &ProcedureStep>,
    outputs: &BTreeMap<String, ObservationDraft>,
) -> Result<ObservationDraft, AuthoredObservationError> {
    match expression {
        AuthoredOutputExpression::AnalysisField {
            step_id,
            quantity,
            location,
        } => {
            let step = steps
                .get(step_id.as_str())
                .ok_or_else(|| AuthoredObservationError::UnknownStep(step_id.clone()))?;
            Ok(ObservationDraft {
                step_id: step_id.clone(),
                spec: base_spec(*quantity, *location, field_domain(step.procedure)),
                kind: ObservationKind::Field,
            })
        }
        AuthoredOutputExpression::OutputReference { output_id } => outputs
            .get(output_id)
            .cloned()
            .ok_or_else(|| AuthoredObservationError::UnknownOutput(output_id.clone())),
        AuthoredOutputExpression::Projection { projection, source } => {
            let mut draft = lower_expression(source, steps, outputs)?;
            apply_projection(&mut draft.spec, *projection)?;
            Ok(draft)
        }
        AuthoredOutputExpression::SpatialReduction {
            reduction,
            source,
            target,
            ties,
        } => {
            let mut draft = lower_expression(source, steps, outputs)?;
            draft.kind = ObservationKind::Reduction;
            draft.spec.target =
                ObservationTarget::Spatial(ExcitationTarget::Region(region_target(target)?));
            draft.spec.reduction = Some(ReductionPolicy {
                spatial: match reduction {
                    AuthoredSpatialReduction::Integral => SpatialReduction::Integral,
                    AuthoredSpatialReduction::Mean => SpatialReduction::Average,
                    AuthoredSpatialReduction::RootMeanSquare => SpatialReduction::RootMeanSquare,
                    AuthoredSpatialReduction::EuclideanNorm => SpatialReduction::EuclideanNorm,
                    AuthoredSpatialReduction::Minimum => SpatialReduction::Minimum,
                    AuthoredSpatialReduction::Maximum => SpatialReduction::Maximum,
                    AuthoredSpatialReduction::Resultant => SpatialReduction::Resultant,
                    AuthoredSpatialReduction::Moment { about } => {
                        SpatialReduction::Moment { about: *about }
                    }
                },
                series: SeriesReduction::None,
                extrema_ties: match ties {
                    AuthoredExtremaTiePolicy::FirstCanonical => ExtremaTiePolicy::FirstCanonical,
                    AuthoredExtremaTiePolicy::AllLocations => ExtremaTiePolicy::AllLocations,
                },
            });
            if !matches!(
                reduction,
                AuthoredSpatialReduction::Minimum | AuthoredSpatialReduction::Maximum
            ) {
                draft.spec.output_location = FieldLocation::Global;
            }
            Ok(draft)
        }
    }
}

fn base_spec(
    quantity: ObservationQuantity,
    location: FieldLocation,
    domain: AuthoredFieldDomain,
) -> ObservationSpec {
    ObservationSpec {
        observation_id: ObservationId(String::new()),
        quantity,
        value_shape: quantity.canonical_value_shape(),
        target: ObservationTarget::WholeModel,
        source_location: location,
        output_location: location,
        recovery: RecoveryPolicy::Native,
        component: ObservationComponent::All,
        frame: CoordinateFrame::Global,
        sign_convention: SignConvention::Native,
        complex_representation: match domain {
            AuthoredFieldDomain::Complex => ComplexRepresentation::Complex,
            AuthoredFieldDomain::Real | AuthoredFieldDomain::Statistical => {
                ComplexRepresentation::Real
            }
        },
        sampling: SamplingPolicy::Final,
        reduction: None,
        storage: ObservationStoragePolicy {
            precision: SignalPrecision::Real64,
            compression: CompressionPolicy::Lossless,
            retention: RetentionPolicy::Canonical,
            partial_access: PartialAccessPolicy::ManifestAndChunks,
        },
        purpose: ObservationPurpose::Publication,
        cost: ObservationCostPolicy::Automatic,
    }
}

fn apply_projection(
    spec: &mut ObservationSpec,
    projection: AuthoredFieldProjection,
) -> Result<(), AuthoredObservationError> {
    match projection {
        AuthoredFieldProjection::Magnitude => {
            if spec.value_shape == ObservationValueShape::Vector3 {
                spec.component = ObservationComponent::Magnitude;
            }
            if spec.complex_representation == ComplexRepresentation::Complex {
                spec.complex_representation = ComplexRepresentation::Magnitude;
            }
        }
        AuthoredFieldProjection::Component(component) => {
            spec.component = match component {
                AuthoredVectorComponent::X => ObservationComponent::X,
                AuthoredVectorComponent::Y => ObservationComponent::Y,
                AuthoredVectorComponent::Z => ObservationComponent::Z,
            };
        }
        AuthoredFieldProjection::Real => spec.complex_representation = ComplexRepresentation::Real,
        AuthoredFieldProjection::Imaginary => {
            spec.complex_representation = ComplexRepresentation::Imaginary;
        }
        AuthoredFieldProjection::Phase => {
            spec.complex_representation = ComplexRepresentation::PhaseRadians;
        }
        AuthoredFieldProjection::Principal(which) => {
            if !matches!(
                spec.quantity,
                ObservationQuantity::Stress | ObservationQuantity::Strain
            ) {
                return Err(AuthoredObservationError::InvalidProjection(
                    "principal requires stress or strain".into(),
                ));
            }
            spec.component = match which {
                AuthoredPrincipalSelection::Maximum => ObservationComponent::MaximumPrincipal,
                AuthoredPrincipalSelection::Minimum => ObservationComponent::MinimumPrincipal,
            };
        }
        AuthoredFieldProjection::VonMises => {
            if spec.quantity != ObservationQuantity::Stress {
                return Err(AuthoredObservationError::InvalidProjection(
                    "von_mises requires stress".into(),
                ));
            }
            spec.component = ObservationComponent::VonMises;
        }
    }
    Ok(())
}

fn region_target(source: &AuthoredRegionSource) -> Result<RegionTarget, AuthoredObservationError> {
    let AuthoredRegionSource::Geometry(region) = source else {
        return Err(AuthoredObservationError::UnrealizedRegionInput);
    };
    let cardinality = TargetCardinality::OneOrMore;
    Ok(match region.role {
        GeometryRegionRole::Vertex => RegionTarget {
            region_id: region.region_id.clone(),
            entity_dimension: EntityDimension::Point,
            expected_topology: ExpectedTopology::MeshNode,
            cardinality,
        },
        GeometryRegionRole::Edge => RegionTarget {
            region_id: region.region_id.clone(),
            entity_dimension: EntityDimension::Edge,
            expected_topology: ExpectedTopology::GeometryEntity,
            cardinality,
        },
        GeometryRegionRole::Face => RegionTarget::boundary(region.region_id.clone()),
        GeometryRegionRole::Solid => RegionTarget::volume(region.region_id.clone()),
        GeometryRegionRole::Cell => RegionTarget {
            region_id: region.region_id.clone(),
            entity_dimension: EntityDimension::Volume,
            expected_topology: ExpectedTopology::Cell,
            cardinality,
        },
    })
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AuthoredObservationError {
    #[error("output references unknown analysis step `{0}`")]
    UnknownStep(String),
    #[error("output references unknown or forward output `{0}`")]
    UnknownOutput(String),
    #[error("output projection is invalid: {0}")]
    InvalidProjection(String),
    #[error("output region model inputs must be realized before observation lowering")]
    UnrealizedRegionInput,
}
