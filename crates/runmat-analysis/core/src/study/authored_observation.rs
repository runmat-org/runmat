use std::collections::BTreeMap;

use runmat_geometry_core::GeometryRegionRole;

use crate::{
    AnalysisSequence, AuthoredExtremaTiePolicy, AuthoredFieldDomain, AuthoredFieldProjection,
    AuthoredOutput, AuthoredOutputExpression, AuthoredPortReduction, AuthoredPrincipalSelection,
    AuthoredRegionSource, AuthoredSeriesReduction, AuthoredSpatialReduction,
    AuthoredVectorComponent, ComplexRepresentation, CompressionPolicy, CoordinateFrame,
    EntityDimension, ExcitationTarget, ExpectedTopology, ExtremaTiePolicy, FieldLocation,
    ObservationComponent, ObservationCostPolicy, ObservationId, ObservationPurpose,
    ObservationQuantity, ObservationRequest, ObservationSpec, ObservationStoragePolicy,
    ObservationTarget, ObservationValueShape, PartialAccessPolicy, ProcedureStep, ProcedureStepId,
    RecoveryPolicy, ReductionPolicy, RegionTarget, RetentionPolicy, SamplingPolicy,
    SeriesReduction, SignConvention, SignalPrecision, SpatialReduction, TargetCardinality,
};

use super::authored_output::field_domain;

// Serialization supports borrowed resource profiling; import remains checked lowering.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
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
        draft.spec.storage = ObservationStoragePolicy {
            precision: output
                .policy
                .precision
                .unwrap_or(draft.spec.storage.precision),
            compression: output.policy.compression,
            retention: output.policy.retention.clone(),
            partial_access: output.policy.partial_access,
        };
        draft.spec.purpose = output.policy.purpose;
        let request = match draft.kind {
            ObservationKind::Field => ObservationRequest::Field(draft.spec.clone()),
            ObservationKind::Reduction => ObservationRequest::Reduction(draft.spec.clone()),
            ObservationKind::Probe => ObservationRequest::Probe(draft.spec.clone()),
            ObservationKind::History => ObservationRequest::History(draft.spec.clone()),
        };
        let step = steps
            .get(draft.step_id.as_str())
            .ok_or_else(|| AuthoredObservationError::UnknownStep(draft.step_id.clone()))?;
        crate::validate_authored_observation_request(step, &request)
            .map_err(|error| AuthoredObservationError::InvalidRequest(error.to_string()))?;
        requests.push(LoweredAuthoredObservation {
            step_id: ProcedureStepId(draft.step_id.clone()),
            request,
        });
        lowered.insert(output.output_id.clone(), draft);
    }
    Ok(requests)
}

/// Attach checked study outputs to the exact procedure steps that evaluate them.
///
/// Authored outputs remain the public declaration and identity authority. Procedure observations
/// are their executable form; keeping this bridge in analysis core prevents compilers and runtimes
/// from independently interpreting the output graph.
pub fn attach_authored_outputs(
    outputs: &[AuthoredOutput],
    sequence: &mut AnalysisSequence,
) -> Result<(), AuthoredObservationError> {
    crate::validate_authored_outputs(outputs, &sequence.steps)
        .map_err(|error| AuthoredObservationError::InvalidOutput(error.to_string()))?;
    let lowered = lower_authored_observations(outputs, &sequence.steps)?;
    let mut existing_ids = sequence
        .steps
        .iter()
        .flat_map(|step| step.observations.iter())
        .map(|request| request.observation_id().0.clone())
        .collect::<std::collections::BTreeSet<_>>();
    for output in &lowered {
        let observation_id = output.request.observation_id().0.as_str();
        if !existing_ids.insert(observation_id.to_string()) {
            return Err(AuthoredObservationError::DuplicateObservation(
                observation_id.to_string(),
            ));
        }
        let step = sequence
            .steps
            .iter_mut()
            .find(|step| step.step_id == output.step_id)
            .ok_or_else(|| AuthoredObservationError::UnknownStep(output.step_id.0.clone()))?;
        step.observations.push(output.request.clone());
    }
    crate::validate_analysis_sequence(sequence)
        .map(|_| ())
        .map_err(|error| AuthoredObservationError::InvalidRequest(error.to_string()))
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
    Probe,
    History,
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
                spec: base_spec(
                    *quantity,
                    *location,
                    field_domain(step.procedure, *quantity),
                ),
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
        AuthoredOutputExpression::Recovery {
            source,
            output_location,
            recovery,
        } => {
            let mut draft = lower_expression(source, steps, outputs)?;
            let projected_von_mises = draft.spec.quantity == ObservationQuantity::Stress
                && draft.spec.component == ObservationComponent::VonMises;
            let native_von_mises = draft.spec.quantity == ObservationQuantity::VonMisesStress
                && draft.spec.component == ObservationComponent::All;
            if !projected_von_mises && !native_von_mises {
                return Err(AuthoredObservationError::InvalidProjection(
                    "volume-weighted nodal recovery requires an element von_mises projection"
                        .into(),
                ));
            }
            draft.spec.quantity = ObservationQuantity::VonMisesStress;
            draft.spec.value_shape = ObservationValueShape::Scalar;
            draft.spec.component = ObservationComponent::All;
            draft.spec.output_location = *output_location;
            draft.spec.recovery = *recovery;
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
                    AuthoredSpatialReduction::NormalFlux => SpatialReduction::NormalFlux,
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
        AuthoredOutputExpression::PortReduction {
            reduction,
            source,
            first,
            second,
        } => {
            let mut draft = lower_expression(source, steps, outputs)?;
            draft.kind = ObservationKind::Reduction;
            draft.spec.target = ObservationTarget::RegionPair {
                first: region_target(first)?,
                second: region_target(second)?,
            };
            draft.spec.reduction = Some(ReductionPolicy {
                spatial: match reduction {
                    AuthoredPortReduction::MeanDifference => SpatialReduction::MeanDifference,
                    AuthoredPortReduction::TransmissionLoss => SpatialReduction::TransmissionLoss,
                },
                series: SeriesReduction::None,
                extrema_ties: ExtremaTiePolicy::FirstCanonical,
            });
            draft.spec.output_location = FieldLocation::Global;
            Ok(draft)
        }
        AuthoredOutputExpression::SpatialSelection { source, target } => {
            let mut draft = lower_expression(source, steps, outputs)?;
            draft.spec.target =
                ObservationTarget::Spatial(ExcitationTarget::Region(region_target(target)?));
            Ok(draft)
        }
        AuthoredOutputExpression::Interpolation { source, positions } => {
            let mut draft = lower_expression(source, steps, outputs)?;
            draft.kind = ObservationKind::Probe;
            draft.spec.target = match positions.as_slice() {
                [position] => ObservationTarget::Point {
                    position: *position,
                },
                positions => ObservationTarget::Path {
                    positions: positions.to_vec(),
                },
            };
            match (&draft.spec.target, draft.spec.source_location) {
                // An element field is piecewise per element, and a finite-volume cell field holds
                // the cell average: a point samples the native value of the element or cell that
                // contains it.
                (
                    ObservationTarget::Point { .. },
                    location @ (FieldLocation::Element | FieldLocation::Cell),
                ) => {
                    draft.spec.output_location = location;
                    draft.spec.recovery = RecoveryPolicy::Native;
                }
                _ => {
                    draft.spec.output_location = FieldLocation::Point;
                    draft.spec.recovery = RecoveryPolicy::ShapeFunctionInterpolation;
                }
            }
            Ok(draft)
        }
        AuthoredOutputExpression::Sampling {
            source, sampling, ..
        } => {
            let mut draft = lower_expression(source, steps, outputs)?;
            draft.kind = ObservationKind::History;
            draft.spec.sampling = sampling.clone();
            Ok(draft)
        }
        AuthoredOutputExpression::SeriesReduction {
            reduction,
            source,
            domain: _,
            ties,
        } => {
            let mut draft = lower_expression(source, steps, outputs)?;
            draft.kind = ObservationKind::History;
            let policy = draft.spec.reduction.get_or_insert(ReductionPolicy {
                spatial: SpatialReduction::None,
                series: SeriesReduction::None,
                extrema_ties: ExtremaTiePolicy::FirstCanonical,
            });
            policy.series = match reduction {
                AuthoredSeriesReduction::Minimum => SeriesReduction::Minimum,
                AuthoredSeriesReduction::Maximum => SeriesReduction::Maximum,
                AuthoredSeriesReduction::AbsoluteMaximum => SeriesReduction::AbsoluteMaximum,
                AuthoredSeriesReduction::Mean { weighting } => SeriesReduction::Mean {
                    weighting: *weighting,
                },
                AuthoredSeriesReduction::RootMeanSquare { weighting } => {
                    SeriesReduction::RootMeanSquare {
                        weighting: *weighting,
                    }
                }
                AuthoredSeriesReduction::Variance { weighting } => SeriesReduction::Variance {
                    weighting: *weighting,
                },
                AuthoredSeriesReduction::Envelope => SeriesReduction::Envelope,
            };
            policy.extrema_ties = match ties {
                AuthoredExtremaTiePolicy::FirstCanonical => ExtremaTiePolicy::FirstCanonical,
                AuthoredExtremaTiePolicy::AllLocations => ExtremaTiePolicy::AllLocations,
            };
            Ok(draft)
        }
    }
}

fn base_spec(
    quantity: ObservationQuantity,
    location: FieldLocation,
    domain: AuthoredFieldDomain,
) -> ObservationSpec {
    let precision = match domain {
        AuthoredFieldDomain::Complex => SignalPrecision::Complex128,
        AuthoredFieldDomain::Real | AuthoredFieldDomain::Statistical => SignalPrecision::Real64,
    };
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
            precision,
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
        AuthoredFieldProjection::Level => {
            if spec.quantity != ObservationQuantity::AcousticPressure
                || spec.complex_representation != ComplexRepresentation::Complex
            {
                return Err(AuthoredObservationError::InvalidProjection(
                    "level requires complex acoustic pressure".into(),
                ));
            }
            spec.complex_representation = ComplexRepresentation::SoundPressureLevel;
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
    spec.storage.precision = if spec.complex_representation == ComplexRepresentation::Complex {
        SignalPrecision::Complex128
    } else {
        SignalPrecision::Real64
    };
    Ok(())
}

fn region_target(source: &AuthoredRegionSource) -> Result<RegionTarget, AuthoredObservationError> {
    let AuthoredRegionSource::Geometry(region) = source else {
        return Err(AuthoredObservationError::UnrealizedRegionInput);
    };
    let cardinality = TargetCardinality::OneOrMore;
    Ok(match region.role {
        GeometryRegionRole::Vertex => RegionTarget::new(
            region.region_id.clone(),
            EntityDimension::Point,
            ExpectedTopology::MeshNode,
            cardinality,
        ),
        GeometryRegionRole::Edge => RegionTarget::new(
            region.region_id.clone(),
            EntityDimension::Edge,
            ExpectedTopology::GeometryEntity,
            cardinality,
        ),
        GeometryRegionRole::Face => RegionTarget::boundary(region.region_id.clone()),
        GeometryRegionRole::Solid => RegionTarget::volume(region.region_id.clone()),
        GeometryRegionRole::Cell => RegionTarget::new(
            region.region_id.clone(),
            EntityDimension::Volume,
            ExpectedTopology::Cell,
            cardinality,
        ),
    })
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AuthoredObservationError {
    #[error("output declaration is invalid: {0}")]
    InvalidOutput(String),
    #[error("output references unknown analysis step `{0}`")]
    UnknownStep(String),
    #[error("output references unknown or forward output `{0}`")]
    UnknownOutput(String),
    #[error("output observation `{0}` conflicts with an existing procedure observation")]
    DuplicateObservation(String),
    #[error("output projection is invalid: {0}")]
    InvalidProjection(String),
    #[error("output request is invalid: {0}")]
    InvalidRequest(String),
    #[error("output region model inputs must be realized before observation lowering")]
    UnrealizedRegionInput,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn displacement_output() -> AuthoredOutput {
        AuthoredOutput {
            output_id: "displacement".into(),
            expression: AuthoredOutputExpression::AnalysisField {
                step_id: "static".into(),
                quantity: ObservationQuantity::Displacement,
                location: FieldLocation::Node,
            },
            policy: crate::AuthoredOutputPolicy::default(),
        }
    }

    #[test]
    fn authored_outputs_attach_to_their_exact_procedure_step() {
        let mut sequence = AnalysisSequence::from_steps(
            "structural",
            vec![ProcedureStep::structural_static("static")],
        );

        attach_authored_outputs(&[displacement_output()], &mut sequence).unwrap();

        assert_eq!(sequence.steps[0].observations.len(), 1);
        assert_eq!(
            sequence.steps[0].observations[0].observation_id().0,
            "displacement"
        );
    }

    #[test]
    fn authored_outputs_cannot_replace_an_existing_observation() {
        let output = displacement_output();
        let mut sequence = AnalysisSequence::from_steps(
            "structural",
            vec![ProcedureStep::structural_static("static")],
        );
        sequence.steps[0].observations = vec![
            lower_authored_observations(std::slice::from_ref(&output), &sequence.steps)
                .unwrap()
                .remove(0)
                .request,
        ];

        assert!(matches!(
            attach_authored_outputs(&[output], &mut sequence),
            Err(AuthoredObservationError::DuplicateObservation(id)) if id == "displacement"
        ));
    }
    // witness: FEA-SHARED-OBSERVATIONS-001/quantity_location_and_cost_contract @ shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial; boundary: interpolate(...) of a finite-volume cell field at one point lowers to the containing cell's native value and passes authored validation; path sampling of cell fields is not covered
    #[test]
    fn point_interpolation_of_a_cell_field_samples_the_containing_cell() {
        let step = ProcedureStep::fluid_static("flow");
        let output = AuthoredOutput {
            output_id: "p_mid".into(),
            expression: AuthoredOutputExpression::Interpolation {
                source: Box::new(AuthoredOutputExpression::AnalysisField {
                    step_id: "flow".into(),
                    quantity: ObservationQuantity::FluidPressure,
                    location: FieldLocation::Cell,
                }),
                positions: vec![crate::Position::new([0.3, 0.0, 0.0])],
            },
            policy: crate::AuthoredOutputPolicy::default(),
        };
        let lowered =
            lower_authored_observations(std::slice::from_ref(&output), std::slice::from_ref(&step))
                .unwrap()
                .remove(0);
        let spec = lowered.request.spec().unwrap();
        assert_eq!(spec.source_location, FieldLocation::Cell);
        assert_eq!(spec.output_location, FieldLocation::Cell);
        assert_eq!(spec.recovery, RecoveryPolicy::Native);
        crate::validate_authored_observation_request(&step, &lowered.request).unwrap();
    }
}
