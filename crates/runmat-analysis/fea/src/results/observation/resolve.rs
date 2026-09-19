use std::collections::BTreeSet;

use runmat_analysis_core::{
    AnalysisModel, ComplexRepresentation, CompressionPolicy, EntityDimension, ExcitationTarget,
    ExpectedTopology, FieldLocation, ObservationComponent, ObservationCostEstimate,
    ObservationQuantity, ObservationRequest, ObservationSpec, ObservationTarget,
    ObservationValueShape, ProcedureStep, RecoveryPolicy, RetentionPolicy, SamplingPolicy,
    SeriesReduction, SignConvention, SpatialReduction, TargetCardinality,
};
use runmat_execution::Digest;
use runmat_meshing_core::{SolverMeshArtifact, StableDigest};

use crate::{
    constraints::CanonicalStructuralConstraintPlan,
    results::field::{
        RecoveredFieldDefinition, RecoveredFieldDerivation, RecoveredFieldEntityDomain,
        RecoveredFieldNumericRepresentation,
    },
    topology::AnalysisProjection,
};

use super::contract::{
    checked_bytes, derive_resolved_point_identity, materialized_field_location,
    observation_result_storage_component_count, shape_component_count,
    validate_output_numeric_contract, ObservationError, ObservationLimits, ObservationSampleBounds,
    ResolvedComponentProjection, ResolvedObservation, ResolvedObservationPlan,
    ResolvedObservationPlanIdentity, ResolvedObservationPoint, ResolvedObservationTarget,
};
use super::history::resolve_history_sample_bounds;
use super::topology::StructuralObservationTopology;

pub fn resolve_observation_plan(
    step: &ProcedureStep,
    projection: &AnalysisProjection,
    fields: &[RecoveredFieldDefinition],
    limits: ObservationLimits,
) -> Result<ResolvedObservationPlan, ObservationError> {
    let topology = StructuralObservationTopology::from_projection(projection);
    resolve_observation_plan_for_topology(step, &topology, None, fields, limits)
}

pub fn resolve_structural_observation_plan(
    step: &ProcedureStep,
    model: &AnalysisModel,
    mesh: Option<&SolverMeshArtifact>,
    constraint_plan: &CanonicalStructuralConstraintPlan,
    fields: &[RecoveredFieldDefinition],
    limits: ObservationLimits,
) -> Result<ResolvedObservationPlan, ObservationError> {
    let topology = StructuralObservationTopology::derive(model, mesh)?;
    constraint_plan
        .validate()
        .map_err(|error| ObservationError::Invalid(error.to_string()))?;
    if constraint_plan.topology != topology.identity {
        return Err(ObservationError::Identity(
            "constraint plan topology differs from the observation topology",
        ));
    }
    resolve_observation_plan_for_topology(step, &topology, Some(constraint_plan), fields, limits)
}

fn resolve_observation_plan_for_topology(
    step: &ProcedureStep,
    topology: &StructuralObservationTopology,
    constraint_plan: Option<&CanonicalStructuralConstraintPlan>,
    fields: &[RecoveredFieldDefinition],
    limits: ObservationLimits,
) -> Result<ResolvedObservationPlan, ObservationError> {
    if step.observations.len() > limits.maximum_observations {
        return Err(ObservationError::Limit("observation count"));
    }
    validate_field_catalog(fields)?;
    let observations = step
        .observations
        .iter()
        .map(|request| resolve_request(step, request, topology, constraint_plan, fields, limits))
        .collect::<Result<Vec<_>, _>>()?;
    let mut plan = ResolvedObservationPlan {
        schema_version: 1,
        identity: ResolvedObservationPlanIdentity(Digest::sha256([])),
        topology_digest: topology.digest(),
        step_id: step.step_id.clone(),
        independent_domain: step.independent_domain.clone(),
        observations,
    };
    plan.identity = plan.derive_identity()?;
    plan.validate_identity()?;
    Ok(plan)
}

fn validate_field_catalog(fields: &[RecoveredFieldDefinition]) -> Result<(), ObservationError> {
    let mut identities = std::collections::BTreeSet::new();
    for field in fields {
        if !identities.insert(field.identity()) {
            return Err(ObservationError::Invalid(
                "recovered-field declarations require unique identities".into(),
            ));
        }
        if !matches!(
            field.location(),
            FieldLocation::Node | FieldLocation::Element | FieldLocation::Constraint
        ) {
            return Err(ObservationError::Unsupported(
                "this evaluator accepts only recovered nodal, element, and constraint-group fields",
            ));
        }
        shape_component_count(field.value_shape())?;
    }
    Ok(())
}

fn resolve_request(
    step: &ProcedureStep,
    request: &ObservationRequest,
    topology: &StructuralObservationTopology,
    constraint_plan: Option<&CanonicalStructuralConstraintPlan>,
    fields: &[RecoveredFieldDefinition],
    limits: ObservationLimits,
) -> Result<ResolvedObservation, ObservationError> {
    runmat_analysis_core::validate_observation_sampling(step, request)
        .map_err(|error| ObservationError::Invalid(error.to_string()))?;
    let spec = match request {
        ObservationRequest::Field(spec)
        | ObservationRequest::History(spec)
        | ObservationRequest::Probe(spec)
        | ObservationRequest::Reduction(spec) => spec,
        ObservationRequest::Diagnostic(_) | ObservationRequest::Monitor(_) => {
            return Err(ObservationError::Unsupported(
                "diagnostics and monitors are evaluated by their owning pipeline",
            ));
        }
    };
    if !runmat_analysis_core::procedure_produces_observation_quantity(step, spec.quantity) {
        return Err(ObservationError::Invalid(
            "observation quantity is not produced by the procedure physics family".into(),
        ));
    }
    validate_supported_spec(request, spec)?;
    let source_field = bind_field(spec, fields)?;
    validate_numeric_representation(spec, &source_field)?;
    let target = resolve_target(
        request,
        spec,
        source_field.entity_domain(),
        topology,
        constraint_plan,
    )?;
    validate_resolved_spatial_reduction(spec, &source_field, &target)?;
    let component = resolve_component(spec, &source_field)?;
    let result_value_shape = spec.result_value_shape();
    let result_component_count = shape_component_count(&result_value_shape)?;
    let spatial_extrema = spec.reduction.as_ref().is_some_and(|reduction| {
        matches!(
            reduction.spatial,
            SpatialReduction::Minimum | SpatialReduction::Maximum
        )
    });
    let has_spatial_reduction = spec
        .reduction
        .as_ref()
        .is_some_and(|reduction| reduction.spatial != SpatialReduction::None);
    if spatial_extrema && result_component_count != 1 {
        return Err(ObservationError::Invalid(
            "spatial extrema require a scalar quantity or scalar component projection".into(),
        ));
    }
    let selected = target.selected_entity_count();
    if selected == 0 {
        return Err(target_error(
            spec,
            "target resolves to no admitted topology",
        ));
    }
    if selected > limits.maximum_selected_entities {
        return Err(ObservationError::Limit("selected topology entities"));
    }
    let result_storage_component_count = observation_result_storage_component_count(
        result_component_count,
        spec.complex_representation,
    )?;
    let value_count = match request {
        _ if spatial_extrema
            && spec.reduction.as_ref().is_some_and(|reduction| {
                reduction.extrema_ties == runmat_analysis_core::ExtremaTiePolicy::AllLocations
            }) =>
        {
            selected
                .checked_mul(result_storage_component_count)
                .ok_or(ObservationError::Limit("observation extrema value count"))?
        }
        ObservationRequest::Reduction(_) => result_storage_component_count,
        ObservationRequest::History(_) if has_spatial_reduction => result_storage_component_count,
        _ => selected
            .checked_mul(result_storage_component_count)
            .ok_or(ObservationError::Limit("observation value count"))?,
    };
    if value_count > limits.maximum_values {
        return Err(ObservationError::Limit("observation value count"));
    }
    let sample_bounds = match request {
        ObservationRequest::History(_) => resolve_history_sample_bounds(
            &spec.sampling,
            &spec.storage.retention,
            step.accuracy.resources.maximum_samples,
        )?,
        _ => ObservationSampleBounds {
            maximum_produced_samples: 1,
            maximum_retained_samples: 1,
        },
    };
    if spec
        .reduction
        .as_ref()
        .is_some_and(|reduction| reduction.series != SeriesReduction::None)
        && sample_bounds.maximum_retained_samples < sample_bounds.maximum_produced_samples
    {
        return Err(ObservationError::Invalid(
            "series reduction requires retention of every produced history sample".into(),
        ));
    }
    let retained_sample_count = u64::try_from(sample_bounds.maximum_retained_samples)
        .map_err(|_| ObservationError::Limit("observation retained sample count"))?;
    let result_component_count_u64 = u64::try_from(result_storage_component_count)
        .map_err(|_| ObservationError::Limit("observation result component count"))?;
    let work_entities = integration_work(&target, topology)?;
    let operation_count = u64::try_from(work_entities)
        .ok()
        .and_then(|value| value.checked_mul(result_component_count_u64))
        .and_then(|value| value.checked_mul(operation_multiplier(request, spec)))
        .and_then(|value| {
            value.checked_mul(u64::try_from(sample_bounds.maximum_produced_samples).ok()?)
        })
        .ok_or(ObservationError::Limit("observation operation count"))?;
    let value_bytes = checked_bytes(value_count)?;
    let retained_value_bytes = value_bytes
        .checked_mul(retained_sample_count)
        .ok_or(ObservationError::Limit("observation retained value bytes"))?;
    let series_output_bytes = match spec.reduction.as_ref().map(|reduction| reduction.series) {
        Some(SeriesReduction::Envelope) => value_bytes
            .checked_mul(2)
            .ok_or(ObservationError::Limit("series envelope bytes"))?,
        Some(series) if series != SeriesReduction::None => value_bytes,
        _ => 0,
    };
    let series_evidence_bytes = if series_output_bytes == 0 {
        0
    } else {
        retained_value_bytes
    };
    let source_work_bytes = checked_bytes(
        work_entities
            .checked_mul(
                source_field
                    .storage_component_count()
                    .map_err(|error| ObservationError::Invalid(error.to_string()))?,
            )
            .ok_or(ObservationError::Limit("observation working-set values"))?,
    )?;
    let stored_bytes = retained_value_bytes
        .checked_add(
            16_u64
                .checked_mul(1024)
                .and_then(|overhead| overhead.checked_mul(retained_sample_count))
                .ok_or(ObservationError::Limit("observation storage overhead"))?,
        )
        .and_then(|value| value.checked_add(series_output_bytes))
        .and_then(|value| value.checked_add(series_evidence_bytes))
        .ok_or(ObservationError::Limit("observation storage overhead"))?;
    let resolved_cost = ObservationCostEstimate {
        operation_count,
        peak_resident_bytes: source_work_bytes
            .checked_add(retained_value_bytes)
            .and_then(|value| value.checked_add(series_output_bytes))
            .and_then(|value| value.checked_add(series_evidence_bytes))
            .ok_or(ObservationError::Limit("observation working set"))?,
        stored_bytes,
        transfer_bytes: stored_bytes,
        collective_count: 0,
    };
    validate_cost(spec.cost, &resolved_cost, limits)?;
    Ok(ResolvedObservation {
        spec: spec.clone(),
        source_field_definition_identity: source_field.identity(),
        source_entity_domain: source_field.entity_domain(),
        target,
        component,
        result_value_shape,
        sample_bounds,
        resolved_cost,
    })
}

fn validate_supported_spec(
    request: &ObservationRequest,
    spec: &ObservationSpec,
) -> Result<(), ObservationError> {
    validate_output_numeric_contract(spec)?;
    let native_location =
        spec.source_location == spec.output_location && spec.recovery == RecoveryPolicy::Native;
    let reduced_location = spec.output_location == FieldLocation::Global
        && spec.recovery == RecoveryPolicy::Native
        && spec
            .reduction
            .as_ref()
            .is_some_and(|reduction| reduction.spatial != SpatialReduction::None);
    let solid_point_interpolation = spec.source_location == FieldLocation::Node
        && spec.output_location == FieldLocation::Point
        && spec.recovery == RecoveryPolicy::ShapeFunctionInterpolation
        && matches!(
            spec.target,
            ObservationTarget::Point { .. } | ObservationTarget::Path { .. }
        );
    let materialized_location_transfer = spec.source_location == FieldLocation::Element
        && spec.output_location == FieldLocation::Node
        && spec.recovery == RecoveryPolicy::VolumeWeightedAverage;
    if !native_location
        && !reduced_location
        && !solid_point_interpolation
        && !materialized_location_transfer
    {
        return Err(ObservationError::Unsupported(
            "field recovery and location transfer must be completed before observation evaluation",
        ));
    }
    if !matches!(
        spec.source_location,
        FieldLocation::Node | FieldLocation::Element | FieldLocation::Constraint
    ) {
        return Err(ObservationError::Unsupported(
            "this evaluator accepts only nodal, element, and constraint-group fields",
        ));
    }
    match request {
        ObservationRequest::History(_) if spec.sampling == SamplingPolicy::Final => {
            return Err(ObservationError::Invalid(
                "history requests require a sampled series rather than one final state".into(),
            ));
        }
        ObservationRequest::Field(_)
        | ObservationRequest::Probe(_)
        | ObservationRequest::Reduction(_)
            if spec.sampling != SamplingPolicy::Final =>
        {
            return Err(ObservationError::Unsupported(
                "sampled field, probe, and reduction series require a history request",
            ));
        }
        _ => {}
    }
    if !matches!(
        spec.storage.compression,
        CompressionPolicy::None | CompressionPolicy::Lossless
    ) {
        return Err(ObservationError::Unsupported(
            "error-bounded observation compression is not implemented",
        ));
    }
    match &spec.storage.retention {
        RetentionPolicy::LatestSamples { count } if *count < 1 => {
            return Err(ObservationError::Invalid(
                "retention count must be nonzero".into(),
            ));
        }
        RetentionPolicy::ExplicitSamples { maximum_count } if *maximum_count < 1 => {
            return Err(ObservationError::Invalid(
                "retention count must be nonzero".into(),
            ));
        }
        _ => {}
    }
    if let Some(reduction) = &spec.reduction {
        if matches!(spec.sampling, SamplingPolicy::Event(_))
            && reduction.series != SeriesReduction::None
        {
            return Err(ObservationError::Invalid(
                "event histories may be empty, so they cannot request a series reduction".into(),
            ));
        }
        if reduction.series != SeriesReduction::None
            && !matches!(request, ObservationRequest::History(_))
        {
            return Err(ObservationError::Invalid(
                "series reductions require a history request".into(),
            ));
        }
        if !matches!(
            reduction.spatial,
            SpatialReduction::None
                | SpatialReduction::Integral
                | SpatialReduction::Average
                | SpatialReduction::RootMeanSquare
                | SpatialReduction::EuclideanNorm
                | SpatialReduction::Minimum
                | SpatialReduction::Maximum
                | SpatialReduction::Resultant
                | SpatialReduction::Moment { .. }
        ) {
            return Err(ObservationError::Unsupported(
                "unsupported spatial reduction",
            ));
        }
    }
    match request {
        ObservationRequest::Field(_) if spec.reduction.is_some() => Err(ObservationError::Invalid(
            "field requests cannot apply reductions".into(),
        )),
        ObservationRequest::Probe(_) if spec.reduction.is_some() => Err(ObservationError::Invalid(
            "probe requests cannot apply reductions".into(),
        )),
        ObservationRequest::Reduction(_)
            if spec
                .reduction
                .as_ref()
                .is_none_or(|reduction| reduction.spatial == SpatialReduction::None) =>
        {
            Err(ObservationError::Invalid(
                "reduction requests require a spatial reduction".into(),
            ))
        }
        ObservationRequest::History(ObservationSpec {
            reduction: Some(_), ..
        }) if spec.reduction.as_ref().is_none_or(|reduction| {
            reduction.spatial == SpatialReduction::None && reduction.series == SeriesReduction::None
        }) =>
        {
            Err(ObservationError::Invalid(
                "history reduction requests require a spatial or series reduction".into(),
            ))
        }
        _ => Ok(()),
    }
}

fn validate_resolved_spatial_reduction(
    spec: &ObservationSpec,
    source: &RecoveredFieldDefinition,
    target: &ResolvedObservationTarget,
) -> Result<(), ObservationError> {
    let Some(operation) = spec.reduction.as_ref().map(|reduction| reduction.spatial) else {
        return Ok(());
    };
    if matches!(
        operation,
        SpatialReduction::Integral
            | SpatialReduction::Average
            | SpatialReduction::RootMeanSquare
            | SpatialReduction::EuclideanNorm
            | SpatialReduction::Resultant
            | SpatialReduction::Moment { .. }
    ) && spec.output_location != FieldLocation::Global
    {
        return Err(ObservationError::Invalid(
            "integrated spatial reductions require global output location".into(),
        ));
    }
    if !matches!(
        operation,
        SpatialReduction::Resultant | SpatialReduction::Moment { .. }
    ) {
        return Ok(());
    }
    if spec.component != ObservationComponent::All {
        return Err(ObservationError::Invalid(
            "resultant and moment reductions require all source components".into(),
        ));
    }
    let representation_is_valid = matches!(
        (source.numeric_representation(), spec.complex_representation),
        (
            RecoveredFieldNumericRepresentation::Real,
            ComplexRepresentation::Real
        ) | (
            RecoveredFieldNumericRepresentation::ComplexInterleaved,
            ComplexRepresentation::Real
                | ComplexRepresentation::Imaginary
                | ComplexRepresentation::Complex
        )
    );
    if !representation_is_valid {
        return Err(ObservationError::Unsupported(
            "resultants require real values, one Cartesian phasor component, or the complete complex phasor",
        ));
    }
    let source_shape = source.value_shape();
    match (source_shape, target) {
        (ObservationValueShape::Vector3, ResolvedObservationTarget::Nodes { .. })
            if matches!(
                spec.quantity,
                ObservationQuantity::ReactionForce | ObservationQuantity::ReactionMoment
            ) => {}
        (ObservationValueShape::Vector3, ResolvedObservationTarget::ConstraintGroups { .. })
            if matches!(
                spec.quantity,
                ObservationQuantity::ReactionForce | ObservationQuantity::ReactionMoment
            ) => {}
        (ObservationValueShape::Vector3, ResolvedObservationTarget::Volume { .. })
        | (ObservationValueShape::Vector3, ResolvedObservationTarget::Boundary { .. }) => {}
        (ObservationValueShape::SymmetricTensor3, ResolvedObservationTarget::Boundary { .. })
            if spec.quantity == ObservationQuantity::Stress
                && matches!(
                    spec.sign_convention,
                    SignConvention::Native | SignConvention::TensionPositive
                ) => {}
        (ObservationValueShape::Scalar, ResolvedObservationTarget::Boundary { .. })
            if matches!(
                spec.quantity,
                ObservationQuantity::FluidPressure
                    | ObservationQuantity::AcousticPressure
                    | ObservationQuantity::ContactPressure
            ) && matches!(
                spec.sign_convention,
                SignConvention::OutwardPositive | SignConvention::InwardPositive
            ) => {}
        _ => {
            return Err(ObservationError::Unsupported(
                "resultants require a vector volume/surface field, boundary stress, or explicitly oriented boundary pressure",
            ))
        }
    }
    if matches!(operation, SpatialReduction::Moment { .. })
        && spec.quantity == ObservationQuantity::ReactionMoment
    {
        return Err(ObservationError::Invalid(
            "a moment reduction cannot take a second moment of reaction moments".into(),
        ));
    }
    Ok(())
}

fn validate_numeric_representation(
    spec: &ObservationSpec,
    source: &RecoveredFieldDefinition,
) -> Result<(), ObservationError> {
    match source.numeric_representation() {
        RecoveredFieldNumericRepresentation::Real
            if spec.complex_representation == ComplexRepresentation::Real =>
        {
            Ok(())
        }
        RecoveredFieldNumericRepresentation::ComplexInterleaved => Ok(()),
        RecoveredFieldNumericRepresentation::Real => Err(ObservationError::Invalid(
            "a real recovered field cannot provide imaginary, phase, or complex values".into(),
        )),
    }
}

fn bind_field(
    spec: &ObservationSpec,
    fields: &[RecoveredFieldDefinition],
) -> Result<RecoveredFieldDefinition, ObservationError> {
    let matches = fields
        .iter()
        .filter(|field| {
            let derivation_matches = if spec.recovery == RecoveryPolicy::ShapeFunctionInterpolation
                && spec.output_location == FieldLocation::Point
            {
                field.derivation() == RecoveredFieldDerivation::Native
                    && field.location() == spec.source_location
            } else {
                field.satisfies_recovery(spec.source_location, spec.output_location, spec.recovery)
            };
            field.quantity() == spec.quantity
                && derivation_matches
                && field.value_shape() == &spec.value_shape
        })
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [field] => Ok((*field).clone()),
        [] => Err(ObservationError::Field {
            observation_id: spec.observation_id.0.clone(),
            reason: "no declaration matches quantity, location, and value shape".into(),
        }),
        _ => Err(ObservationError::Field {
            observation_id: spec.observation_id.0.clone(),
            reason:
                "multiple declarations match; recovery must provide an unambiguous field catalog"
                    .into(),
        }),
    }
}

fn resolve_target(
    request: &ObservationRequest,
    spec: &ObservationSpec,
    field_domain: RecoveredFieldEntityDomain,
    topology: &StructuralObservationTopology,
    constraint_plan: Option<&CanonicalStructuralConstraintPlan>,
) -> Result<ResolvedObservationTarget, ObservationError> {
    if spec.reduction.as_ref().is_some_and(|reduction| {
        matches!(
            reduction.spatial,
            SpatialReduction::Minimum | SpatialReduction::Maximum
        )
    }) {
        return resolve_extrema_target(spec, field_domain, topology);
    }
    match request {
        ObservationRequest::Probe(_) if spec.output_location == FieldLocation::Point => {
            resolve_interpolated_points(spec, field_domain, topology)
        }
        ObservationRequest::Probe(_) => resolve_probe(spec, field_domain, topology),
        ObservationRequest::Reduction(_) => {
            resolve_reduction_target(spec, field_domain, topology, constraint_plan)
        }
        ObservationRequest::Field(_) => resolve_field_target(spec, field_domain, topology),
        ObservationRequest::History(_)
            if spec
                .reduction
                .as_ref()
                .is_some_and(|reduction| reduction.spatial != SpatialReduction::None) =>
        {
            resolve_reduction_target(spec, field_domain, topology, constraint_plan)
        }
        ObservationRequest::History(_)
            if matches!(spec.target, ObservationTarget::Path { .. })
                || spec.output_location == FieldLocation::Point =>
        {
            resolve_interpolated_points(spec, field_domain, topology)
        }
        ObservationRequest::History(_)
            if matches!(spec.target, ObservationTarget::Point { .. }) =>
        {
            resolve_probe(spec, field_domain, topology)
        }
        ObservationRequest::History(_) => resolve_field_target(spec, field_domain, topology),
        _ => Err(ObservationError::Unsupported(
            "unsupported observation request",
        )),
    }
}

fn resolve_extrema_target(
    spec: &ObservationSpec,
    field_domain: RecoveredFieldEntityDomain,
    topology: &StructuralObservationTopology,
) -> Result<ResolvedObservationTarget, ObservationError> {
    if matches!(spec.target, ObservationTarget::Path { .. })
        || spec.output_location == FieldLocation::Point
    {
        return resolve_interpolated_points(spec, field_domain, topology);
    }
    resolve_field_target(spec, field_domain, topology)
}

fn resolve_interpolated_points(
    spec: &ObservationSpec,
    field_domain: RecoveredFieldEntityDomain,
    topology: &StructuralObservationTopology,
) -> Result<ResolvedObservationTarget, ObservationError> {
    if spec.source_location != FieldLocation::Node
        || spec.output_location != FieldLocation::Point
        || field_domain != RecoveredFieldEntityDomain::StructuralNodes
    {
        return Err(target_error(
            spec,
            "finite-element point interpolation requires a nodal structural field and point output",
        ));
    }
    let positions = match &spec.target {
        ObservationTarget::Point { position } => std::slice::from_ref(position),
        ObservationTarget::Path { positions } => positions.as_slice(),
        _ => {
            return Err(target_error(
                spec,
                "point interpolation requires a point or path target",
            ))
        }
    };
    let projection = topology
        .solid_projection
        .as_ref()
        .ok_or(ObservationError::Unsupported(
            "point and path interpolation require admitted solid topology",
        ))?;
    let samples = positions
        .iter()
        .enumerate()
        .map(|(ordinal, position)| {
            let location = projection
                .locate_volume_point(position.meters)
                .ok_or_else(|| {
                    target_error(spec, "point lies outside the admitted solid topology")
                })?;
            let ordinal = u64::try_from(ordinal)
                .map_err(|_| ObservationError::Limit("path sample ordinal"))?;
            let mut sample = ResolvedObservationPoint {
                identity: StableDigest::ZERO,
                ordinal,
                position_m: location.position_m,
                source_element_index: location.element_index,
                source_element_identity: location.element_identity,
                source_node_indices: location.node_indices,
                source_node_identities: location.node_identities,
                shape_values: location.shape_values,
            };
            sample.identity = derive_resolved_point_identity(topology.digest(), &sample)?;
            Ok(sample)
        })
        .collect::<Result<Vec<_>, ObservationError>>()?;
    Ok(ResolvedObservationTarget::Points { samples })
}

fn resolve_probe(
    spec: &ObservationSpec,
    field_domain: RecoveredFieldEntityDomain,
    topology: &StructuralObservationTopology,
) -> Result<ResolvedObservationTarget, ObservationError> {
    let ObservationTarget::Point { position } = spec.target else {
        return Err(target_error(
            spec,
            "probe requests require an exact point target",
        ));
    };
    match materialized_field_location(spec) {
        FieldLocation::Node => {
            let matches = topology
                .nodes
                .iter()
                .enumerate()
                .filter(|(_, node)| node.coordinates_m == position.meters)
                .collect::<Vec<_>>();
            match matches.as_slice() {
                [(index, node)] => Ok(ResolvedObservationTarget::Nodes {
                    indices: vec![*index],
                    identities: vec![node.identity],
                }),
                [] => Err(target_error(
                    spec,
                    "point does not exactly coincide with an admitted mesh node",
                )),
                _ => Err(target_error(
                    spec,
                    "point matches multiple admitted mesh nodes",
                )),
            }
        }
        FieldLocation::Element => {
            let elements = topology.elements(field_domain)?;
            let matches = elements
                .iter()
                .enumerate()
                .filter(|(_, element)| element.centroid_m == position.meters)
                .collect::<Vec<_>>();
            match matches.as_slice() {
                [(index, element)] => Ok(ResolvedObservationTarget::Elements {
                    domain: field_domain,
                    indices: vec![*index],
                    identities: vec![element.identity],
                }),
                [] => Err(target_error(
                    spec,
                    "point does not exactly coincide with an admitted element centroid",
                )),
                _ => Err(target_error(
                    spec,
                    "point matches multiple element centroids",
                )),
            }
        }
        _ => Err(ObservationError::Unsupported(
            "this evaluator accepts only nodal and element probes",
        )),
    }
}

fn resolve_field_target(
    spec: &ObservationSpec,
    field_domain: RecoveredFieldEntityDomain,
    topology: &StructuralObservationTopology,
) -> Result<ResolvedObservationTarget, ObservationError> {
    if matches!(spec.target, ObservationTarget::Path { .. })
        || spec.output_location == FieldLocation::Point
    {
        return resolve_interpolated_points(spec, field_domain, topology);
    }
    match &spec.target {
        ObservationTarget::WholeModel
        | ObservationTarget::Spatial(ExcitationTarget::WholeModel) => {
            match materialized_field_location(spec) {
                FieldLocation::Node => nodes_target((0..topology.nodes.len()).collect(), topology),
                FieldLocation::Element => {
                    let elements = topology.elements(field_domain)?;
                    elements_target((0..elements.len()).collect(), field_domain, topology)
                }
                _ => Err(ObservationError::Unsupported("unsupported field location")),
            }
        }
        ObservationTarget::Spatial(ExcitationTarget::Region(region)) => {
            match (
                region.entity_dimension,
                region.expected_topology,
                materialized_field_location(spec),
            ) {
                (EntityDimension::Point, ExpectedTopology::MeshNode, FieldLocation::Node) => {
                    let id = parse_node_selector(&region.region_id, spec)?;
                    let index = topology.node_index(id)?.ok_or_else(|| {
                        target_error(spec, "structural-node selector does not resolve")
                    })?;
                    enforce_cardinality(region.cardinality, 1, spec)?;
                    nodes_target(vec![index], topology)
                }
                (EntityDimension::Face, ExpectedTopology::BoundaryFace, FieldLocation::Node) => {
                    let projection = solid_projection(topology, spec)?;
                    let faces = matching_faces(region, projection);
                    enforce_cardinality(region.cardinality, faces.len(), spec)?;
                    let indices = faces
                        .into_iter()
                        .flat_map(|index| {
                            projection.boundary_faces[index]
                                .node_indices
                                .iter()
                                .copied()
                        })
                        .collect::<BTreeSet<_>>()
                        .into_iter()
                        .collect();
                    nodes_target(indices, topology)
                }
                (
                    EntityDimension::Volume,
                    ExpectedTopology::MeshElement,
                    FieldLocation::Element,
                ) => {
                    let indices = matching_elements(region, field_domain, topology)?;
                    enforce_cardinality(region.cardinality, indices.len(), spec)?;
                    elements_target(indices, field_domain, topology)
                }
                (EntityDimension::Volume, ExpectedTopology::MeshElement, FieldLocation::Node) => {
                    let projection = solid_projection(topology, spec)?;
                    let elements = matching_elements(
                        region,
                        RecoveredFieldEntityDomain::SolidElements,
                        topology,
                    )?;
                    enforce_cardinality(region.cardinality, elements.len(), spec)?;
                    let indices = elements
                        .into_iter()
                        .flat_map(|index| projection.elements[index].node_indices.iter().copied())
                        .collect::<BTreeSet<_>>()
                        .into_iter()
                        .collect();
                    nodes_target(indices, topology)
                }
                _ => Err(target_error(
                    spec,
                    "target topology is incompatible with the recovered field location",
                )),
            }
        }
        _ => Err(target_error(
            spec,
            "field requests support whole-model and admitted region targets",
        )),
    }
}

fn resolve_reduction_target(
    spec: &ObservationSpec,
    field_domain: RecoveredFieldEntityDomain,
    topology: &StructuralObservationTopology,
    constraint_plan: Option<&CanonicalStructuralConstraintPlan>,
) -> Result<ResolvedObservationTarget, ObservationError> {
    if field_domain == RecoveredFieldEntityDomain::ConstraintGroups {
        if spec.source_location != FieldLocation::Constraint
            || !matches!(
                spec.quantity,
                ObservationQuantity::ReactionForce | ObservationQuantity::ReactionMoment
            )
            || !matches!(
                spec.reduction.as_ref().map(|reduction| reduction.spatial),
                Some(SpatialReduction::Resultant)
            )
        {
            return Err(ObservationError::Invalid(
                "constraint-group fields support reaction resultants only".into(),
            ));
        }
        let ObservationTarget::Constraint { constraint_id } = &spec.target else {
            return Err(target_error(
                spec,
                "constraint-group reaction fields require one authored constraint target",
            ));
        };
        let plan = constraint_plan.ok_or(ObservationError::Unsupported(
            "constraint-target reactions require the canonical executed constraint plan",
        ))?;
        let (index, group) = plan
            .groups
            .iter()
            .enumerate()
            .find(|(_, group)| group.constraint_id == *constraint_id)
            .ok_or_else(|| {
                target_error(spec, "constraint target is absent from the canonical plan")
            })?;
        return Ok(ResolvedObservationTarget::ConstraintGroups {
            indices: vec![index],
            identities: vec![StableDigest::from_bytes(*group.source_digest.bytes())],
        });
    }
    if !matches!(
        field_domain,
        RecoveredFieldEntityDomain::StructuralNodes | RecoveredFieldEntityDomain::SolidElements
    ) {
        return Err(ObservationError::Unsupported(
            "beam and shell reductions require formulation-owned integration rules",
        ));
    }
    if matches!(
        spec.quantity,
        ObservationQuantity::ReactionForce | ObservationQuantity::ReactionMoment
    ) {
        if spec.source_location != FieldLocation::Node
            || field_domain != RecoveredFieldEntityDomain::StructuralNodes
        {
            return Err(ObservationError::Invalid(
                "reaction resultants require a nodal structural reaction field".into(),
            ));
        }
        return match &spec.target {
            ObservationTarget::WholeModel
            | ObservationTarget::Spatial(ExcitationTarget::WholeModel) => {
                nodes_target((0..topology.nodes.len()).collect(), topology)
            }
            ObservationTarget::Spatial(ExcitationTarget::Region(region))
                if region.entity_dimension == EntityDimension::Point
                    && region.expected_topology == ExpectedTopology::MeshNode =>
            {
                let id = parse_node_selector(&region.region_id, spec)?;
                let index = topology.node_index(id)?.ok_or_else(|| {
                    target_error(spec, "structural-node selector does not resolve")
                })?;
                enforce_cardinality(region.cardinality, 1, spec)?;
                nodes_target(vec![index], topology)
            }
            ObservationTarget::Constraint { .. } => Err(ObservationError::Unsupported(
                "constraint-target reaction resultants require the canonical executed constraint plan",
            )),
            _ => Err(target_error(
                spec,
                "reaction resultants support the whole model or an exact structural node",
            )),
        };
    }
    let projection = solid_projection(topology, spec)?;
    match &spec.target {
        ObservationTarget::WholeModel
        | ObservationTarget::Spatial(ExcitationTarget::WholeModel) => {
            Ok(ResolvedObservationTarget::Volume {
                element_indices: (0..projection.elements.len()).collect(),
                element_identities: projection
                    .elements
                    .iter()
                    .map(|element| element.stable_identity)
                    .collect(),
            })
        }
        ObservationTarget::Spatial(ExcitationTarget::Region(region)) => {
            match (region.entity_dimension, region.expected_topology) {
                (EntityDimension::Volume, ExpectedTopology::MeshElement) => {
                    let element_indices = matching_elements(
                        region,
                        RecoveredFieldEntityDomain::SolidElements,
                        topology,
                    )?;
                    enforce_cardinality(region.cardinality, element_indices.len(), spec)?;
                    let element_identities = element_indices
                        .iter()
                        .map(|index| projection.elements[*index].stable_identity)
                        .collect();
                    Ok(ResolvedObservationTarget::Volume {
                        element_indices,
                        element_identities,
                    })
                }
                (EntityDimension::Face, ExpectedTopology::BoundaryFace)
                    if materialized_field_location(spec) == FieldLocation::Node =>
                {
                    let face_indices = matching_faces(region, projection);
                    enforce_cardinality(region.cardinality, face_indices.len(), spec)?;
                    let face_identities = face_indices
                        .iter()
                        .map(|index| projection.boundary_faces[*index].stable_identity)
                        .collect();
                    Ok(ResolvedObservationTarget::Boundary {
                        face_indices,
                        face_identities,
                    })
                }
                _ => Err(target_error(
                    spec,
                    "reductions support admitted volumes and nodal fields on admitted boundaries",
                )),
            }
        }
        _ => Err(target_error(
            spec,
            "reductions support whole-model and admitted region targets",
        )),
    }
}

fn nodes_target(
    indices: Vec<usize>,
    topology: &StructuralObservationTopology,
) -> Result<ResolvedObservationTarget, ObservationError> {
    let identities =
        indices
            .iter()
            .map(|index| {
                topology.nodes.get(*index).map(|node| node.identity).ok_or(
                    ObservationError::Identity("resolved node target exceeds structural topology"),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
    Ok(ResolvedObservationTarget::Nodes {
        indices,
        identities,
    })
}

fn elements_target(
    indices: Vec<usize>,
    domain: RecoveredFieldEntityDomain,
    topology: &StructuralObservationTopology,
) -> Result<ResolvedObservationTarget, ObservationError> {
    let elements = topology.elements(domain)?;
    let identities = indices
        .iter()
        .map(|index| {
            elements
                .get(*index)
                .map(|element| element.identity)
                .ok_or(ObservationError::Identity(
                    "resolved element target exceeds structural topology",
                ))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ResolvedObservationTarget::Elements {
        domain,
        indices,
        identities,
    })
}

fn matching_elements(
    target: &runmat_analysis_core::RegionTarget,
    domain: RecoveredFieldEntityDomain,
    topology: &StructuralObservationTopology,
) -> Result<Vec<usize>, ObservationError> {
    if domain == RecoveredFieldEntityDomain::SolidElements {
        if let Some(projection) = &topology.solid_projection {
            return Ok(projection
                .elements
                .iter()
                .enumerate()
                .filter_map(|(index, element)| {
                    target
                        .matches_persistent_entity(&element.region_id)
                        .then_some(index)
                })
                .collect());
        }
    }
    Ok(topology
        .elements(domain)?
        .iter()
        .enumerate()
        .filter_map(|(index, element)| (element.region_id == target.region_id).then_some(index))
        .collect())
}

fn solid_projection<'a>(
    topology: &'a StructuralObservationTopology,
    spec: &ObservationSpec,
) -> Result<&'a AnalysisProjection, ObservationError> {
    topology.solid_projection.as_ref().ok_or_else(|| {
        target_error(
            spec,
            "volume and boundary observations require admitted solid topology",
        )
    })
}

fn matching_faces(
    target: &runmat_analysis_core::RegionTarget,
    projection: &AnalysisProjection,
) -> Vec<usize> {
    projection
        .boundary_faces
        .iter()
        .enumerate()
        .filter_map(|(index, face)| {
            face.provenance
                .iter()
                .any(|entity| target.matches_persistent_entity(entity))
                .then_some(index)
        })
        .collect()
}

fn parse_node_selector(selector: &str, spec: &ObservationSpec) -> Result<u64, ObservationError> {
    selector
        .strip_prefix("node:")
        .and_then(|value| value.parse().ok())
        .ok_or_else(|| target_error(spec, "mesh-node targets require a `node:<id>` selector"))
}

fn enforce_cardinality(
    cardinality: TargetCardinality,
    count: usize,
    spec: &ObservationSpec,
) -> Result<(), ObservationError> {
    if count == 0 {
        Err(target_error(
            spec,
            "region resolves to no admitted topology",
        ))
    } else if cardinality == TargetCardinality::One && count != 1 {
        Err(target_error(
            spec,
            "region cardinality requires exactly one matched entity",
        ))
    } else {
        Ok(())
    }
}

fn resolve_component(
    spec: &ObservationSpec,
    field: &RecoveredFieldDefinition,
) -> Result<ResolvedComponentProjection, ObservationError> {
    let source = shape_component_count(field.value_shape())?;
    match spec.component {
        ObservationComponent::All => Ok(ResolvedComponentProjection::All),
        ObservationComponent::X if source == 3 => {
            Ok(ResolvedComponentProjection::Cartesian { axis: 0 })
        }
        ObservationComponent::Y if source == 3 => {
            Ok(ResolvedComponentProjection::Cartesian { axis: 1 })
        }
        ObservationComponent::Z if source == 3 => {
            Ok(ResolvedComponentProjection::Cartesian { axis: 2 })
        }
        ObservationComponent::Magnitude if source == 3 => {
            Ok(ResolvedComponentProjection::Magnitude3)
        }
        ObservationComponent::VonMises if source == 6 => {
            Ok(ResolvedComponentProjection::VonMisesSymmetricTensor)
        }
        ObservationComponent::MaximumPrincipal if source == 6 => {
            Ok(ResolvedComponentProjection::MaximumPrincipalSymmetricTensor)
        }
        ObservationComponent::MinimumPrincipal if source == 6 => {
            Ok(ResolvedComponentProjection::MinimumPrincipalSymmetricTensor)
        }
        _ => Err(ObservationError::Unsupported(
            "this evaluator does not support the requested component for the recovered value shape",
        )),
    }
}

fn integration_work(
    target: &ResolvedObservationTarget,
    topology: &StructuralObservationTopology,
) -> Result<usize, ObservationError> {
    match target {
        ResolvedObservationTarget::Nodes { indices, .. }
        | ResolvedObservationTarget::Elements { indices, .. }
        | ResolvedObservationTarget::ConstraintGroups { indices, .. } => Ok(indices.len()),
        ResolvedObservationTarget::Points { samples } => {
            samples.iter().try_fold(0usize, |work, sample| {
                work.checked_add(sample.source_node_indices.len().saturating_mul(8))
                    .ok_or(ObservationError::Limit("point interpolation work"))
            })
        }
        ResolvedObservationTarget::Volume {
            element_indices, ..
        } => {
            let projection =
                topology
                    .solid_projection
                    .as_ref()
                    .ok_or(ObservationError::Unsupported(
                        "volume reduction work requires admitted solid topology",
                    ))?;
            element_indices.iter().try_fold(0usize, |work, index| {
                projection
                    .elements
                    .get(*index)
                    .and_then(|element| {
                        work.checked_add(element.node_indices.len().saturating_mul(8))
                    })
                    .ok_or(ObservationError::Limit("volume reduction work"))
            })
        }
        ResolvedObservationTarget::Boundary { face_indices, .. } => {
            let projection =
                topology
                    .solid_projection
                    .as_ref()
                    .ok_or(ObservationError::Unsupported(
                        "boundary reduction work requires admitted solid topology",
                    ))?;
            face_indices.iter().try_fold(0usize, |work, index| {
                projection
                    .boundary_faces
                    .get(*index)
                    .and_then(|face| work.checked_add(face.node_indices.len().saturating_mul(8)))
                    .ok_or(ObservationError::Limit("boundary reduction work"))
            })
        }
    }
}

fn operation_multiplier(request: &ObservationRequest, spec: &ObservationSpec) -> u64 {
    if spec
        .reduction
        .as_ref()
        .is_some_and(|reduction| reduction.series != SeriesReduction::None)
    {
        return 16;
    }
    match request {
        ObservationRequest::Reduction(_)
        | ObservationRequest::History(ObservationSpec {
            reduction: Some(_), ..
        }) => match spec.reduction.as_ref().map(|reduction| reduction.spatial) {
            Some(SpatialReduction::RootMeanSquare) => 12,
            Some(SpatialReduction::Integral | SpatialReduction::Average) => 8,
            _ => 3,
        },
        _ => 2,
    }
}

fn validate_cost(
    authored: runmat_analysis_core::ObservationCostPolicy,
    resolved: &ObservationCostEstimate,
    limits: ObservationLimits,
) -> Result<(), ObservationError> {
    if let Some(authored) = authored.upper_bound() {
        if authored.collective_count != 0 {
            return Err(ObservationError::Unsupported(
                "serial observation evaluation cannot promise collective operations",
            ));
        }
        if authored.operation_count < resolved.operation_count
            || authored.peak_resident_bytes < resolved.peak_resident_bytes
            || authored.stored_bytes < resolved.stored_bytes
            || authored.transfer_bytes < resolved.transfer_bytes
        {
            return Err(ObservationError::Invalid(
                "observation cost upper bound understates the resolved serial plan".into(),
            ));
        }
        if authored.operation_count > limits.maximum_operation_count
            || authored.peak_resident_bytes > limits.maximum_peak_resident_bytes
            || authored.stored_bytes > limits.maximum_stored_bytes
            || authored.transfer_bytes > limits.maximum_transfer_bytes
        {
            return Err(ObservationError::Limit("observation cost upper bound"));
        }
    }
    if resolved.operation_count > limits.maximum_operation_count {
        return Err(ObservationError::Limit("operation count"));
    }
    if resolved.peak_resident_bytes > limits.maximum_peak_resident_bytes {
        return Err(ObservationError::Limit("peak resident bytes"));
    }
    if resolved.stored_bytes > limits.maximum_stored_bytes {
        return Err(ObservationError::Limit("stored bytes"));
    }
    if resolved.transfer_bytes > limits.maximum_transfer_bytes {
        return Err(ObservationError::Limit("transfer bytes"));
    }
    Ok(())
}

fn target_error(spec: &ObservationSpec, reason: impl Into<String>) -> ObservationError {
    ObservationError::Target {
        observation_id: spec.observation_id.0.clone(),
        reason: reason.into(),
    }
}
