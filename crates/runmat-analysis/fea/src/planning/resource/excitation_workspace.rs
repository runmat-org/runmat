//! Sampling-free active-field workspace, shared by structural execution phases.
use super::{ByteCount, Count, ExcitationWorkspaceResourceFacts, ResourcePlanningError};
use runmat_analysis_core::{
    AnalysisModel, ExcitationAction, FieldLocation, PhysicalSignal, ProcedureStep, SignalKnotValue,
    SignalOrdinate, SignalRepresentation, SignalShape, SignalValue,
};
use runmat_execution::Digest;
use runmat_meshing_core::SolverMeshArtifact;

pub(super) fn derive_excitation_workspace(
    model: &AnalysisModel,
    step: &ProcedureStep,
    mesh: Option<&SolverMeshArtifact>,
) -> Result<ExcitationWorkspaceResourceFacts, ResourcePlanningError> {
    let mut facts = ExcitationWorkspaceResourceFacts::default();
    let mut frozen_bytes = 0_u64;
    let mut peak_scratch = 0_u64;
    for selected in crate::excitation::active_actions(model, Some(step))
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?
    {
        let signal = model
            .signals
            .iter()
            .find(|signal| signal.signal_id() == &selected.action.excitation().signal_id)
            .ok_or_else(|| {
                ResourcePlanningError::Invalid("active excitation names an absent signal".into())
            })?;
        let SignalShape::TopologyField {
            topology_id,
            location,
            component_count,
        } = signal.shape()
        else {
            continue;
        };
        if !matches!(selected.action, ExcitationAction::BodyForce(_))
            || *location != FieldLocation::Node
            || *component_count != 3
        {
            return Err(ResourcePlanningError::Invalid("structural field workspace requires the supported nodal body-density vector envelope".into()));
        }
        let mesh = mesh.ok_or_else(|| {
            ResourcePlanningError::Invalid(
                "nodal field workspace requires the canonical solver mesh".into(),
            )
        })?;
        let digest = Digest::from_bytes(*mesh.canonical_digest.bytes());
        if topology_id.parse::<Digest>().ok() != Some(digest) || mesh.topology.nodes.is_empty() {
            return Err(ResourcePlanningError::Invalid(
                "field workspace differs from canonical mesh identity or has no nodes".into(),
            ));
        }
        let PhysicalSignal::BodyForceDensity(signal) = signal else {
            return Err(ResourcePlanningError::Invalid(
                "body-density action requires the matching physical signal quantity".into(),
            ));
        };
        validate_representation(&signal.representation, mesh.topology.nodes.len())?;
        let components = u64::try_from(mesh.topology.nodes.len())
            .map_err(|_| ResourcePlanningError::Overflow("field rows"))?
            .checked_mul(3)
            .ok_or(ResourcePlanningError::Overflow("field components"))?;
        accumulate(
            &mut facts,
            &mut frozen_bytes,
            &mut peak_scratch,
            components,
            crate::excitation::sample_identity::nodal_sample_encoding_workspace(
                digest,
                mesh.topology.nodes.len(),
            )
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?,
        )?;
    }
    facts.resident_bytes_upper_bound = ByteCount(add(frozen_bytes, peak_scratch)?);
    Ok(facts)
}

fn validate_representation<Q>(
    representation: &SignalRepresentation<Q>,
    rows: usize,
) -> Result<(), ResourcePlanningError> {
    match representation {
        SignalRepresentation::Constant(value) => validate_value(value, rows),
        SignalRepresentation::Waveform(waveform) => {
            validate_value(&waveform.amplitude, rows)?;
            if let Some(offset) = &waveform.offset { validate_value(offset, rows)?; }
            Ok(())
        }
        SignalRepresentation::Tabulated(table) => {
            if table.knots.is_empty() { return Err(ResourcePlanningError::Invalid("field table has no samples".into())); }
            for knot in &table.knots {
                match &knot.value {
                    SignalKnotValue::Continuous(value) => validate_ordinate(value, rows)?,
                    SignalKnotValue::Jump { left_limit, right_limit, .. } => {
                        validate_ordinate(left_limit, rows)?;
                        validate_ordinate(right_limit, rows)?;
                    }
                }
            }
            Ok(())
        }
        SignalRepresentation::Artifact(_) | SignalRepresentation::Generator(_) => Err(ResourcePlanningError::Invalid("field resource planning requires sealed concrete signal values before numerical admission".into())),
    }
}
fn validate_ordinate<Q>(
    value: &SignalOrdinate<Q>,
    rows: usize,
) -> Result<(), ResourcePlanningError> {
    match value {
        SignalOrdinate::Real(value) => validate_value(value, rows),
        SignalOrdinate::Complex { real, imaginary } => {
            validate_value(real, rows)?;
            validate_value(imaginary, rows)
        }
    }
}
fn validate_value<Q>(value: &SignalValue<Q>, rows: usize) -> Result<(), ResourcePlanningError> {
    let components = rows
        .checked_mul(3)
        .ok_or(ResourcePlanningError::Overflow("field tensor components"))?;
    match value {
        SignalValue::Tensor { shape, values }
            if shape.as_slice() == [rows, 3] && values.len() == components =>
        {
            Ok(())
        }
        _ => Err(ResourcePlanningError::Invalid(
            "field value rows/components differ from canonical mesh".into(),
        )),
    }
}

fn accumulate(
    facts: &mut ExcitationWorkspaceResourceFacts,
    frozen: &mut u64,
    scratch: &mut u64,
    components: u64,
    encoding: u64,
) -> Result<(), ResourcePlanningError> {
    facts.active_field_count = Count(add(facts.active_field_count.0, 1)?);
    facts.scalar_component_count = Count(add(facts.scalar_component_count.0, components)?);
    // Real+imaginary frozen values per action, including vector/identity headers.
    *frozen = add(*frozen, add(mul(components, 16)?, 128)?)?;
    // While previously realized actions remain alive, interpolation retains two inputs
    // and its output (48C). Freezing retains one tensor plus one field (32C);
    // projection clones one frozen field (16C), and identity encoding is separate.
    // Reserving the full 48C in addition to all frozen values is conservative.
    *scratch = (*scratch).max(add(mul(components, 48)?, 256)?.max(encoding));
    Ok(())
}
fn add(left: u64, right: u64) -> Result<u64, ResourcePlanningError> {
    left.checked_add(right)
        .ok_or(ResourcePlanningError::Overflow("excitation workspace"))
}
fn mul(left: u64, right: u64) -> Result<u64, ResourcePlanningError> {
    left.checked_mul(right)
        .ok_or(ResourcePlanningError::Overflow("excitation workspace"))
}

#[cfg(test)]
mod tests;
