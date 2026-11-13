use runmat_analysis_core::{
    AnalysisModel, CoordinateFrame, InitialConditionValue, PhysicalSignal, Signal, StructuralModel,
};
use runmat_meshing_core::SolverMeshArtifact;

use crate::{
    assembly::dofs::{StructuralDofKind, StructuralDofLayout},
    physics::structural::topology::node_space::CanonicalStructuralNodeSpace,
};

use super::{
    sampling::realize_value, target::target_nodes, AppliedStructuralInitialConditionEvidence,
    StructuralInitialConditionQuantity, StructuralInitialStateError,
};

#[derive(Debug, Clone)]
pub(super) struct Assignment {
    pub value: f64,
    initial_condition_id: Option<String>,
}

impl Assignment {
    pub(super) fn constraint_default(value: f64) -> Self {
        Self {
            value,
            initial_condition_id: None,
        }
    }

    pub(super) fn initial_condition_id(&self) -> &str {
        self.initial_condition_id
            .as_deref()
            .expect("only authored assignments are queried for attribution")
    }

    pub(super) fn authored_initial_condition_id(&self) -> Option<&str> {
        self.initial_condition_id.as_deref()
    }
}

pub(super) struct RealizationContext<'a> {
    pub model: &'a AnalysisModel,
    pub mesh: Option<&'a SolverMeshArtifact>,
    pub structural: &'a StructuralModel,
    pub topology: &'a CanonicalStructuralNodeSpace,
    pub layout: &'a StructuralDofLayout,
    pub fields: &'a [crate::execution::InitialFieldBinding],
    pub topology_digest: runmat_execution::Digest,
}

#[derive(Clone, Copy)]
pub(super) struct QuantityMapping {
    pub quantity: StructuralInitialConditionQuantity,
    pub dofs: [StructuralDofKind; 3],
}

pub(super) fn apply_condition<Q>(
    context: &RealizationContext<'_>,
    condition: &InitialConditionValue<Q>,
    mapping: QuantityMapping,
    assignments: &mut [Option<Assignment>],
    typed_signal: impl Fn(&PhysicalSignal) -> Option<&Signal<Q>>,
    component: impl Fn(&Q) -> f64 + Copy,
    applied_conditions: &mut Vec<AppliedStructuralInitialConditionEvidence>,
) -> Result<(), StructuralInitialStateError> {
    if !matches!(condition.frame, CoordinateFrame::Global) {
        return Err(StructuralInitialStateError::UnsupportedFrame {
            initial_condition_id: condition.initial_condition_id.clone(),
            frame: condition.frame.context_label(),
        });
    }
    let value = super::fields::field_values(context, condition)?;
    let uniform = if value.is_none() {
        Some(realize_value(
            context.model,
            condition,
            typed_signal,
            component,
        )?)
    } else {
        None
    };
    let nodes = target_nodes(
        context.mesh,
        context.structural,
        context.topology,
        condition,
    )?;
    for &node in &nodes {
        for (component_index, dof) in mapping.dofs.into_iter().enumerate() {
            let row = context.layout.index(node, dof).ok_or_else(|| {
                StructuralInitialStateError::MissingDof {
                    initial_condition_id: condition.initial_condition_id.clone(),
                    node_index: node,
                    dof,
                }
            })?;
            assign(
                assignments,
                row,
                match &value {
                    Some(field) => field.values()[node * 3 + component_index],
                    None => uniform.as_ref().unwrap().0[component_index],
                },
                &condition.initial_condition_id,
            )?;
        }
    }
    applied_conditions.push(AppliedStructuralInitialConditionEvidence {
        initial_condition_id: condition.initial_condition_id.clone(),
        quantity: mapping.quantity,
        target: condition.target.clone(),
        resolved_node_identities: nodes
            .iter()
            .map(|&node| context.topology.nodes[node].identity)
            .collect(),
        source: match &value {
            Some(field) => super::StructuralInitialValueSourceEvidence::FieldArtifact {
                reference: field.binding().reference().clone(),
            },
            None => uniform.as_ref().unwrap().1.clone(),
        },
        realized_component_bits: uniform.map(|(components, _)| components.map(f64::to_bits)),
    });
    Ok(())
}

fn assign(
    assignments: &mut [Option<Assignment>],
    row: usize,
    value: f64,
    initial_condition_id: &str,
) -> Result<(), StructuralInitialStateError> {
    if let Some(existing) = &assignments[row] {
        if existing.value != value {
            return Err(StructuralInitialStateError::ConflictingValue {
                first_initial_condition_id: existing.initial_condition_id().to_string(),
                second_initial_condition_id: initial_condition_id.to_string(),
                row,
            });
        }
        return Ok(());
    }
    assignments[row] = Some(Assignment {
        value,
        initial_condition_id: Some(initial_condition_id.to_string()),
    });
    Ok(())
}
