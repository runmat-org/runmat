//! Policy-reference admission for exact ordered selected-unit archives.

use super::GraphExecutionRequest;
use crate::{ArtifactError, ArtifactResult, ProgramBuildRecipe};

pub(super) fn validate_refs(
    request: &GraphExecutionRequest,
    selected_count: usize,
) -> ArtifactResult<()> {
    let legacy_pure = request.selected_foreign_policies.is_empty();
    if legacy_pure {
        if request.selected_bundle_archives.iter().any(Option::is_some) {
            return Err(ArtifactError::Identity(
                "graph archives require ordered issued policy refs".into(),
            ));
        }
    } else if request.selected_foreign_policies.len() != selected_count {
        return Err(ArtifactError::Invalid(
            "graph policy ref count differs from selected V3 units".into(),
        ));
    } else if request
        .selected_foreign_policies
        .iter()
        .all(Option::is_none)
        && request.selected_bundle_archives.iter().all(Option::is_none)
    {
        return Err(ArtifactError::Invalid(
            "pure graph policy refs must use the legacy empty encoding".into(),
        ));
    }
    for policy in request.selected_foreign_policies.iter().flatten() {
        crate::foreign_policy_descriptor(*policy)?;
    }
    let top_policy = request
        .recipe
        .foreign_policy
        .as_ref()
        .map(crate::foreign_policy_ref)
        .transpose()?;
    let primary_ref = request.selected_foreign_policies.first().copied().flatten();
    if top_policy != primary_ref {
        return Err(ArtifactError::Identity(
            "graph primary recipe differs from its issued policy ref".into(),
        ));
    }
    Ok(())
}

pub(super) fn validate_archive(
    request: &GraphExecutionRequest,
    index: usize,
    recipe: &ProgramBuildRecipe,
    expected_policy: runmat_execution::ForeignPolicyRef,
) -> ArtifactResult<()> {
    let archived_policy = recipe
        .foreign_policy
        .as_ref()
        .map(crate::foreign_policy_ref)
        .transpose()?;
    if archived_policy != Some(expected_policy) {
        return Err(ArtifactError::Identity(
            "graph unit policy ref differs from its own archive recipe".into(),
        ));
    }
    if index == 0 && recipe.foreign_policy != request.recipe.foreign_policy {
        return Err(ArtifactError::Identity(
            "graph primary policy differs from its admitted archive recipe".into(),
        ));
    }
    Ok(())
}
