//! Publish a native reference only while its actual GC definition is rooted.
use super::super::common::ScalarCooperation;
use super::super::common::{model_policy::Policy, model_storage};
use super::*;
use runmat_value::optimization_variable::{
    OptimizationVariableCandidate, OptimizationVariableIndices, OptimizationVariablePayload,
    OptimizationVariableReferenceRole, OptimizationVariableSelectionCandidate,
};
fn storage_policy() -> Policy {
    Policy {
        identity: runmat_builtins::OPTIMVAR_CATALOG_ENTRY.identity,
        invalid: &runmat_builtins::OPTIMVAR_ERROR_INVALID_INPUT,
    }
}
pub(super) async fn validate(
    candidate: OptimizationVariableCandidate,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<runmat_value::optimization_variable::OptimizationVariableData> {
    super::parse::validate_domain(&candidate, cooperation).await?;
    let mut preparation = candidate.begin_incremental().map_err(data_error)?;
    while preparation.prepare_next().map_err(data_error)? {
        cooperation.completed(1).await?;
    }
    let mut validation = preparation.finish().map_err(data_error)?;
    while validation.validate_next().map_err(data_error)? {
        cooperation.completed(1).await?;
    }
    validation.finish().map_err(data_error)
}
pub(super) async fn create(
    candidate: OptimizationVariableCandidate,
    registration: registration::Fence,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    registration.check(cooperation)?;
    let data = validate(candidate, cooperation).await?;
    let shape = data.shape().clone();
    let count = data.element_count();
    registration.ensure_current(cooperation)?;
    cooperation.completed(0).await?;
    let definition = model_storage::publish(
        runmat_value::optimization_model::OptimizationNativeInput::new(
            OptimizationVariablePayload::Definition(data),
        ),
        CLASS,
        storage_policy(),
        cooperation,
    )
    .await?;
    registration.check(cooperation)?;
    let rooted = model_storage::allocate_gc(&definition, storage_policy(), cooperation)?;
    registration.check(cooperation)?;
    let mut axis_sources = model_storage::array(cooperation, shape.len())?;
    for axis in 0..shape.len() {
        axis_sources
            .try_push(
                runmat_value::optimization_variable::OptimizationVariableAxisSource::DefinitionAxis(
                    axis,
                ),
            )
            .map_err(|_| invalid("reference axis extent changed"))?;
        cooperation.completed(1).await?;
    }
    let candidate = OptimizationVariableSelectionCandidate {
        definition: rooted.handle(),
        definition_elements: count,
        role: OptimizationVariableReferenceRole::Definition,
        indices: OptimizationVariableIndices::Whole,
        shape,
        axis_sources: axis_sources
            .try_finish()
            .map_err(|_| invalid("reference axes are incomplete"))?,
    };
    let mut preparation = candidate.begin_incremental().map_err(data_error)?;
    while preparation.prepare_next().map_err(data_error)? {
        cooperation.completed(1).await?;
    }
    let selected = preparation
        .finish()
        .map_err(data_error)?
        .finish()
        .map_err(data_error)?;
    // The final root forest traces the original target before the temporary
    // explicit GC root is retired. No raw unrooted result crosses an await.
    let returned = model_storage::publish(
        runmat_value::optimization_model::OptimizationNativeInput::new(
            OptimizationVariablePayload::Reference(selected),
        ),
        CLASS,
        storage_policy(),
        cooperation,
    )
    .await?;
    registration.check(cooperation)?;
    Ok(returned)
}
pub(super) async fn publish_reference(
    candidate: OptimizationVariableSelectionCandidate,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    let _root = model_storage::retain_handle(candidate.definition, storage_policy(), cooperation)?;
    let mut validation = candidate.begin().map_err(data_error)?;
    while validation.validate_next().map_err(data_error)? {
        cooperation.completed(1).await?;
    }
    let selected = validation.finish().map_err(data_error)?;
    model_storage::publish(
        runmat_value::optimization_model::OptimizationNativeInput::new(
            OptimizationVariablePayload::Reference(selected),
        ),
        CLASS,
        storage_policy(),
        cooperation,
    )
    .await
}

pub(in crate::builtins::math::optim) async fn original_definition_reference(
    view: &View,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    use runmat_value::optimization_variable::OptimizationVariableAxisSource;
    let mut axis_sources = model_storage::array(cooperation, view.definition().shape().len())?;
    for axis in 0..view.definition().shape().len() {
        axis_sources
            .try_push(OptimizationVariableAxisSource::DefinitionAxis(axis))
            .map_err(|_| invalid("reference axis extent changed"))?;
        cooperation.completed(1).await?;
    }
    publish_reference(
        OptimizationVariableSelectionCandidate {
            definition: view.selection().definition(),
            definition_elements: view.definition().element_count(),
            role: OptimizationVariableReferenceRole::Definition,
            indices: OptimizationVariableIndices::Whole,
            shape: view.definition().shape().clone(),
            axis_sources: axis_sources
                .try_finish()
                .map_err(|_| invalid("reference axes are incomplete"))?,
        },
        cooperation,
    )
    .await
}
