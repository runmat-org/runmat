//! Selected durable-decomposition proof over borrowed aggregate values.
//! The numerical owner alone interprets the public recipe and factor DATA.

use super::PreferenceServiceError;
use crate::context::RuntimeContext;
use crate::execution::{CpuCooperationBudget, RootedValueAdmissionError, RootedValueHandles};
use runmat_value::Value;
use std::future::Future;
use std::pin::Pin;

pub(super) async fn validate(
    value: &Value,
    runtime: &RuntimeContext,
    budget: &mut CpuCooperationBudget,
) -> Result<(), PreferenceServiceError> {
    // Direct service callers need the same borrowed-input lifetime across
    // cooperative proof as the public builtin and the selected decoder.
    let _roots = RootedValueHandles::for_value_in_context(value, runtime).map_err(|source| {
        let source = match source {
            RootedValueAdmissionError::Host(source) => {
                crate::context::host_allocation::host_allocation_runtime_error(
                    Some(runtime),
                    "RunMat:Preference:RootLease",
                    "preference decomposition root admission failed",
                    source,
                )
            }
            RootedValueAdmissionError::Gc(source) => {
                crate::runtime_error::semantic_error_with_source(
                    "RunMat:Preference:RootLease",
                    "preference decomposition root admission failed",
                    source,
                )
            }
        };
        PreferenceServiceError::Storage(source)
    })?;
    visit(value, runtime, budget, 0).await
}

fn visit<'a>(
    value: &'a Value,
    runtime: &'a RuntimeContext,
    budget: &'a mut CpuCooperationBudget,
    depth: u16,
) -> Pin<Box<dyn Future<Output = Result<(), PreferenceServiceError>> + 'a>> {
    Box::pin(async move {
        if depth > runmat_execution::value::ValueLimits::default().max_depth {
            return Err(PreferenceServiceError::ValueEncoding(
                crate::execution::value_codec::ValueCodecError::Invalid {
                    path: "$".into(),
                    message: "durable decomposition aggregate depth exceeded".into(),
                },
            ));
        }
        budget
            .checkpoint(1)
            .await
            .map_err(PreferenceServiceError::Execution)?;
        let child_depth = depth + 1;
        match value {
            Value::Cell(cell) => {
                for child in cell.data() {
                    visit(child, runtime, budget, child_depth).await?;
                }
            }
            Value::Struct(structure) => {
                for child in structure.fields.values() {
                    visit(child, runtime, budget, child_depth).await?;
                }
            }
            Value::StructArray(array) => {
                for name in array.field_names() {
                    for child in array.field_values(name).expect("field from same array") {
                        visit(child, runtime, budget, child_depth).await?;
                    }
                }
            }
            Value::Object(object) => {
                if object.is_class(runmat_types::standard::DECOMPOSITION) {
                    crate::builtins::math::linalg::factor::decomposition::validate_factor_object_binding(
                        object, runtime,
                    )
                    .await
                    .map_err(PreferenceServiceError::Decomposition)?;
                }
                for child in object.properties().values() {
                    visit(child, runtime, budget, child_depth).await?;
                }
            }
            Value::ObjectArray(array) => {
                for child in array.data() {
                    visit(child, runtime, budget, child_depth).await?;
                }
            }
            _ => {}
        }
        Ok(())
    })
}
