//! A rooted custom generation uses normal size/subsref semantics. It does not
//! reinterpret cell genomes as numeric chromosomes or copy their payloads.
use super::super::callbacks::Invocation;
use super::super::{common, invalid, ScalarCooperation, NAME, POLICY};
use crate::{execution::RootedValue, BuiltinResult};
use common::population::projection::{Argument, Arguments, Prepared};
use runmat_value::{IntegerStorage, Tensor, Value};
use std::ops::ControlFlow;

#[derive(Clone, Copy)]
enum Members {
    Exact(usize),
    AtMost(usize),
}
pub(in super::super) struct Custom {
    generation: RootedValue,
    members: usize,
}
impl Custom {
    pub fn fork(&self, cooperation: &ScalarCooperation) -> BuiltinResult<Self> {
        Ok(Self {
            generation: RootedValue::in_optional_context(
                self.generation.value().clone(),
                cooperation.runtime(),
            )?,
            members: self.members,
        })
    }
    pub async fn from_result<I: Invocation>(
        generation: RootedValue,
        expected_members: usize,
        invocation: &mut I,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<ControlFlow<I::Stop, Self>> {
        Self::admit(
            generation,
            Members::Exact(expected_members),
            invocation,
            cooperation,
        )
        .await
    }
    pub async fn from_initial<I: Invocation>(
        generation: RootedValue,
        maximum_members: usize,
        invocation: &mut I,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<ControlFlow<I::Stop, Self>> {
        Self::admit(
            generation,
            Members::AtMost(maximum_members),
            invocation,
            cooperation,
        )
        .await
    }
    async fn admit<I: Invocation>(
        generation: RootedValue,
        members: Members,
        invocation: &mut I,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<ControlFlow<I::Stop, Self>> {
        let runtime = selected_runtime(cooperation)?.clone();
        let size = Value::FunctionHandle(runmat_builtins::SIZE_CATALOG_ENTRY.identity.name.into());
        let arguments = [generation.value().clone()];
        let returned = match invocation
            .protocol(
                move |original| async move {
                    common::call_function_sequence_in_context(
                        original.as_ref(),
                        &size,
                        arguments,
                        1,
                    )
                    .await
                },
                cooperation,
            )
            .await?
        {
            ControlFlow::Continue(returned) => returned,
            ControlFlow::Break(stop) => return Ok(ControlFlow::Break(stop)),
        };
        let returned = {
            let _scope = runtime.enter();
            returned.try_into_single()?
        };
        let count = returned
            .with_owned_async(|value| async {
                let Value::Tensor(dimensions) = value else {
                    return Err(invalid(
                        "custom population size must be a double dimension vector",
                    ));
                };
                let Some(dimensions) = dimensions.as_f64_slice() else {
                    return Err(invalid(
                        "custom population size must be a double dimension vector",
                    ));
                };
                if dimensions.len() < 2 {
                    return Err(invalid(
                        "custom population size must include rows and columns",
                    ));
                }
                let mut count = 0;
                for (axis, &extent) in dimensions.iter().enumerate() {
                    let extent = common::option_usize_value(
                        NAME,
                        Some(&Value::Num(extent)),
                        "custom population dimension",
                        0,
                    )?;
                    if axis == 0 {
                        count = extent;
                    }
                    if axis > 1 && extent != 1 {
                        return Err(invalid(
                            "custom population must have no nonsingleton higher dimensions",
                        ));
                    }
                    cooperation.completed(1).await?;
                }
                match members {
                    Members::Exact(expected) if expected == 0 || count != expected => Err(invalid(
                        "custom population must have the requested number of rows",
                    )),
                    Members::AtMost(maximum) if count > maximum => {
                        Err(invalid("initial custom population exceeds PopulationSize"))
                    }
                    _ => Ok(count),
                }
            })
            .await?;
        Ok(ControlFlow::Continue(Self {
            generation,
            members: count,
        }))
    }
    /// Install a complete custom operator result through normal subsasgn.
    /// Both input generations remain rooted until the returned generation has
    /// been checked and published; an operator cannot alter the row extent.
    pub async fn replace_rows<I: Invocation>(
        &mut self,
        rows: &[usize],
        children: &Custom,
        invocation: &mut I,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<ControlFlow<I::Stop>> {
        if rows.is_empty() || rows.len() != children.members {
            return Err(invalid(
                "custom children must match the destination row count",
            ));
        }
        // Do not depend on unchecked user operator output for target geometry.
        for &row in rows {
            if row >= self.members {
                return Err(invalid("custom offspring target is out of range"));
            }
            cooperation.completed(1).await?;
        }
        let runtime = selected_runtime(cooperation)?.clone();
        let path = row_path(rows, &runtime, cooperation).await?;
        let mut values = POLICY.capacity(1, Some(&runtime))?;
        values.push(children.value().clone());
        let receiver = self.value().clone();
        let returned = match invocation
            .protocol(
                move |original| {
                    crate::object::protocol::assign_subscript_path_with_runtime(
                        original,
                        receiver,
                        path,
                        values,
                        Some(NAME),
                    )
                },
                cooperation,
            )
            .await?
        {
            ControlFlow::Continue(returned) => returned,
            ControlFlow::Break(stop) => return Ok(ControlFlow::Break(stop)),
        };
        let validated =
            match Self::from_result(returned, self.members, invocation, cooperation).await? {
                ControlFlow::Continue(validated) => validated,
                ControlFlow::Break(stop) => return Ok(ControlFlow::Break(stop)),
            };
        self.generation = validated.generation;
        Ok(ControlFlow::Continue(()))
    }
    pub fn value(&self) -> &Value {
        self.generation.value()
    }
}
impl Arguments for Custom {
    fn members(&self) -> usize {
        self.members
    }
    async fn prepare_projection<'a>(
        &'a self,
        rows: &[usize],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Prepared<'a>> {
        if rows.is_empty() {
            return Err(invalid("custom callback requires selected members"));
        }
        let mut original = rows.len() == self.members;
        for (index, &row) in rows.iter().enumerate() {
            if row >= self.members {
                return Err(invalid("custom callback member is out of range"));
            }
            original &= index == row;
            cooperation.completed(1).await?;
        }
        if original {
            return Ok(Prepared::Ready(Argument::Source(self.value().clone())));
        }
        let runtime = selected_runtime(cooperation)?.clone();
        let path = row_path(rows, &runtime, cooperation).await?;
        Ok(Prepared::Subscript {
            receiver: self.value(),
            path,
            builtin: NAME,
        })
    }
}
async fn row_path(
    rows: &[usize],
    runtime: &crate::context::RuntimeContext,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<crate::object::indexing::ObjectSubscriptPath> {
    use crate::object::indexing::{
        ObjectIndexComponent, ObjectIndexSelector, ObjectSubscript, ObjectSubscriptPath,
    };
    let mut indices = POLICY.capacity(rows.len(), Some(runtime))?;
    for &row in rows {
        let one_based = row
            .checked_add(1)
            .and_then(|index| u64::try_from(index).ok())
            .ok_or_else(|| invalid("custom population index overflow"))?;
        indices.push(one_based);
        cooperation.completed(1).await?;
    }
    let indices = Value::Tensor(
        Tensor::new_integer(IntegerStorage::U64(indices), vec![rows.len(), 1]).map_err(invalid)?,
    );
    let mut first = Some(indices);
    let components =
        crate::context::host_allocation::initialize_host_array(Some(runtime), 2, |axis| {
            Ok(if axis == 0 {
                ObjectIndexComponent::Value(first.take().expect("first selector"))
            } else {
                ObjectIndexComponent::Colon
            })
        })?;
    let path = ObjectSubscriptPath::single_in_context(
        Some(runtime),
        ObjectSubscript::parentheses(ObjectIndexSelector::IndexValues { components }),
    )?;
    Ok(path)
}

fn selected_runtime(
    cooperation: &ScalarCooperation,
) -> BuiltinResult<&crate::context::RuntimeContext> {
    cooperation.runtime().ok_or_else(|| {
        let mut error = crate::context::RuntimeCapabilityError::new(
            crate::context::RuntimeCapability::Builtin,
            "custom genetic population operations require their selected execution context",
        )
        .into_runtime_error();
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        error
    })
}
