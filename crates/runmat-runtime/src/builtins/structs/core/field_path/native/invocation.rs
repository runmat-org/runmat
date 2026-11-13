//! Execute an already selected native field protocol under its original owner.
use super::super::FieldStep;
use super::{invalid, selectors, Prepared};
use crate::object::{
    indexing::{ObjectSubscript, ObjectSubscriptPath},
    protocol,
};
use crate::{execution::RootedValue, BuiltinResult};
use runmat_value::{ObjectInstance, Value};

impl Prepared {
    pub(super) async fn checkpoint(&mut self, units: u64) -> BuiltinResult<()> {
        if let Some(budget) = self.budget.as_mut() {
            budget
                .checkpoint(units)
                .await
                .map_err(|source| source.into_runtime_error_in_context(self.runtime.as_ref()))?;
        }
        Ok(())
    }
    fn check(&self) -> BuiltinResult<()> {
        let current = match &self.runtime {
            Some(runtime) => self.issuance.is_current_in_context(runtime),
            None => {
                let scope = crate::context::PreparedUnselectedScope::prepare(None)?;
                let _scope = scope.enter();
                self.issuance.is_current_ordinary()
            }
        };
        let method_current = crate::context::with_original_scope(self.runtime.as_ref(), || {
            protocol::resolved_method_is_current(&self.method)
        });
        if current && method_current {
            Ok(())
        } else {
            Err(invalid(
                "native class issuance changed during field path preparation",
            ))
        }
    }
    /// This selected protocol is an owned dynamic continuation. Keep its
    /// concrete dispatcher state out of each functional-field caller frame.
    pub(in crate::builtins::structs::core) fn read<'a>(
        self,
        object: &'a ObjectInstance,
        name: &'a str,
    ) -> crate::execution::RuntimeContinuationFuture<'a, RootedValue> {
        let runtime = self.runtime.clone();
        crate::execution::rooted_continuation::admit_rooted_future_with(
            runtime,
            "getfield",
            crate::builtin::builtin_control_error,
            move |_| self.read_in_scope(object, name),
        )
    }
    async fn read_in_scope(
        mut self,
        object: &ObjectInstance,
        name: &str,
    ) -> BuiltinResult<RootedValue> {
        self.checkpoint(0).await?;
        let name = crate::call::argument_text::copy_str(name, self.runtime.as_ref())?;
        let path = ObjectSubscriptPath::single_in_context(
            self.runtime.as_ref(),
            ObjectSubscript::member_retained(name),
        )?;
        self.check()?;
        let method = Some(self.method);
        let work = protocol::invoke_resolved_object_protocol_with_context(
            self.runtime.as_ref(),
            &method,
            Value::Object(object.clone()),
            path,
            1,
        );
        let output = match &self.runtime {
            Some(_) => work.await?,
            None => {
                crate::context::PreparedUnselectedScope::prepare(None)?
                    .scope(work)
                    .await?
            }
        };
        output.try_into_single()
    }
    pub(in crate::builtins::structs::core) fn assign<'a>(
        self,
        object: ObjectInstance,
        steps: &'a [FieldStep],
        rhs: Value,
    ) -> crate::execution::RuntimeContinuationFuture<'a, RootedValue> {
        let runtime = self.runtime.clone();
        crate::execution::rooted_continuation::admit_rooted_future_with(
            runtime,
            "setfield",
            crate::builtin::builtin_control_error,
            move |_| self.assign_in_scope(object, steps, rhs),
        )
    }
    async fn assign_in_scope(
        mut self,
        object: ObjectInstance,
        steps: &[FieldStep],
        rhs: Value,
    ) -> BuiltinResult<RootedValue> {
        self.checkpoint(0).await?;
        let path = selectors::path(&mut self, steps).await?;
        self.check()?;
        if path.has_deferred_end() {
            let target = self.deferred.ok_or_else(|| {
                invalid("native protocol has no typed contextual-index invocation")
            })?;
            let receiver = Value::Object(object);
            let rhs = crate::context::with_original_scope(self.runtime.as_ref(), || {
                RootedValue::in_optional_context(rhs, self.runtime.as_ref())
            })?;
            let _inputs = crate::context::with_original_scope(self.runtime.as_ref(), || {
                protocol::retain_protocol_inputs(self.runtime.as_ref(), &receiver, &path)
            })?;
            let work = crate::context::with_original_scope(self.runtime.as_ref(), || {
                target.assign(
                    self.runtime.as_ref(),
                    &receiver,
                    &path,
                    rhs.value(),
                    &self.issuance,
                    &self.method,
                )
            });
            return match &self.runtime {
                Some(_) => work.await,
                None => {
                    crate::context::PreparedUnselectedScope::prepare(None)?
                        .scope(work)
                        .await
                }
            };
        }
        let values = runmat_value::ValueSequence::comma_separated(vec![rhs])
            .map_err(crate::sequence::sequence_error_to_runtime)?;
        let work = protocol::invoke_resolved_object_assignment_with_context(
            self.runtime.as_ref(),
            &self.method,
            Value::Object(object),
            path,
            values,
        );
        match &self.runtime {
            Some(_) => work.await,
            None => {
                crate::context::PreparedUnselectedScope::prepare(None)?
                    .scope(work)
                    .await
            }
        }
    }
}
