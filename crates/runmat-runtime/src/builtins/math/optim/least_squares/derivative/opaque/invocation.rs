//! Three typed product roles own the complete public flag/shape boundary.
use super::*;
use runmat_value::Tensor;
#[derive(Clone, Copy)]
pub(super) enum Input<'a> {
    Forward(&'a [f64]),
    Transpose(&'a [f64]),
    Normal(&'a [f64]),
    Column(usize),
}
impl Opaque {
    pub(super) async fn invoke(
        &self,
        input: Input<'_>,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<RootedValue> {
        let (from, to, lanes, flag, source) = match input {
            Input::Forward(source) => (self.columns, self.rows, 1, 1., Some(source)),
            Input::Transpose(source) => (self.rows, self.columns, 1, -1., Some(source)),
            Input::Normal(source) => (self.columns, self.columns, 2, 0., Some(source)),
            Input::Column(axis) => {
                if axis >= self.columns {
                    return Err(invalid("Jacobian column is out of range"));
                }
                (self.columns, self.rows, 1, 1., None)
            }
        };
        if source.is_some_and(|value| value.len() != from) {
            return Err(invalid("Jacobian multiply input geometry mismatch"));
        }
        let length = from
            .checked_mul(lanes)
            .ok_or_else(|| invalid("Jacobian product extent overflow"))?;
        c.completed(0).await?;
        let mut values = runmat_value::exact_capacity_vec(length).map_err(allocation)?;
        for first in (0..length).step_by(1024) {
            let end = length.min(first + 1024);
            for index in first..end {
                let value = if index >= from {
                    0.
                } else {
                    match input {
                        Input::Column(axis) => {
                            if index == axis {
                                1.
                            } else {
                                0.
                            }
                        }
                        _ => source.expect("typed vector product")[index],
                    }
                };
                if !value.is_finite() {
                    return Err(invalid("Jacobian product input must be finite"));
                }
                values.push(value);
            }
            c.completed((end - first) as u64).await?;
        }
        let mut argument = Some(Value::Tensor(
            Tensor::new(values, vec![from, lanes]).map_err(|error| invalid(&error))?,
        ));
        let empty = crate::context::with_original_scope(c.runtime(), || {
            crate::execution::RootedValues::from_borrowed_arguments(&[], c.runtime())
                .map(RootedFunctionArguments::ordinary)
        })?;
        // Exactly the final three argument slots. Original Jinfo registrations
        // are retained rather than traversed again. Published Value clones
        // retain the immutable physical generation, including structs/cells.
        let arguments = crate::context::with_original_scope(c.runtime(), || {
            crate::execution::RootedValues::transform_arguments(
                &empty,
                3,
                Some(&self.info),
                c.runtime(),
                |slot| {
                    Ok(match slot {
                        0 => self.info.value().clone(),
                        1 => argument.take().expect("one actual input matrix slot"),
                        2 => Value::Num(flag),
                        _ => unreachable!("three product inputs"),
                    })
                },
            )
        })?;
        let returned = self
            .function
            .get()
            .invoke_retained(c.runtime(), BuiltinArguments::from(arguments), 1)
            .await?;
        c.completed(0).await?;
        let returned = crate::context::with_original_scope(c.runtime(), || {
            returned
                .resolve_sequence(
                    runmat_types::SequenceUse::from_requested_outputs(
                        runmat_types::RequestedOutputCount::Exactly(1),
                    ),
                    crate::sequence::SequenceResolutionContext::default(),
                )?
                .try_into_single()
        })?;
        let value = returned.value();
        if !matches!(
            value,
            Value::Num(_) | Value::Tensor(_) | Value::SparseTensor(_) | Value::GpuTensor(_)
        ) {
            return Err(invalid("Jacobian multiply output must be a real matrix"));
        }
        let fact = crate::value_fact::value_fact(value);
        if !matches!(fact.kind,runmat_types::ValueKindFact::Numeric(n)
            if matches!(n.class,runmat_types::NumericClass::Single|runmat_types::NumericClass::Double)
                && n.domain == runmat_types::NumericDomain::Real)
            || !fact.shape.known_dims().is_some_and(|shape| {
                shape.first().copied().flatten().unwrap_or(1) == to
                    && shape.get(1).copied().flatten().unwrap_or(1) == lanes
                    && shape.iter().skip(2).all(|extent| *extent == Some(1))
            })
        {
            return Err(invalid(
                "Jacobian multiply output has incompatible class or shape",
            ));
        }
        crate::builtins::math::optim::common::preflight_host_residency(
            "lsqnonlin",
            value,
            None,
            c.runtime(),
        )?;
        let returned = if matches!(value, Value::GpuTensor(_)) {
            let runtime = c.runtime().cloned();
            returned
                .with_owned_async(|value| async move {
                    let value = crate::dispatcher::gather_if_needed_in_context_async(
                        &value,
                        runtime.as_ref(),
                    )
                    .await?;
                    crate::context::with_original_scope(runtime.as_ref(), || {
                        RootedValue::in_optional_context(value, runtime.as_ref())
                    })
                })
                .await?
        } else {
            returned
        };
        Ok(returned)
    }
}
