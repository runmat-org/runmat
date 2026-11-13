use super::*;
pub(super) use common::nelder_mead::Point;
pub(super) struct Evaluator<'a> {
    function: &'a Value,
    shape: &'a [usize],
    scalar: bool,
    max_calls: usize,
    check_values: bool,
    pub calls: usize,
    pub cooperation: &'a mut ScalarCooperation,
}
impl<'a> Evaluator<'a> {
    pub fn new(
        function: &'a Value,
        shape: &'a [usize],
        scalar: bool,
        options: &options::Options<'_>,
        cooperation: &'a mut ScalarCooperation,
    ) -> Self {
        Self {
            function,
            shape,
            scalar,
            max_calls: options.max_calls,
            check_values: options.check_values,
            calls: 0,
            cooperation,
        }
    }
    pub fn remaining(&self) -> usize {
        self.max_calls.saturating_sub(self.calls)
    }
    pub async fn measure(&mut self, x: &[f64]) -> BuiltinResult<Option<f64>> {
        self.cooperation.completed(0).await?;
        if self.remaining() == 0 {
            return Ok(None);
        }
        let mut callback_x = buffer(x.len())?;
        for chunk in x.chunks(1024) {
            callback_x.extend_from_slice(chunk);
            self.cooperation.completed(chunk.len() as u64).await?;
        }
        let argument = common::vector_to_value(NAME, callback_x, self.shape, self.scalar)?;
        let value = common::call_function_in_context(
            self.cooperation.runtime(),
            self.function,
            [argument],
            1,
        )
        .await?;
        self.calls += 1;
        self.cooperation.completed(0).await?;
        let runtime = self.cooperation.runtime().cloned();
        let check = self.check_values;
        let f = value
            .with_owned_async(|value| async move {
                input::admit_numeric(&value)?;
                objective_metadata(&value)?;
                let value =
                    crate::dispatcher::gather_if_needed_in_context_async(&value, runtime.as_ref())
                        .await?;
                objective_metadata(&value)?;
                let tensor = crate::builtins::common::tensor::value_into_tensor_for(NAME, value)
                    .map_err(invalid)?;
                let f = crate::builtins::common::tensor::tensor_value_f64(&tensor, 0);
                if check && f.is_nan() {
                    return Err(build_runtime_error(
                        "fminsearch: FunValCheck rejected NaN objective value",
                    )
                    .with_builtin(NAME)
                    .with_identifier("RunMat:fminsearch:InvalidFunctionValue")
                    .build());
                }
                Ok(f)
            })
            .await?;
        Ok(Some(f))
    }
}

fn objective_metadata(value: &Value) -> BuiltinResult<()> {
    let fact = crate::value_fact::value_fact(value);
    if !matches!(
        fact.kind,
        runmat_types::ValueKindFact::Numeric(runmat_types::NumericFact {
            class: runmat_types::NumericClass::Double | runmat_types::NumericClass::Single,
            domain: runmat_types::NumericDomain::Real
        })
    ) || fact.shape.element_count() != Some(1)
        || fact.storage == runmat_types::StorageFact::Sparse
    {
        return Err(invalid("objective must return one real floating scalar"));
    }

    Ok(())
}
