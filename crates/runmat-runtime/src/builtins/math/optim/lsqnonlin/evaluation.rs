//! A callback result is pending until the numerical controller accepts its x.
//! Original tuple roots cover every asynchronous decode and successor root.
use super::*;
use crate::builtins::math::optim::least_squares::Jacobian;
use crate::call::{descriptor::RepeatableCallable, BuiltinArguments};
use crate::execution::RootedValue;
pub(super) struct NonlinEvaluator {
    pub(super) cooperation: crate::builtins::math::optim::common::ScalarCooperation,
    function: RepeatableCallable,
    x_shape: Vec<usize>,
    x_scalar: bool,
    residual_shape: Option<Vec<usize>>,
    supplied: bool,
    pending: Option<Jacobian>,
    accepted: Option<Jacobian>,
}
impl NonlinEvaluator {
    pub(super) fn new(
        cooperation: crate::builtins::math::optim::common::ScalarCooperation,
        function: Value,
        x_shape: Vec<usize>,
        x_scalar: bool,
        supplied: bool,
    ) -> BuiltinResult<Self> {
        let runtime = cooperation.runtime();
        let rooted = RootedValue::in_optional_context(function, runtime)?;
        let function = crate::context::with_original_scope(runtime, || {
            match runtime.and_then(|runtime| runtime.selected_graph_program()) {
                Some(program) => program.select_repeatable_callable(
                    runtime.expect("selected program has original runtime"),
                    rooted,
                ),
                None => RepeatableCallable::select_deferred_rooted(runtime, rooted),
            }
        })?;
        Ok(Self {
            cooperation,
            function,
            x_shape,
            x_scalar,
            residual_shape: None,
            supplied,
            pending: None,
            accepted: None,
        })
    }
    pub(super) fn residual_shape(&self) -> Option<&[usize]> {
        self.residual_shape.as_deref()
    }
}
impl LeastSquaresEvaluator for NonlinEvaluator {
    async fn dense_jacobian(&mut self) -> BuiltinResult<Option<Vec<f64>>> {
        match self.accepted.as_ref() {
            Some(source) => source.copy_row_major(&mut self.cooperation).await.map(Some),
            None => Ok(None),
        }
    }
    fn supplies_jacobian(&self) -> bool {
        self.supplied
    }
    fn accept_residual(&mut self) {
        self.accepted = self.pending.take();
    }
    fn take_jacobian(&mut self) -> Option<Jacobian> {
        self.accepted.take()
    }
    fn cooperation(
        &mut self,
    ) -> Option<&mut crate::builtins::math::optim::common::ScalarCooperation> {
        Some(&mut self.cooperation)
    }
    fn residual<'a>(&'a mut self, x: &'a [f64]) -> ResidualFuture<'a> {
        Box::pin(async move {
            self.pending = None;
            let arg = x_value(x, &self.x_shape, self.x_scalar)?;
            let runtime = self.cooperation.runtime();
            let arguments = BuiltinArguments::from_borrowed(&[arg], runtime)?;
            let width = if self.supplied { 2 } else { 1 };
            let returned = self
                .function
                .invoke_retained(runtime, arguments, width)
                .await?;
            let returned = crate::context::with_original_scope(runtime, || {
                returned.resolve_sequence(
                    runmat_types::SequenceUse::from_requested_outputs(
                        runmat_types::RequestedOutputCount::Exactly(width),
                    ),
                    crate::sequence::SequenceResolutionContext::default(),
                )
            })?;
            crate::builtins::math::optim::common::preflight_real_array(
                NAME,
                &returned.as_slice()[0],
                "function output",
                false,
            )?;
            let rows = crate::value_fact::value_fact(&returned.as_slice()[0])
                .shape
                .element_count()
                .ok_or_else(|| {
                    lsq_error_with_detail(
                        &LSQNONLIN_ERROR_INVALID_INPUT,
                        "residual extent overflow",
                    )
                })?;
            if self.supplied {
                preflight(&returned.as_slice()[1], rows, x.len(), runtime)?;
            }
            let mut values =
                crate::context::with_original_scope(runtime, || returned.try_into_values())?;
            let residual = real_array_with_cooperation(
                "function output",
                values.next().expect("selected original residual"),
                &mut self.cooperation,
            )
            .await?;
            if let Some(expected) = &self.residual_shape {
                if expected != &residual.shape {
                    return Err(lsq_error_with_detail(
                        &LSQNONLIN_ERROR_INVALID_INPUT,
                        "function output shape changed during iteration",
                    ));
                }
            } else {
                self.residual_shape = Some(residual.shape);
            }
            if self.supplied {
                let source = RootedValue::in_optional_context(
                    values.next().expect("selected original Jacobian"),
                    self.cooperation.runtime(),
                )?;
                self.pending =
                    Some(Jacobian::matrix(source, rows, x.len(), &mut self.cooperation).await?);
            }
            Ok(residual.values)
        })
    }
}
fn preflight(
    value: &Value,
    rows: usize,
    columns: usize,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<()> {
    use runmat_types::{NumericClass, NumericDomain, ValueKindFact};
    if !matches!(
        value,
        Value::Num(_) | Value::Tensor(_) | Value::SparseTensor(_) | Value::GpuTensor(_)
    ) {
        return Err(lsq_error_with_detail(
            &LSQNONLIN_ERROR_INVALID_INPUT,
            "Jacobian must be a real m-by-n single or double matrix",
        ));
    }
    let fact = crate::value_fact::value_fact(value);
    let real = matches!(fact.kind, ValueKindFact::Numeric(n)
        if matches!(n.class, NumericClass::Single | NumericClass::Double)
            && n.domain == NumericDomain::Real);
    let shape = fact.shape.known_dims().is_some_and(|shape| {
        shape.first().copied().flatten().unwrap_or(1) == rows
            && shape.get(1).copied().flatten().unwrap_or(1) == columns
            && shape.iter().skip(2).all(|size| *size == Some(1))
    });
    if !real || !shape {
        return Err(lsq_error_with_detail(
            &LSQNONLIN_ERROR_INVALID_INPUT,
            "Jacobian must match residual elements by variables",
        ));
    }
    crate::builtins::math::optim::common::preflight_host_residency(NAME, value, None, runtime)
}
