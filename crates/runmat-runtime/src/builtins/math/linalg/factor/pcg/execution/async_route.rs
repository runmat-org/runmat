use super::*;

struct PcgCallback<T> {
    selected: RepeatedCallableVector,
    scalar: PhantomData<T>,
}

impl<T: PcgValue> PcgCallback<T> {
    fn select(target: Value, n: usize, context: &RuntimeContext) -> BuiltinResult<Self> {
        Ok(Self {
            selected: RepeatedCallableVector::select(
                context,
                target,
                VectorContract {
                    len: n,
                    class: T::VECTOR_CLASS,
                },
            )?,
            scalar: PhantomData,
        })
    }
}

impl<T: PcgValue> CallableOperation<T> for PcgCallback<T> {
    async fn apply_into(
        &mut self,
        input: &[T],
        output: &mut [T],
        runtime: &RuntimeContext,
    ) -> Result<(), RuntimeError> {
        // The callback may retain its language input. Its numeric payload is
        // therefore an independent Value, not an alias of operation scratch.
        let argument = T::unbounded_output(input.to_vec())?;
        let result = self.selected.apply(runtime, argument).await?;
        let value = result
            .value()
            .as_slice()
            .first()
            .ok_or_else(|| invalid("operator callback must return exactly one value"))?;
        let values = validate_vector::<T>(
            value,
            output.len(),
            "operator callback result must be an n-by-1 vector",
        )?;
        output.copy_from_slice(values);
        Ok(())
    }
}

fn preconditioner<'a, T: PcgValue>(
    value: &'a mut Option<Value>,
    n: usize,
    context: &RuntimeContext,
) -> BuiltinResult<Option<Preconditioner<'a, T, PcgCallback<T>>>> {
    if value
        .as_ref()
        .is_some_and(super::super::request::is_callable)
    {
        let target = value.take().expect("callable option was present");
        return PcgCallback::select(target, n, context)
            .map(Preconditioner::callable)
            .map(Some);
    }
    value
        .as_ref()
        .map(|factor| {
            let factor = T::triangle(factor, context).map_err(|error| {
                super::super::async_operations::triangular_error(error, context)
            })?;
            if factor.order() != n {
                return Err(invalid("preconditioner dimensions must match A"));
            }
            Ok(Preconditioner::triangular(factor))
        })
        .transpose()
}

pub(super) struct AsyncRequest<'a> {
    pub(super) rhs: &'a Value,
    pub(super) initial: Option<&'a Value>,
    pub(super) tolerance: f64,
    pub(super) maximum_iterations: usize,
    pub(super) first: &'a mut Option<Value>,
    pub(super) second: &'a mut Option<Value>,
}

pub(super) enum CoefficientInput<'a, T: PcgValue> {
    Matrix(TypedLinearOperator<'a, T>),
    Callable(Value),
}

pub(super) async fn run_async_typed<T: PcgValue>(
    source: CoefficientInput<'_, T>,
    request: &mut AsyncRequest<'_>,
    context: &RuntimeContext,
    requested: Option<usize>,
) -> BuiltinResult<Value> {
    let n = match &source {
        CoefficientInput::Callable(_) => match request.rhs {
            Value::Tensor(tensor) => tensor.rows(),
            Value::ComplexTensor(tensor) => tensor.rows(),
            _ => 0,
        },
        CoefficientInput::Matrix(matrix) => matrix.rows(),
    };
    let rhs = validate_vector::<T>(
        request.rhs,
        n,
        "b must be a column vector with size(A,1) rows",
    )?;
    let initial = request
        .initial
        .map(|value| validate_vector::<T>(value, n, "x0 must match b's column shape"))
        .transpose()?;
    let coefficient = match source {
        CoefficientInput::Matrix(matrix) => Coefficient::matrix(matrix),
        CoefficientInput::Callable(callback) => {
            Coefficient::callable(PcgCallback::select(callback, n, context)?)
        }
    };
    let first = preconditioner::<T>(request.first, n, context)?;
    let second = preconditioner::<T>(request.second, n, context)?;
    let both = first.is_some() && second.is_some();
    let routes = Operations::new(coefficient, first, second);
    let vector_count = if both { 6usize } else { 5usize };
    let PcgQuote {
        main_count,
        residual_count,
        plan,
    } = quote::<T>(
        n,
        request.maximum_iterations,
        vector_count,
        requested,
        Some(context),
    )?;
    let admission =
        |error| map_operator_in_context(SparseOperatorError::Admission(error), Some(context));
    let receipt = HostAllocationReceipt::admit(Some(context), plan).map_err(admission)?;
    if let Some(receipt) = receipt.as_ref() {
        let mut main = receipt
            .temporary_numeric(context, T::DTYPE, main_count)
            .map_err(admission)?;
        let mut residuals = receipt
            .temporary_numeric(context, NumericDType::F64, residual_count)
            .map_err(admission)?;
        let work = T::scratch(&mut main).ok_or_else(|| invalid("scratch class mismatch"))?;
        let (x, rest) = work.split_at_mut(n);
        let (best, rest) = rest.split_at_mut(n);
        let (residual, rest) = rest.split_at_mut(n);
        let (direction, rest) = rest.split_at_mut(n);
        let (operation_scratch, intermediate) = rest.split_at_mut(n);
        let norms = residuals
            .as_f64_slice_mut()
            .ok_or_else(|| invalid("residual scratch class mismatch"))?;
        let outcome = routes
            .solve(
                PcgSolveInput {
                    rhs,
                    initial,
                    tolerance: request.tolerance,
                    maximum_iterations: request.maximum_iterations,
                    work: PcgWork {
                        x,
                        best,
                        residual,
                        direction,
                        intermediate: both.then_some(intermediate),
                        residual_norms: norms,
                    },
                    operation_scratch,
                },
                context,
                Some(receipt),
            )
            .await?;
        publish_admitted::<T>(receipt, context, x, norms, outcome, requested)
    } else {
        let mut main = vec![T::zero(); vector_count * n];
        let mut norms = vec![0.0; residual_count];
        let (x, rest) = main.split_at_mut(n);
        let (best, rest) = rest.split_at_mut(n);
        let (residual, rest) = rest.split_at_mut(n);
        let (direction, rest) = rest.split_at_mut(n);
        let (operation_scratch, intermediate) = rest.split_at_mut(n);
        let outcome = routes
            .solve(
                PcgSolveInput {
                    rhs,
                    initial,
                    tolerance: request.tolerance,
                    maximum_iterations: request.maximum_iterations,
                    work: PcgWork {
                        x,
                        best,
                        residual,
                        direction,
                        intermediate: both.then_some(intermediate),
                        residual_norms: &mut norms,
                    },
                    operation_scratch,
                },
                context,
                None,
            )
            .await?;
        publish_unbounded::<T>(main, n, &norms, outcome, requested)
    }
}
