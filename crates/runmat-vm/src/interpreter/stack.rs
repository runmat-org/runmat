use crate::interpreter::errors::mex;
use runmat_runtime::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationPolicy, HostAllocationReceipt,
};
use runmat_runtime::context::RuntimeContext;
use runmat_runtime::RuntimeError;
use runmat_value::{ExactHostAllocationClaim, Value};
use std::alloc::Layout;
use std::ops::{Deref, Index, IndexMut, RangeBounds};

/// Canonical operand-stack owner. Unbounded execution retains the historical
/// allocation-free empty `Vec`; managed execution owns one exact backing and
/// its physical claim for the entire invocation.
pub struct InterpreterStack {
    values: Vec<Value>,
    _claim: Option<ExactHostAllocationClaim>,
    exact_capacity: Option<usize>,
    receipt: Option<HostAllocationReceipt>,
    context: Option<RuntimeContext>,
}

impl std::fmt::Debug for InterpreterStack {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("InterpreterStack")
            .field("values", &self.values)
            .field("exact_capacity", &self.exact_capacity)
            .finish_non_exhaustive()
    }
}

impl InterpreterStack {
    pub fn unbounded() -> Self {
        Self {
            values: Vec::new(),
            _claim: None,
            exact_capacity: None,
            receipt: None,
            context: None,
        }
    }

    /// Transfer an already-owned ordinary stack without reallocating it.
    /// Managed execution must use `prepare` so its physical claim is retained.
    pub(crate) fn from_unbounded(values: Vec<Value>) -> Self {
        Self {
            values,
            _claim: None,
            exact_capacity: None,
            receipt: None,
            context: None,
        }
    }

    pub fn prepare(
        context: &RuntimeContext,
        geometry: crate::bytecode::BytecodeStackGeometry,
    ) -> Result<Self, RuntimeError> {
        let policy = HostAllocationReceipt::policy(Some(context)).map_err(|error| {
            crate::runtime::workspace::frame_allocation_error_in_context(Some(context), error)
        })?;
        if policy == HostAllocationPolicy::Unbounded {
            return Ok(Self::unbounded());
        }
        let maximum = geometry.exact_maximum().ok_or_else(|| {
            crate::runtime::workspace::frame_allocation_error_in_context(
                Some(context),
                HostAllocationError::MissingManagedOwner,
            )
        })?;
        let layout = Layout::array::<Value>(maximum).map_err(|_| {
            crate::runtime::workspace::frame_allocation_error_in_context(
                Some(context),
                HostAllocationError::Overflow,
            )
        })?;
        let stack_retained = HostAllocationPlan::retained_exact_output_bytes(
            u64::try_from(layout.size()).map_err(|_| {
                crate::runtime::workspace::frame_allocation_error_in_context(
                    Some(context),
                    HostAllocationError::Overflow,
                )
            })?,
            usize::from(maximum != 0),
        )
        .map_err(|error| {
            crate::runtime::workspace::frame_allocation_error_in_context(Some(context), error)
        })?;
        let sequence_physical = u64::try_from(
            runmat_value::OutputListStorage::requested_bytes(maximum).map_err(|error| {
                crate::runtime::workspace::frame_allocation_error_in_context(
                    Some(context),
                    HostAllocationError::Numeric(error),
                )
            })?,
        )
        .map_err(|_| {
            crate::runtime::workspace::frame_allocation_error_in_context(
                Some(context),
                HostAllocationError::Overflow,
            )
        })?;
        let sequence_retained = HostAllocationPlan::retained_exact_output_bytes(
            sequence_physical,
            usize::from(maximum != 0),
        )
        .map_err(|error| {
            crate::runtime::workspace::frame_allocation_error_in_context(Some(context), error)
        })?;
        let retained = stack_retained
            .checked_add(sequence_retained)
            .ok_or_else(|| {
                crate::runtime::workspace::frame_allocation_error_in_context(
                    Some(context),
                    HostAllocationError::Overflow,
                )
            })?;
        let receipt = HostAllocationReceipt::admit(
            Some(context),
            HostAllocationPlan::new(0, retained).map_err(|error| {
                crate::runtime::workspace::frame_allocation_error_in_context(Some(context), error)
            })?,
        )
        .map_err(|error| {
            crate::runtime::workspace::frame_allocation_error_in_context(Some(context), error)
        })?
        .ok_or_else(|| {
            crate::runtime::workspace::frame_allocation_error_in_context(
                Some(context),
                HostAllocationError::StaleAuthority,
            )
        })?;
        let (values, claim) = receipt
            .with_output_allocation(context, |permit| {
                let claim = if maximum == 0 {
                    None
                } else {
                    Some(permit.reserve_exact_claim(layout.size())?)
                };
                let values = runmat_value::exact_capacity_vec(maximum)?;
                Ok((values, claim))
            })
            .map_err(|error| {
                crate::runtime::workspace::frame_allocation_error_in_context(Some(context), error)
            })?;
        receipt.checkpoint(context).map_err(|error| {
            crate::runtime::workspace::frame_allocation_error_in_context(Some(context), error)
        })?;
        Ok(Self {
            values,
            _claim: claim,
            exact_capacity: Some(maximum),
            receipt: Some(receipt),
            context: Some(context.clone()),
        })
    }

    pub fn exact_capacity(&self) -> Option<usize> {
        self.exact_capacity
    }

    pub fn try_push(&mut self, value: Value) -> Result<(), RuntimeError> {
        self.ensure_additional(1)?;
        self.values.push(value);
        Ok(())
    }

    /// Admit one stack slot before asking another owner to relinquish a
    /// value. The factory runs only after the verified stack extent accepts
    /// the push, and `Vec::push` cannot grow an exact managed backing here.
    pub(crate) fn try_push_with(
        &mut self,
        make_value: impl FnOnce() -> Result<Value, RuntimeError>,
    ) -> Result<(), RuntimeError> {
        self.ensure_additional(1)?;
        let value = make_value()?;
        self.values.push(value);
        Ok(())
    }

    pub fn try_extend(
        &mut self,
        values: impl IntoIterator<Item = Value>,
    ) -> Result<(), RuntimeError> {
        let values = values.into_iter();
        if self.exact_capacity.is_none() {
            self.values.extend(values);
            return Ok(());
        }
        let (lower, upper) = values.size_hint();
        if upper != Some(lower) {
            return Err(stack_extent_error(
                "managed operand-stack extension requires exact cardinality",
            ));
        }
        self.ensure_additional(lower)?;
        // size_hint is not a trusted cardinality certificate. Keep actual
        // iteration within the preflighted extent even for a faulty iterator.
        let initial_len = self.values.len();
        for value in values {
            if self.values.len() - initial_len == lower {
                return Err(stack_extent_error(
                    "operand iterator exceeded its declared cardinality",
                ));
            }
            self.values.push(value);
        }
        if self.values.len() - initial_len != lower {
            return Err(stack_extent_error(
                "operand iterator produced fewer values than declared",
            ));
        }
        Ok(())
    }

    pub fn try_extend_output(
        &mut self,
        values: runmat_value::OutputListStorage,
    ) -> Result<(), RuntimeError> {
        let values = values.try_into_owned().map_err(|_| {
            runmat_runtime::sequence::sequence_error_to_runtime(
                runmat_value::ValueSequenceError::SharedOutputStorage,
            )
        })?;
        self.try_extend(values)
    }

    /// The frame registrar follows this stable Vec owner during synchronous
    /// GC scans. It cannot take or grow the backing through this reference.
    pub(super) fn gc_root_owner(&self) -> &Vec<Value> {
        &self.values
    }

    pub fn pop(&mut self) -> Option<Value> {
        self.values.pop()
    }
    pub fn truncate(&mut self, len: usize) {
        self.values.truncate(len);
    }
    pub fn clear(&mut self) {
        self.values.clear();
    }
    pub fn remove(&mut self, index: usize) -> Value {
        self.values.remove(index)
    }
    pub fn try_insert(&mut self, index: usize, value: Value) -> Result<(), RuntimeError> {
        self.ensure_additional(1)?;
        self.values.insert(index, value);
        Ok(())
    }
    pub fn drain<R: RangeBounds<usize>>(&mut self, range: R) -> std::vec::Drain<'_, Value> {
        self.values.drain(range)
    }
    pub fn last_mut(&mut self) -> Option<&mut Value> {
        self.values.last_mut()
    }
    pub fn as_slice(&self) -> &[Value] {
        &self.values
    }

    fn ensure_additional(&self, additional: usize) -> Result<(), RuntimeError> {
        let Some(capacity) = self.exact_capacity else {
            return Ok(());
        };
        let required = self
            .values
            .len()
            .checked_add(additional)
            .ok_or_else(|| stack_extent_error("operand-stack length overflow"))?;
        if required > capacity {
            return Err(stack_extent_error(
                "verified operand-stack capacity was exceeded",
            ));
        }
        Ok(())
    }

    /// Restore a slice previously removed by `try_extract_output`. The stack
    /// still owns the same physical capacity, so this cannot grow its backing.
    pub(crate) fn restore_extracted(&mut self, values: runmat_value::OwnedOutputListValues) {
        let count = values.as_slice().len();
        debug_assert!(self.ensure_additional(count).is_ok());
        self.values.extend(values);
    }

    pub fn try_extract_output(
        &mut self,
        start: usize,
    ) -> Result<runmat_value::OutputListStorage, RuntimeError> {
        let count = self.values.len().checked_sub(start).ok_or_else(|| {
            mex(
                "StackWindow",
                "operand-stack extraction starts outside the stack",
            )
        })?;
        let (Some(receipt), Some(context)) = (&self.receipt, &self.context) else {
            let values = if start == 0 {
                std::mem::take(&mut self.values)
            } else {
                self.values.split_off(start)
            };
            return Ok(runmat_value::OutputListStorage::unbounded(values));
        };
        let mut drained = None;
        let mut source = Some(&mut self.values);
        let result = receipt
            .with_output_allocation(context, |permit| {
                runmat_value::OutputListStorage::try_initialize(count, permit, |_| {
                    if drained.is_none() {
                        let source = source.take().ok_or(
                            runmat_value::ExactHostNumericAllocationError::AllocationFailed,
                        )?;
                        drained = Some(source.drain(start..));
                    }
                    drained
                        .as_mut()
                        .and_then(Iterator::next)
                        .ok_or(runmat_value::ExactHostNumericAllocationError::AllocationFailed)
                })
                .map_err(|error| match error {
                    runmat_value::OutputListInitializationError::Allocation(error)
                    | runmat_value::OutputListInitializationError::Element(error) => error,
                })
            })
            .map_err(|error| {
                crate::runtime::workspace::frame_allocation_error_in_context(Some(context), error)
            })?;
        drop(drained);
        Ok(result)
    }

    /// Move a non-tail captured window into the canonical exact output owner.
    /// The source backing and its claim remain live while the separately
    /// admitted destination initializes; allocation denial precedes draining.
    pub(crate) fn try_extract_output_range(
        &mut self,
        start: usize,
        end: usize,
    ) -> Result<runmat_value::OutputListStorage, RuntimeError> {
        let context = self.context.clone();
        self.extract_output_range(start, end, context.as_ref())
    }

    /// Explicit-context ingress also funds extraction from an ordinary source
    /// stack. A managed source retains and validates its original authority.
    pub(crate) fn try_extract_output_range_in_context(
        &mut self,
        start: usize,
        end: usize,
        context: &RuntimeContext,
    ) -> Result<runmat_value::OutputListStorage, RuntimeError> {
        self.extract_output_range(start, end, Some(context))
    }

    fn extract_output_range(
        &mut self,
        start: usize,
        end: usize,
        context: Option<&RuntimeContext>,
    ) -> Result<runmat_value::OutputListStorage, RuntimeError> {
        if start > end || end > self.values.len() {
            return Err(mex(
                "StackWindow",
                "captured output range is outside the operand stack",
            ));
        }
        if let (Some(receipt), Some(context)) = (&self.receipt, context) {
            receipt.checkpoint(context).map_err(|source| {
                crate::runtime::workspace::frame_allocation_error_in_context(Some(context), source)
            })?;
        }
        let Some(context) = context else {
            return Ok(runmat_value::OutputListStorage::unbounded(
                self.values.drain(start..end).collect(),
            ));
        };
        let count = end - start;
        let allocation = |source| {
            crate::runtime::workspace::frame_allocation_error_in_context(Some(context), source)
        };
        let bytes = u64::try_from(
            runmat_value::OutputListStorage::requested_bytes(count)
                .map_err(HostAllocationError::Numeric)
                .map_err(allocation)?,
        )
        .map_err(|_| allocation(HostAllocationError::Overflow))?;
        let retained =
            HostAllocationPlan::retained_exact_output_bytes(bytes, usize::from(count != 0))
                .map_err(allocation)?;
        let receipt = HostAllocationReceipt::admit(
            Some(context),
            HostAllocationPlan::new(0, retained).map_err(allocation)?,
        )
        .map_err(allocation)?;
        let Some(receipt) = receipt else {
            return Ok(runmat_value::OutputListStorage::unbounded(
                self.values.drain(start..end).collect(),
            ));
        };
        let mut drained = None;
        let mut source = Some(&mut self.values);
        let result = receipt
            .with_output_construction(context, |permit| {
                runmat_value::OutputListStorage::try_initialize(count, permit, |_| {
                    if drained.is_none() {
                        drained = Some(
                            source
                                .take()
                                .expect("one range initializer")
                                .drain(start..end),
                        );
                    }
                    drained
                        .as_mut()
                        .and_then(Iterator::next)
                        .ok_or(runmat_value::ExactHostNumericAllocationError::AllocationFailed)
                })
            })
            .map_err(allocation)?
            .map_err(|error| match error {
                runmat_value::OutputListInitializationError::Allocation(source)
                | runmat_value::OutputListInitializationError::Element(source) => {
                    allocation(HostAllocationError::Numeric(source))
                }
            })?;
        drop(drained);
        receipt.checkpoint(context).map_err(allocation)?;
        Ok(result)
    }
}

fn stack_extent_error(message: &'static str) -> RuntimeError {
    runmat_runtime::execution::ExecutionServiceError::Infrastructure(format!(
        "RunMat:Interpreter:OperandStackExtent: {message}"
    ))
    .into_runtime_error()
}

impl Deref for InterpreterStack {
    type Target = [Value];

    fn deref(&self) -> &Self::Target {
        self.values.as_slice()
    }
}

impl Index<usize> for InterpreterStack {
    type Output = Value;
    fn index(&self, index: usize) -> &Value {
        &self.values[index]
    }
}
impl IndexMut<usize> for InterpreterStack {
    fn index_mut(&mut self, index: usize) -> &mut Value {
        &mut self.values[index]
    }
}

#[inline]
pub fn pop_value(
    stack: &mut crate::interpreter::stack::InterpreterStack,
) -> Result<Value, RuntimeError> {
    stack
        .pop()
        .ok_or_else(|| mex("StackUnderflow", "stack underflow"))
}

#[inline]
pub fn pop2(
    stack: &mut crate::interpreter::stack::InterpreterStack,
) -> Result<(Value, Value), RuntimeError> {
    let b = pop_value(stack)?;
    let a = pop_value(stack)?;
    Ok((a, b))
}

#[inline]
pub fn pop_args(
    stack: &mut crate::interpreter::stack::InterpreterStack,
    argc: usize,
) -> Result<Vec<Value>, RuntimeError> {
    let mut args = Vec::with_capacity(argc);
    for _ in 0..argc {
        args.push(pop_value(stack)?);
    }
    args.reverse();
    Ok(args)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_runtime::context::host_effect::ManagedHostEffectLedger;
    use runmat_runtime::context::{
        ParallelCapability, RuntimeParallelResources, RuntimeParallelService, RuntimeServicePorts,
    };
    use std::cell::Cell;
    use std::rc::Rc;

    struct ManagedParallel(ManagedHostEffectLedger);

    impl RuntimeParallelService for ManagedParallel {
        fn supports(&self, _: ParallelCapability) -> bool {
            false
        }

        fn placement_resources(&self) -> RuntimeParallelResources {
            let snapshot = self.0.snapshot();
            RuntimeParallelResources {
                memory_available_bytes: Some(snapshot.free_bytes()),
                epoch: snapshot.epoch,
                ..RuntimeParallelResources::default()
            }
        }

        fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
            Some(self.0.clone())
        }
    }

    #[test]
    fn managed_stack_claims_exact_backings_and_refuses_unbounded_or_excess_growth() {
        let mut ordinary_values = Vec::with_capacity(4);
        ordinary_values.extend([Value::Num(1.0), Value::Num(2.0)]);
        let ordinary_pointer = ordinary_values.as_ptr();
        let ordinary_capacity = ordinary_values.capacity();
        let mut ordinary_stack = InterpreterStack::from_unbounded(ordinary_values);
        let ordinary_output = ordinary_stack
            .try_extract_output(0)
            .expect("extract ordinary output");
        assert_eq!(ordinary_output.as_slice().as_ptr(), ordinary_pointer);
        let ordinary_values = ordinary_output
            .try_into_owned()
            .expect("ordinary output is uniquely owned")
            .try_into_unbounded_vec()
            .expect("ordinary output remains unbounded");
        assert_eq!(ordinary_values.as_ptr(), ordinary_pointer);
        assert_eq!(ordinary_values.capacity(), ordinary_capacity);

        let maximum = 2;
        let stack_layout = Layout::array::<Value>(maximum).expect("stack layout");
        let stack_retained =
            HostAllocationPlan::retained_exact_output_bytes(stack_layout.size() as u64, 1)
                .expect("stack retained bytes");
        let output_retained = HostAllocationPlan::retained_exact_output_bytes(
            runmat_value::OutputListStorage::requested_bytes(maximum).expect("output bytes") as u64,
            1,
        )
        .expect("output retained bytes");
        let total_retained = stack_retained + output_retained;
        let ledger = ManagedHostEffectLedger::new(total_retained);
        let context = RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ))
        .with_service_ports(
            RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
        );

        let reached_effect = Cell::new(false);
        let dynamic =
            InterpreterStack::prepare(&context, crate::bytecode::BytecodeStackGeometry::Dynamic)
                .map(|_| reached_effect.set(true));
        assert!(dynamic.is_err());
        assert!(!reached_effect.get());
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);

        let mut stack = InterpreterStack::prepare(
            &context,
            crate::bytecode::BytecodeStackGeometry::Exact { maximum },
        )
        .expect("prepare exact managed stack");
        assert_eq!(ledger.snapshot().live_reserved_bytes, total_retained);
        assert_eq!(ledger.snapshot().live_claimed_bytes, stack_retained);

        stack.try_push(Value::Num(1.0)).expect("first operand");
        stack.try_push(Value::Num(2.0)).expect("second operand");
        let moved = Cell::new(false);
        assert!(stack
            .try_push_with(|| {
                moved.set(true);
                Ok(Value::Num(3.0))
            })
            .is_err());
        assert!(!moved.get());
        assert!(stack.try_push(Value::Num(3.0)).is_err());
        assert_eq!(stack.as_slice(), &[Value::Num(1.0), Value::Num(2.0)]);

        // A separate ingress destination must be funded before stack drain.
        // The verified frame already reserves the entire finite envelope.
        assert!(stack
            .try_extract_output_range_in_context(0, 2, &context)
            .is_err());
        assert_eq!(stack.as_slice(), &[Value::Num(1.0), Value::Num(2.0)]);
        assert_eq!(ledger.snapshot().live_reserved_bytes, total_retained);
        assert_eq!(ledger.snapshot().live_claimed_bytes, stack_retained);

        let output = stack.try_extract_output(0).expect("extract output");
        assert!(stack.is_empty());
        assert_eq!(output.as_slice(), &[Value::Num(1.0), Value::Num(2.0)]);
        assert_eq!(ledger.snapshot().live_claimed_bytes, total_retained);
        drop(stack);
        assert_eq!(ledger.snapshot().live_reserved_bytes, output_retained);
        assert_eq!(ledger.snapshot().live_claimed_bytes, output_retained);
        drop(output);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);

        // The same frame admission owner funds CSL metadata; a slot move
        // retains its exact stack window, including on denied cancellation.
        let ledger = ManagedHostEffectLedger::new(65_536);
        let cancellation = runmat_execution::cancellation::CancellationSource::new();
        let context = RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ))
        .with_service_ports(
            RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
        )
        .with_cancellation_token(cancellation.token());
        let mut source = InterpreterStack::unbounded();
        source.try_push(Value::Num(3.0)).unwrap();
        source.try_push(Value::Num(4.0)).unwrap();
        let ingress = source
            .try_extract_output_range_in_context(0, 2, &context)
            .unwrap();
        assert!(source.is_empty());
        assert_eq!(ingress.as_slice(), &[Value::Num(3.0), Value::Num(4.0)]);
        assert!(ledger.snapshot().live_claimed_bytes > 0);
        drop(source);
        assert!(ledger.snapshot().live_claimed_bytes > 0);
        drop(ingress);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
        let mut sequences =
            crate::interpreter::dispatch::SequenceState::new_in_context(&context).unwrap();
        let mut operands = InterpreterStack::unbounded();
        let values = runmat_runtime::execution::RootedValues::from_output_storage_in_context(
            vec![Value::Num(7.0), Value::Num(8.0)].into(),
            &context,
        )
        .unwrap();
        sequences
            .capture_rooted_values(1, &mut operands, values)
            .unwrap();
        let pointer = operands.as_slice().as_ptr();
        sequences.move_capture(1, 2).unwrap();
        assert_eq!(operands.as_slice().as_ptr(), pointer);
        assert_eq!(operands.as_slice(), &[Value::Num(7.0), Value::Num(8.0)]);
        assert!(ledger.snapshot().live_claimed_bytes > 0);
        let claimed = ledger.snapshot().live_claimed_bytes;
        cancellation.cancel(runmat_execution::CancellationReason::User);
        assert!(sequences.move_capture(2, 3).is_err());
        assert_eq!(ledger.snapshot().live_claimed_bytes, claimed);
        assert_eq!(operands.as_slice().as_ptr(), pointer);
        drop(sequences);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);

        let ledger = ManagedHostEffectLedger::new(0);
        let context = RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ))
        .with_service_ports(
            RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
        );
        assert!(crate::interpreter::dispatch::SequenceState::new_in_context(&context).is_err());
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    }
}
