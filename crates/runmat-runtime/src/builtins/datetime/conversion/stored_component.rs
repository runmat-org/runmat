//! Stored-datetime numeric components share borrowed input and exact output.
use crate::context::host_allocation::{
    host_allocation_runtime_error, HostAllocationCheckpoint, HostAllocationError,
    HostAllocationPolicy, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy, RootedValueLease};
use crate::{BuiltinResult, GpuGatherRetry, RuntimeError};
use chrono::NaiveDateTime;
use runmat_value::Value;
mod input;
mod legacy;
mod output;
pub(in crate::builtins::datetime) use input::SerialView;
pub(in crate::builtins::datetime) use legacy::PreparedControl;
pub(in crate::builtins::datetime) use legacy::{map_legacy_component, ComponentKind};

pub(in crate::builtins::datetime) struct Work<'a> {
    runtime: Option<&'a RuntimeContext>,
    label: &'static str,
    owner: Option<HostAllocationCheckpoint>,
    budget: Option<CpuCooperationBudget>,
    quantum: u64,
    remaining: u64,
    pending: u64,
}
impl<'a> Work<'a> {
    fn new(runtime: Option<&'a RuntimeContext>, label: &'static str) -> BuiltinResult<Self> {
        let owner = HostAllocationReceipt::capture_owner(runtime)
            .map_err(|source| allocation_error(runtime, label, source))?;
        Self::with_owner(runtime, label, owner)
    }
    pub(in crate::builtins::datetime) fn with_owner(
        runtime: Option<&'a RuntimeContext>,
        label: &'static str,
        owner: Option<HostAllocationCheckpoint>,
    ) -> BuiltinResult<Self> {
        let quantum = runtime.map_or(256, |runtime| {
            match runtime.execution().cpu_cooperation_policy() {
                CpuCooperationPolicy::NeverYield => 256,
                CpuCooperationPolicy::WorkQuantum(quantum) => quantum.get(),
            }
        });
        let work = Self {
            runtime,
            label,
            owner,
            budget: runtime.map(RuntimeContext::cpu_cooperation),
            quantum,
            remaining: quantum,
            pending: 0,
        };
        work.check()?;
        Ok(work)
    }
    pub(in crate::builtins::datetime) fn check(&self) -> BuiltinResult<()> {
        if let Some(owner) = &self.owner {
            owner.check(
                self.runtime
                    .expect("captured allocation owner has a context"),
            )
        } else {
            HostAllocationReceipt::policy(self.runtime).and_then(|policy| match policy {
                HostAllocationPolicy::Unbounded => Ok(()),
                HostAllocationPolicy::Managed => Err(HostAllocationError::StaleAuthority),
            })
        }
        .map_err(|source| allocation_error(self.runtime, self.label, source))
    }
    fn capacity(&self) -> usize {
        (self.remaining.min(256) - self.pending) as usize
    }
    pub(in crate::builtins::datetime) async fn charge(
        &mut self,
        count: usize,
    ) -> BuiltinResult<()> {
        assert!(
            count <= self.capacity(),
            "calendar work must stay inside its current quantum"
        );
        self.pending += count as u64;
        if self.pending == self.remaining.min(256) {
            self.flush().await?;
        }
        Ok(())
    }
    pub(in crate::builtins::datetime) async fn flush(&mut self) -> BuiltinResult<()> {
        self.check()?;
        if let Some(budget) = &mut self.budget {
            budget.checkpoint(self.pending).await.map_err(|source| {
                never_retry(super::super::datetime_execution_error(self.runtime, source))
            })?;
        }
        self.check()?;
        self.remaining -= self.pending;
        if self.remaining == 0 {
            self.remaining = self.quantum;
        }
        self.pending = 0;
        Ok(())
    }

    pub(in crate::builtins::datetime) async fn cooperate(
        &mut self,
        count: u64,
    ) -> BuiltinResult<()> {
        self.flush().await?;
        self.check()?;
        if let Some(budget) = &mut self.budget {
            budget.checkpoint(count).await.map_err(|source| {
                never_retry(super::super::datetime_execution_error(self.runtime, source))
            })?;
        }
        self.check()
    }
}

fn allocation_error(
    runtime: Option<&RuntimeContext>,
    label: &str,
    source: HostAllocationError,
) -> RuntimeError {
    never_retry(host_allocation_runtime_error(
        runtime,
        "RunMat:datetime:ComponentOutput",
        format!("{label}: stored calendar component allocation failed; {source}"),
        source,
    ))
}

fn never_retry(mut error: RuntimeError) -> RuntimeError {
    error.gpu_gather_retry = GpuGatherRetry::Never;
    error
}

pub(in crate::builtins::datetime) async fn map(
    runtime: Option<&RuntimeContext>,
    value: &Value,
    label: &'static str,
    extract: impl Fn(&NaiveDateTime) -> f64,
) -> BuiltinResult<Value> {
    let mut work = Work::new(runtime, label)?;
    let _roots = if let Some(runtime) = runtime {
        RootedValueLease::for_values_iterator_in_context(std::iter::once(value), runtime)
            .map_err(never_retry)?
    } else {
        RootedValueLease::for_values(std::iter::once(value))
            .map_err(|source| never_retry(source.into_runtime_error()))?
    };
    work.check()?;
    let source = SerialView::borrow(value)?;
    if source.len() == 1 {
        work.charge(1).await?;
        let naive = super::super::naive_from_datenum(source.at(0))?;
        let result = Value::Num(extract(&naive));
        work.flush().await?;
        work.check()?;
        return Ok(result);
    }
    let SerialView::Tensor(tensor) = &source else {
        unreachable!("inline serials are scalar")
    };
    // Keep the canonical receipt outside the output enum. Its borrowed
    // capability remains live through construction, all awaits and finish.
    let receipt = output::admit(&work, tensor.shape(), 0)?;
    let mut output = output::Output::new(&work, tensor.shape(), tensor.len(), receipt.as_ref())?;
    let mut start = 0;
    while start < source.len() {
        output.check(&work)?;
        let count = work.capacity().min(source.len() - start);
        for index in start..start + count {
            let naive = super::super::naive_from_datenum(source.at(index))?;
            output.set(index, extract(&naive));
        }
        work.charge(count).await?;
        output.check(&work)?;
        start += count;
    }
    work.flush().await?;
    output.check(&work)?;
    output.finish(&work)
}
