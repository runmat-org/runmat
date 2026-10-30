//! Canonical final input slots while original argument/root owners survive.
use super::*;
use crate::call::BuiltinArguments;
use crate::context::{
    host_allocation::{
        host_allocation_runtime_error, HostAllocationError, HostAllocationPlan,
        HostAllocationReceipt,
    },
    RuntimeContext,
};
use runmat_value::{ExactArrayStorage, ExactArrayStorageBuilder, OutputListStorage, SharedShape};

enum FixedBuilder {
    Exact(ExactArrayStorageBuilder<Option<Value>>),
    Ordinary(Vec<Option<Value>>),
}
impl FixedBuilder {
    fn slots(&self) -> &[Option<Value>] {
        match self {
            Self::Exact(s) => s.initialized(),
            Self::Ordinary(s) => s,
        }
    }
    fn slots_mut(&mut self) -> &mut [Option<Value>] {
        match self {
            Self::Exact(s) => s.initialized_mut(),
            Self::Ordinary(s) => s,
        }
    }
    fn finish(self) -> ExactArrayStorage<Option<Value>> {
        match self {
            Self::Exact(s) => s.try_finish().expect("all fixed input slots initialized"),
            Self::Ordinary(s) => {
                if s.capacity() == 0 {
                    ExactArrayStorage::empty()
                } else {
                    ExactArrayStorage::shared_unbounded(s)
                }
            }
        }
    }
}
pub async fn prepare_function_inputs_rooted(
    function_name: &str,
    supplied: BuiltinArguments,
    fixed_input_count: usize,
    accepts_varargin: bool,
    specs: &[FunctionInputSpec<'_>],
    runtime: Option<&RuntimeContext>,
) -> Result<PreparedFunctionInputs, RuntimeError> {
    if supplied.len() > fixed_input_count && !accepts_varargin {
        return Err(semantic_error(
            "TooManyInputs",
            format!(
                "semantic function {function_name} expected {fixed_input_count} inputs, got {}",
                supplied.len()
            ),
        ));
    }
    if specs
        .iter()
        .any(|spec| spec.input_index >= fixed_input_count)
    {
        return Err(semantic_error(
            "InvalidInputSlot",
            "function argument slot out of bounds",
        ));
    }
    let receipt = admit(
        runtime,
        ExactArrayStorage::<Option<Value>>::requested_bytes(fixed_input_count)
            .map_err(|e| allocation(runtime, e.into()))?,
        usize::from(fixed_input_count != 0),
    )?;
    let mut fixed = if let Some(receipt) = &receipt {
        let context = runtime.expect("selected managed context");
        let permit = receipt
            .output_permit(context)
            .map_err(|e| allocation(runtime, e))?;
        let mut builder = ExactArrayStorage::try_begin(fixed_input_count, &permit)
            .map_err(|e| allocation(runtime, e.into()))?;
        for index in 0..fixed_input_count {
            checkpoint(runtime, &Some(receipt))?;
            builder
                .try_push(supplied.get(index).cloned())
                .expect("fixed input count quoted");
        }
        FixedBuilder::Exact(builder)
    } else {
        FixedBuilder::Ordinary(
            (0..fixed_input_count)
                .map(|i| supplied.get(i).cloned())
                .collect(),
        )
    };
    for spec in specs {
        checkpoint(runtime, &receipt.as_ref())?;
        if fixed.slots()[spec.input_index].is_none() {
            if let Some(default) = spec.default_value {
                fixed.slots_mut()[spec.input_index] =
                    Some(default_value_in_context(default, runtime)?);
            }
        }
        if let Some(value) = fixed.slots()[spec.input_index].as_ref() {
            let result = validate_input(function_name, spec.input_index, value, spec).await;
            checkpoint(runtime, &receipt.as_ref())?;
            result?;
        }
    }
    let varargin = if accepts_varargin {
        let start = fixed_input_count.min(supplied.len());
        Some(pack_varargin(
            supplied
                .view()
                .range(start..)
                .expect("valid supplied suffix"),
            runtime,
        )?)
    } else {
        None
    };
    checkpoint(runtime, &receipt.as_ref())?;
    Ok(PreparedFunctionInputs {
        fixed: fixed.finish(),
        varargin,
        nargin: supplied.len(),
        _sources: supplied,
    })
}
fn pack_varargin(
    values: crate::call::FunctionArgumentView<'_>,
    runtime: Option<&RuntimeContext>,
) -> Result<CellArray, RuntimeError> {
    let shape = [1, values.len()];
    let bytes = OutputListStorage::requested_bytes(values.len())
        .map_err(|e| allocation(runtime, e.into()))?
        .checked_add(
            SharedShape::requested_bytes_for_rank(2).map_err(|e| allocation(runtime, e.into()))?,
        )
        .ok_or_else(|| allocation(runtime, HostAllocationError::Overflow))?;
    let receipt = admit(runtime, bytes, 1 + usize::from(!values.is_empty()))?;
    if let Some(receipt) = &receipt {
        let context = runtime.expect("selected managed context");
        let permit = receipt
            .output_permit(context)
            .map_err(|e| allocation(runtime, e))?;
        let shape = SharedShape::new_admitted_from_slice(&shape, &permit)
            .map_err(|e| allocation(runtime, e.into()))?;
        let mut slots = OutputListStorage::try_begin(values.len(), &permit)
            .map_err(|e| allocation(runtime, e.into()))?;
        for value in values {
            checkpoint(runtime, &Some(receipt))?;
            slots
                .try_push(value.clone())
                .expect("quoted varargin count");
        }
        let slots = slots.try_finish().expect("complete varargin slots");
        checkpoint(runtime, &Some(receipt))?;
        CellArray::from_admitted(slots, shape)
            .map_err(|e| semantic_error("VararginPack", format!("varargin: {e}")))
    } else {
        CellArray::new(values.iter().cloned().collect(), 1, values.len())
            .map_err(|e| semantic_error("VararginPack", format!("varargin: {e}")))
    }
}
fn default_value_in_context(
    default: &FunctionArgDefaultValue,
    runtime: Option<&RuntimeContext>,
) -> Result<Value, RuntimeError> {
    match default {
        FunctionArgDefaultValue::String(text) => {
            crate::call::argument_text::copy_str(text, runtime)
                .map(|text| Value::String(text.into()))
        }
        FunctionArgDefaultValue::EmptyArray => {
            let bytes =
                HostAllocationPlan::retained_tensor_bytes(runmat_value::NumericDType::F64, &[0, 0])
                    .map_err(|e| allocation(runtime, e))?;
            let receipt = HostAllocationReceipt::admit(
                runtime,
                HostAllocationPlan::new(0, bytes).map_err(|e| allocation(runtime, e))?,
            )
            .map_err(|e| allocation(runtime, e))?;
            if let Some(receipt) = &receipt {
                let context = runtime.expect("selected managed context");
                let tensor = receipt
                    .exact_output_tensor(context, runmat_value::NumericDType::F64, &[0, 0])
                    .map_err(|e| allocation(runtime, e))?
                    .into_tensor();
                checkpoint(runtime, &Some(receipt))?;
                tensor
                    .map(Value::Tensor)
                    .map_err(|e| allocation(runtime, e.into()))
            } else {
                Ok(super::default_value(default))
            }
        }
        _ => Ok(super::default_value(default)),
    }
}
fn admit(
    runtime: Option<&RuntimeContext>,
    bytes: usize,
    claims: usize,
) -> Result<Option<HostAllocationReceipt>, RuntimeError> {
    let output = HostAllocationPlan::retained_exact_output_bytes(
        u64::try_from(bytes).map_err(|_| allocation(runtime, HostAllocationError::Overflow))?,
        claims,
    )
    .map_err(|e| allocation(runtime, e))?;
    HostAllocationReceipt::admit(
        runtime,
        HostAllocationPlan::new(0, output).map_err(|e| allocation(runtime, e))?,
    )
    .map_err(|e| allocation(runtime, e))
}
fn checkpoint(
    runtime: Option<&RuntimeContext>,
    receipt: &Option<&HostAllocationReceipt>,
) -> Result<(), RuntimeError> {
    if let Some(receipt) = receipt {
        receipt
            .checkpoint(runtime.expect("selected managed context"))
            .map_err(|e| allocation(runtime, e))?;
    }
    Ok(())
}
fn allocation(runtime: Option<&RuntimeContext>, source: HostAllocationError) -> RuntimeError {
    host_allocation_runtime_error(
        runtime,
        "RunMat:Call:Inputs",
        "function input preparation failed",
        source,
    )
}
