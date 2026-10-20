use super::{allocation, ordinary_policy, root};
use crate::context::{
    host_allocation::{HostAllocationError, HostAllocationPolicy, HostAllocationReceipt},
    RuntimeContext,
};
use crate::execution::{RootedValue, RootedValueChildren, RootedValueMember};
use runmat_value::{StructValueBuilder, UnboundedStructValueBuilder, Value};

enum StructureOutput {
    Exact(StructValueBuilder),
    Ordinary(UnboundedStructValueBuilder),
}

pub(super) struct StructureCollector<'a> {
    // Drop partial payload before its transferred child roots and receipt.
    output: Option<StructureOutput>,
    roots: RootedValueChildren<'a>,
    receipt: Option<HostAllocationReceipt>,
    runtime: Option<&'a RuntimeContext>,
}

impl<'a> StructureCollector<'a> {
    pub(super) fn new<'n, I>(
        names: I,
        runtime: Option<&'a RuntimeContext>,
    ) -> crate::BuiltinResult<Self>
    where
        I: ExactSizeIterator<Item = &'n str> + Clone,
    {
        let receipt = crate::execution::structure::admit_struct_generation(runtime, names.clone())
            .map_err(|error| allocation(runtime, error))?;
        let roots = match runtime {
            Some(runtime) => RootedValueChildren::in_context(runtime)
                .map_err(|error| allocation(Some(runtime), error))?,
            None => RootedValueChildren::new(),
        };
        let output = if let Some(receipt) = &receipt {
            let runtime = runtime.expect("managed receipt has selected context");
            let permit = receipt
                .output_permit(runtime)
                .map_err(|error| allocation(Some(runtime), error))?;
            let builder = StructValueBuilder::try_new(names.len(), &permit);
            receipt
                .checkpoint(runtime)
                .map_err(|error| allocation(Some(runtime), error))?;
            StructureOutput::Exact(
                builder.map_err(|error| allocation(Some(runtime), error.into()))?,
            )
        } else {
            StructureOutput::Ordinary(UnboundedStructValueBuilder::with_capacity(names.len()))
        };
        Ok(Self {
            output: Some(output),
            roots,
            receipt,
            runtime,
        })
    }

    fn checkpoint(&self) -> crate::BuiltinResult<()> {
        if let Some(receipt) = &self.receipt {
            receipt
                .checkpoint(self.runtime.expect("managed runtime"))
                .map_err(|error| allocation(self.runtime, error))?;
        } else if let Some(runtime) = self.runtime {
            match HostAllocationReceipt::policy(Some(runtime))
                .map_err(|error| allocation(Some(runtime), error))?
            {
                HostAllocationPolicy::Unbounded => {}
                HostAllocationPolicy::Managed => {
                    return Err(allocation(
                        Some(runtime),
                        HostAllocationError::StaleAuthority,
                    ));
                }
            }
        } else {
            ordinary_policy()?;
        }
        Ok(())
    }

    pub(super) fn push(&mut self, field: &str, value: RootedValue) -> crate::BuiltinResult<()> {
        self.checkpoint()?;
        // Prepare the final key while the callback still owns its roots.
        let key = if let Some(receipt) = &self.receipt {
            let runtime = self.runtime.expect("managed runtime");
            let permit = receipt
                .output_permit(runtime)
                .map_err(|error| allocation(Some(runtime), error))?;
            Some(crate::execution::structure::copy_key(
                runtime, receipt, &permit, field,
            )?)
        } else {
            None
        };
        let value = RootedValueMember::from_rooted(value)
            .into_child(&mut self.roots)
            .map_err(|error| allocation(self.runtime, error))?;
        match self.output.as_mut().expect("unpublished structure") {
            StructureOutput::Exact(builder) => builder
                .try_insert_prepared(key.expect("admitted key"), value)
                .map_err(|error| {
                    crate::runtime_error::semantic_error_with_source(
                        "Structfun",
                        "structfun: output structure insertion failed",
                        error.into_parts().0,
                    )
                })?,
            StructureOutput::Ordinary(builder) => builder.insert(field, value),
        };
        self.checkpoint()
    }

    pub(super) fn finish(
        mut self,
        runtime: Option<&RuntimeContext>,
    ) -> crate::BuiltinResult<RootedValue> {
        self.checkpoint()?;
        let value = match self.output.take().expect("unpublished structure") {
            StructureOutput::Exact(builder) => builder.finish(),
            StructureOutput::Ordinary(builder) => builder.finish(),
        };
        // Existing roots remain in this collector through admitted publication.
        let output = root(Value::Struct(value), runtime)?;
        self.checkpoint()?;
        Ok(output)
    }
}
