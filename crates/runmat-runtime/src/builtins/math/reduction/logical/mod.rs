pub(super) mod arguments;
mod gpu;
pub(super) mod host;
mod shape;

use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_builtins::{
    BuiltinCatalogEntry, BuiltinErrorDescriptor, BuiltinExtensionDescriptor, LogicalReductionKind,
};
use runmat_value::Value;

pub(super) struct LogicalReductionConfig {
    pub(super) entry: &'static BuiltinCatalogEntry,
    pub(super) kind: LogicalReductionKind,
    pub(super) invalid_argument: &'static BuiltinErrorDescriptor,
    pub(super) invalid_input: &'static BuiltinErrorDescriptor,
    pub(super) internal: &'static BuiltinErrorDescriptor,
    pub(super) too_many_outputs: &'static BuiltinErrorDescriptor,
    pub(super) nanflag_extension: &'static BuiltinExtensionDescriptor,
}

impl LogicalReductionConfig {
    pub(super) fn name(&self) -> &'static str {
        self.entry.identity.name
    }

    pub(super) fn default_nan_mode(&self) -> crate::builtins::common::spec::ReductionNaN {
        match self.kind {
            LogicalReductionKind::All => crate::builtins::common::spec::ReductionNaN::Include,
            LogicalReductionKind::Any => crate::builtins::common::spec::ReductionNaN::Omit,
        }
    }

    pub(super) fn error(
        &self,
        descriptor: &'static BuiltinErrorDescriptor,
        detail: impl std::fmt::Display,
    ) -> RuntimeError {
        let mut builder = build_runtime_error(format!("{}: {detail}", descriptor.message))
            .with_builtin(self.name());
        if let Some(identifier) = descriptor.identifier {
            builder = builder.with_identifier(identifier);
        }
        builder.build()
    }
}

pub(super) async fn execute(
    config: &'static LogicalReductionConfig,
    value: Value,
    arguments: Vec<Value>,
) -> BuiltinResult<Value> {
    if matches!(crate::output_count::current_output_count(), Some(count) if count > 1) {
        return Err(config.error(config.too_many_outputs, "only one output is defined"));
    }
    let (spec, nan_mode) = arguments::parse(config, &arguments).await?;
    match value {
        Value::GpuTensor(handle) => gpu::reduce(config, handle, spec, nan_mode).await,
        value => host::reduce(config, value, spec, nan_mode).await,
    }
}
