mod analysis;
mod conversion;
mod parse;
mod placement;

use runmat_builtins::{BuiltinCatalogIdentity, BuiltinErrorDescriptor};
use runmat_value::Value;

use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[cfg(test)]
pub(super) use conversion::real_to_complex;
pub(super) use parse::parse_output_template;

#[derive(Clone, Copy)]
pub(super) struct OutputPrototypeContext {
    pub identity: BuiltinCatalogIdentity,
    pub invalid_argument: &'static BuiltinErrorDescriptor,
    pub invalid_input: &'static BuiltinErrorDescriptor,
}

impl OutputPrototypeContext {
    pub(super) fn described_error(
        self,
        descriptor: &'static BuiltinErrorDescriptor,
        detail: impl AsRef<str>,
    ) -> RuntimeError {
        let mut builder =
            build_runtime_error(format!("{}: {}", descriptor.message, detail.as_ref()))
                .with_builtin(self.identity.name);
        if let Some(identifier) = descriptor.identifier {
            builder = builder.with_identifier(identifier);
        }
        builder.build()
    }

    pub(super) fn internal_error(self, detail: impl std::fmt::Display) -> RuntimeError {
        build_runtime_error(format!("{}: {detail}", self.identity.name))
            .with_builtin(self.identity.name)
            .build()
    }
}

#[derive(Clone)]
pub(super) enum OutputTemplate {
    Default,
    Like(Value),
}

pub(super) async fn apply_output_template(
    context: OutputPrototypeContext,
    value: Value,
    template: &OutputTemplate,
) -> BuiltinResult<Value> {
    match template {
        OutputTemplate::Default => Ok(value),
        OutputTemplate::Like(prototype) => {
            let analysis = analysis::analyse(context, prototype).await?;
            match analysis.class {
                analysis::PrototypeClass::Real => {
                    placement::ensure(context, value, &analysis.device).await
                }
                analysis::PrototypeClass::Complex => {
                    let host_value =
                        placement::ensure(context, value, &analysis::DevicePreference::Host)
                            .await?;
                    conversion::real_to_complex(context, host_value).await
                }
            }
        }
    }
}
