//! Canonical solver metadata checks, with explicit opaque-container facts.
use super::{build_runtime_error, BuiltinResult};
use runmat_value::Value;
/// Reuse the canonical input contract before provider work. Payload values,
/// finite checks and authentic resident allocation admission remain runtime-owned.
pub(crate) fn preflight_catalog_inputs<'a>(
    entry: &'static runmat_builtins::BuiltinCatalogEntry,
    error: &'static runmat_builtins::BuiltinErrorDescriptor,
    values: impl IntoIterator<Item = &'a Value>,
) -> BuiltinResult<()> {
    preflight_catalog_facts(
        entry,
        error,
        values.into_iter().map(crate::value_fact::value_fact),
    )
}

/// The same catalog check for runtime-owned role projections. Opaque option
/// containers can report actual scalar geometry without walking dormant fields.
pub(crate) fn preflight_catalog_facts(
    entry: &'static runmat_builtins::BuiltinCatalogEntry,
    error: &'static runmat_builtins::BuiltinErrorDescriptor,
    facts: impl IntoIterator<Item = runmat_types::ValueFact>,
) -> BuiltinResult<()> {
    let inference = runmat_builtins::infer_catalog_call(
        entry,
        &runmat_types::CallRequest {
            arguments: facts.into_iter().collect(),
            literals: runmat_types::LiteralContext::default(),
            outputs: runmat_types::OutputSelection::new(
                runmat_types::RequestedOutputCount::Exactly(1),
            ),
        },
    );
    if let Some(diagnostic) = inference
        .diagnostics
        .into_iter()
        .find(|diagnostic| diagnostic.severity == runmat_types::InferenceSeverity::Error)
    {
        let mut builder =
            build_runtime_error(format!("{}: {}", entry.identity.name, diagnostic.message))
                .with_builtin(entry.identity.name);
        if let Some(identifier) = error.identifier {
            builder = builder.with_identifier(identifier);
        }
        return Err(builder.build());
    }
    Ok(())
}
