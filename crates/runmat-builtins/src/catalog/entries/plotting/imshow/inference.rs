use crate::BuiltinCatalogEntry;
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut inference = super::super::coordinate_inference::infer(
        request,
        entry,
        1,
        super::super::coordinate_inference::Handles::Scalar,
    );
    if request.arguments.len() > 2 {
        inference
            .diagnostics
            .push(crate::catalog::inference::argument_error(
                "RM-CATALOG-IMSHOW-ARITY",
                "imshow accepts an image and optional grayscale display range",
                2,
            ));
    }
    // Numeric/logical forms never read the filesystem. Unknown values keep the
    // real possible-service union until the runtime filename discriminator runs.
    if request
        .arguments
        .first()
        .is_some_and(|arg| matches!(arg.kind, ValueKindFact::Numeric(_) | ValueKindFact::Logical))
    {
        inference.effects.0.remove(&EffectKind::FilesystemRead);
        inference
            .capabilities
            .0
            .remove(&CapabilityRequirement::Filesystem);
    }
    inference
}
