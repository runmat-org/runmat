use super::*;
use crate::{BuiltinAcceleratorPolicy, BuiltinCatalogEntry, BuiltinParamType};
use std::collections::BTreeSet;

/// Validate declaration consistency only. Runtime option admission stays in its
/// domain; no literal values, inference rules or builtin spellings are examined.
pub fn validate_builtin_preparation(entry: &BuiltinCatalogEntry) -> Vec<&'static str> {
    let BuiltinOperandPreparation::Forms(forms) = entry.placement.preparation else {
        return Vec::new();
    };
    let mut errors = Vec::new();
    if forms.is_empty() {
        errors.push("preparation Forms must not be empty; use RuntimeOwned explicitly");
    }
    let mut ids = BTreeSet::new();
    for form in forms {
        if matches!(form.operation, BuiltinPreparationOperation::FloatingCast(class) if !matches!(class, runmat_types::NumericClass::Single | runmat_types::NumericClass::Double))
        {
            errors.push("floating cast preparation requires Single or Double target");
        }
        if matches!(form.operation, BuiltinPreparationOperation::ElementwiseNary { data_arity } if data_arity < 2)
        {
            errors.push("N-ary preparation requires at least two data operands");
        }
        if !ids.insert(form.id) {
            errors.push("preparation form IDs must be unique within an identity");
        }
        if !entry
            .bindings
            .iter()
            .any(|binding| binding.variant == form.binding_variant)
        {
            errors.push("preparation form refers to an undeclared binding variant");
        }
        if form.signatures.is_empty() || !form.signatures.windows(2).all(|p| p[0] < p[1]) {
            errors.push("preparation signature references must be nonempty, sorted and unique");
        }
        if !form.data_arguments.windows(2).all(|p| p[0] < p[1]) {
            errors.push("preparation data positions must be sorted and unique");
        }
        if form
            .operation
            .data_arity()
            .is_some_and(|n| n != form.data_arguments.len())
        {
            errors.push("preparation operation and data operand count disagree");
        }
        if entry.placement.accelerator == BuiltinAcceleratorPolicy::Forbidden
            && !matches!(form.operation, BuiltinPreparationOperation::Sink(_))
        {
            errors.push("accelerator-forbidden builtin cannot declare automatic promotion");
        }
        if matches!(form.operation, BuiltinPreparationOperation::Sink(_))
            && form.provider_requirement != BuiltinPreparationProviderRequirement::InputSupported
        {
            errors.push("sink preparation cannot require a floating provider precision");
        }
        if matches!(form.operation, BuiltinPreparationOperation::RuntimeOwned)
            && form.provider_requirement != BuiltinPreparationProviderRequirement::InputSupported
        {
            errors.push("runtime-owned preparation has no provider precision requirement");
        }
        for &signature in form.signatures {
            let Some(signature) = entry.descriptor.signatures.get(signature) else {
                errors.push("preparation references an unknown descriptor signature");
                continue;
            };
            for &position in form.data_arguments {
                let Some(parameter) = signature.inputs.get(position) else {
                    errors.push("preparation data position is outside its descriptor signature");
                    continue;
                };
                if matches!(
                    parameter.ty,
                    BuiltinParamType::StringScalar
                        | BuiltinParamType::SizeArg
                        | BuiltinParamType::LikePrototype
                        | BuiltinParamType::AxesHandle
                        | BuiltinParamType::StyleSpec
                        | BuiltinParamType::PropertyName
                        | BuiltinParamType::Callable
                ) {
                    errors.push("preparation marks a declared control parameter as numeric data");
                }
            }
        }
    }
    errors
}
