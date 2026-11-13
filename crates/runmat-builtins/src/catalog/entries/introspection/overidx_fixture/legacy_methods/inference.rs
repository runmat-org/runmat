use crate::catalog::inference::{argument_error, default_double_scalar, finish_fixed_outputs};
use crate::*;
use runmat_types::*;

const OVERIDX_CLASS: StaticClassIdentity = StaticClassIdentity::new("OverIdx");

#[derive(Clone, Copy)]
enum OverIdxOperation {
    NumArguments,
    SaveObj,
    LoadObj,
    Plus,
    Times,
    Mtimes,
    Lt,
    Gt,
    Eq,
    Uplus,
    Rdivide,
    Mrdivide,
    Ldivide,
    Mldivide,
    And,
    Or,
    Xor,
}
impl OverIdxOperation {
    const fn arity(self) -> usize {
        match self {
            Self::NumArguments => 3,
            Self::SaveObj => 1,
            Self::LoadObj => 1,
            Self::Plus => 2,
            Self::Times => 2,
            Self::Mtimes => 2,
            Self::Lt => 2,
            Self::Gt => 2,
            Self::Eq => 2,
            Self::Uplus => 1,
            Self::Rdivide => 2,
            Self::Mrdivide => 2,
            Self::Ldivide => 2,
            Self::Mldivide => 2,
            Self::And => 2,
            Self::Or => 2,
            Self::Xor => 2,
        }
    }
}

fn object(class: StaticClassIdentity) -> ValueFact {
    ValueFact::scalar(ValueKindFact::Object(ObjectFact {
        class: None,
        runtime_class: Some(class.owned()),
        properties: Default::default(),
        properties_complete: false,
        handle_semantics: Some(false),
    }))
}
fn numeric(value: &ValueFact) -> bool {
    matches!(
        value.kind,
        ValueKindFact::Unknown
            | ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Real,
                ..
            })
    ) && !matches!(value.shape.element_count(),Some(n) if n!=1)
        && !matches!(
            value.residency,
            ResidencyFact::Device { .. } | ResidencyFact::Remote { .. }
        )
}
fn value_object(value: &ValueFact) -> bool {
    matches!(value.kind, ValueKindFact::Unknown)
        || matches!(&value.kind,ValueKindFact::Object(o) if o.handle_semantics!=Some(true) && !matches!(value.shape.element_count(),Some(n) if n!=1) && !matches!(value.residency,ResidencyFact::Device{..}|ResidencyFact::Remote{..}))
}
fn infer(
    operation: OverIdxOperation,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let mut diagnostics = vec![];
    let arity = operation.arity();
    if request.arguments.len() != arity {
        diagnostics.push(argument_error(
            "RM-FIXTURE-METHOD-ARITY",
            "fixed method arity mismatch",
            request.arguments.len(),
        ));
    }
    if matches!(
        operation,
        OverIdxOperation::NumArguments | OverIdxOperation::SaveObj
    ) {
        if let Some(v) = request.arguments.first() {
            if !value_object(v) {
                diagnostics.push(argument_error(
                    "RM-OVERIDX-RECEIVER",
                    "requires value object receiver",
                    0,
                ));
            }
        }
    } else if !matches!(
        operation,
        OverIdxOperation::Uplus | OverIdxOperation::LoadObj
    ) && request.arguments.len() == 2
    {
        let a = &request.arguments[0];
        let b = &request.arguments[1];
        let valid = matches!(a.kind, ValueKindFact::Unknown)
            || matches!(b.kind, ValueKindFact::Unknown)
            || (matches!(a.kind, ValueKindFact::Object(_)) && value_object(a) && numeric(b))
            || (matches!(b.kind, ValueKindFact::Object(_)) && value_object(b) && numeric(a));
        if !valid {
            diagnostics.push(argument_error(
                "RM-OVERIDX-OPERANDS",
                "requires one value object and one scalar double/native integer",
                0,
            ));
        }
    }
    let out = match operation {
        OverIdxOperation::Uplus => request
            .arguments
            .first()
            .cloned()
            .unwrap_or_else(|| ValueFact::unknown(DynamicReason::DynamicDispatch)),
        OverIdxOperation::NumArguments => match request.arguments.first().map(|v| &v.kind) {
            Some(ValueKindFact::Object(o)) => {
                o.properties.get("nargs").cloned().unwrap_or_else(|| {
                    if o.properties_complete {
                        default_double_scalar()
                    } else {
                        ValueFact::unknown(DynamicReason::DynamicDispatch)
                    }
                })
            }
            _ => ValueFact::unknown(DynamicReason::DynamicDispatch),
        },
        OverIdxOperation::SaveObj => {
            let mut fields = std::collections::BTreeMap::new();
            let complete = if let Some(ValueKindFact::Object(o)) =
                request.arguments.first().map(|v| &v.kind)
            {
                for key in ["k", "nargs"] {
                    if let Some(v) = o.properties.get(key) {
                        fields.insert(key.into(), v.clone());
                    }
                }
                o.properties_complete
            } else {
                false
            };
            fields.insert("saved_by".into(), ValueFact::scalar(ValueKindFact::String));
            ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(fields, complete)))
        }
        OverIdxOperation::LoadObj => {
            let mut v = object(OVERIDX_CLASS);
            if let ValueKindFact::Object(o) = &mut v.kind {
                if let Some(ValueKindFact::Struct(s)) = request.arguments.first().map(|v| &v.kind) {
                    o.properties = s
                        .fields
                        .iter()
                        .filter(|(k, _)| k.as_str() != "saved_by")
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect();
                    o.properties_complete = s.fields_complete;
                } else {
                    o.properties_complete = !matches!(
                        request.arguments.first().map(|v| &v.kind),
                        Some(ValueKindFact::Unknown) | None
                    );
                }
                o.properties
                    .insert("loaded_by".into(), ValueFact::scalar(ValueKindFact::String));
            }
            v
        }
        OverIdxOperation::Plus
        | OverIdxOperation::Times
        | OverIdxOperation::Mtimes
        | OverIdxOperation::Lt
        | OverIdxOperation::Gt
        | OverIdxOperation::Eq
        | OverIdxOperation::Rdivide
        | OverIdxOperation::Mrdivide
        | OverIdxOperation::Ldivide
        | OverIdxOperation::Mldivide
        | OverIdxOperation::And
        | OverIdxOperation::Or
        | OverIdxOperation::Xor => default_double_scalar(),
    };
    finish_fixed_outputs(entry, request, vec![out], diagnostics)
}

pub(super) fn num_arguments(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer(OverIdxOperation::NumArguments, request, entry)
}
pub(super) fn saveobj(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer(OverIdxOperation::SaveObj, request, entry)
}
pub(super) fn loadobj(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer(OverIdxOperation::LoadObj, request, entry)
}
pub(super) fn plus(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer(OverIdxOperation::Plus, request, entry)
}
pub(super) fn times(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer(OverIdxOperation::Times, request, entry)
}
pub(super) fn mtimes(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer(OverIdxOperation::Mtimes, request, entry)
}
pub(super) fn lt(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer(OverIdxOperation::Lt, request, entry)
}
pub(super) fn gt(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer(OverIdxOperation::Gt, request, entry)
}
pub(super) fn eq(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer(OverIdxOperation::Eq, request, entry)
}
pub(super) fn uplus(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer(OverIdxOperation::Uplus, request, entry)
}
pub(super) fn rdivide(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer(OverIdxOperation::Rdivide, request, entry)
}
pub(super) fn mrdivide(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer(OverIdxOperation::Mrdivide, request, entry)
}
pub(super) fn ldivide(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer(OverIdxOperation::Ldivide, request, entry)
}
pub(super) fn mldivide(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer(OverIdxOperation::Mldivide, request, entry)
}
pub(super) fn and(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer(OverIdxOperation::And, request, entry)
}
pub(super) fn or(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer(OverIdxOperation::Or, request, entry)
}
pub(super) fn xor(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer(OverIdxOperation::Xor, request, entry)
}
