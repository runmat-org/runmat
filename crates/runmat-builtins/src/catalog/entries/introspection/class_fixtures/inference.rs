use crate::catalog::inference::{argument_error, default_double_scalar, finish_fixed_outputs};
use crate::*;
use runmat_types::*;

const POINT_CLASS: StaticClassIdentity = StaticClassIdentity::new("Point");
const CTOR_CLASS: StaticClassIdentity = StaticClassIdentity::new("Ctor");

#[derive(Clone, Copy)]
enum ClassFixtureOperation {
    PointMove,
    PointOrigin,
    ShapeArea,
    CircleArea,
    CtorCtor,
    PkgFFoo,
    PkgGFoo,
}
impl ClassFixtureOperation {
    const fn arity(self) -> usize {
        match self {
            Self::PointMove => 3,
            Self::PointOrigin => 0,
            Self::ShapeArea => 1,
            Self::CircleArea => 1,
            Self::CtorCtor => 1,
            Self::PkgFFoo => 0,
            Self::PkgGFoo => 0,
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
    operation: ClassFixtureOperation,
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
    for index in match operation {
        ClassFixtureOperation::PointMove => &[1usize, 2][..],
        ClassFixtureOperation::CtorCtor => &[0usize][..],
        ClassFixtureOperation::PointOrigin
        | ClassFixtureOperation::ShapeArea
        | ClassFixtureOperation::CircleArea
        | ClassFixtureOperation::PkgFFoo
        | ClassFixtureOperation::PkgGFoo => &[][..],
    } {
        if let Some(v) = request.arguments.get(*index) {
            if !numeric(v) {
                diagnostics.push(argument_error(
                    "RM-FIXTURE-METHOD-NUMERIC",
                    "expected scalar double or native integer conversion",
                    *index,
                ));
            }
        }
    }
    if matches!(
        operation,
        ClassFixtureOperation::PointMove | ClassFixtureOperation::CircleArea
    ) {
        if let Some(v) = request.arguments.first() {
            if !value_object(v) {
                diagnostics.push(argument_error(
                    "RM-FIXTURE-METHOD-RECEIVER",
                    "requires value object receiver",
                    0,
                ));
            }
        }
    }
    let out = match operation {
        ClassFixtureOperation::PointMove => request
            .arguments
            .first()
            .filter(|v| matches!(v.kind, ValueKindFact::Object(_)))
            .map(|v| {
                let mut v = v.clone();
                v.alias = AliasFact::Unknown;
                if let ValueKindFact::Object(o) = &mut v.kind {
                    o.properties.insert("x".into(), default_double_scalar());
                    o.properties.insert("y".into(), default_double_scalar());
                }
                v
            })
            .unwrap_or_else(|| ValueFact::unknown(DynamicReason::DynamicDispatch)),
        ClassFixtureOperation::PointOrigin => {
            let mut v = object(POINT_CLASS);
            if let ValueKindFact::Object(o) = &mut v.kind {
                o.properties.insert("x".into(), default_double_scalar());
                o.properties.insert("y".into(), default_double_scalar());
                o.properties_complete = true;
            }
            v
        }
        ClassFixtureOperation::CtorCtor => {
            let mut v = object(CTOR_CLASS);
            if let ValueKindFact::Object(o) = &mut v.kind {
                o.properties.insert("x".into(), default_double_scalar());
                o.properties_complete = true;
            }
            v
        }
        ClassFixtureOperation::ShapeArea
        | ClassFixtureOperation::CircleArea
        | ClassFixtureOperation::PkgFFoo
        | ClassFixtureOperation::PkgGFoo => default_double_scalar(),
    };
    finish_fixed_outputs(entry, request, vec![out], diagnostics)
}

pub(super) fn point_move(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer(ClassFixtureOperation::PointMove, request, entry)
}
pub(super) fn point_origin(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer(ClassFixtureOperation::PointOrigin, request, entry)
}
pub(super) fn shape_area(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer(ClassFixtureOperation::ShapeArea, request, entry)
}
pub(super) fn circle_area(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer(ClassFixtureOperation::CircleArea, request, entry)
}
pub(super) fn ctor_ctor(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer(ClassFixtureOperation::CtorCtor, request, entry)
}
pub(super) fn pkgf_foo(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer(ClassFixtureOperation::PkgFFoo, request, entry)
}
pub(super) fn pkgg_foo(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer(ClassFixtureOperation::PkgGFoo, request, entry)
}
