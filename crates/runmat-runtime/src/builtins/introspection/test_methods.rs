//! Executable class/method fixtures shared by semantic and VM pipeline tests.

use crate::execution::RootedValue;
use crate::{runtime_descriptor_error, runtime_descriptor_error_with_detail};
use runmat_builtins::{
    CIRCLE_AREA_ERROR_RECEIVER_INVALID, OVERIDX_ERROR_RECEIVER_INVALID,
    OVERIDX_ERROR_SUBSASGN_PAYLOAD_UNSUPPORTED, OVERIDX_ERROR_SUBSREF_PAYLOAD_UNSUPPORTED,
    POINT_MOVE_ERROR_RECEIVER_INVALID,
};
pub(crate) mod canonical_bindings;
use runmat_value::Value;

pub(crate) async fn point_move_method(obj: Value, dx: f64, dy: f64) -> crate::BuiltinResult<Value> {
    match obj {
        Value::Object(o) => {
            let mut x = 0.0;
            let mut y = 0.0;
            if let Some(Value::Num(v)) = o.property("x") {
                x = *v;
            }
            if let Some(Value::Num(v)) = o.property("y") {
                y = *v;
            }
            let mut o = o.into_unbounded_builder().map_err(|_| {
                runtime_descriptor_error("Point.move", &POINT_MOVE_ERROR_RECEIVER_INVALID)
            })?;
            o.insert("x".to_string(), Value::Num(x + dx));
            o.insert("y".to_string(), Value::Num(y + dy));
            Ok(Value::Object(o.finish()))
        }
        other => Err(runtime_descriptor_error_with_detail(
            "Point.move",
            &POINT_MOVE_ERROR_RECEIVER_INVALID,
            format!("got {other:?}"),
        )),
    }
}

pub(crate) async fn point_origin_method() -> crate::BuiltinResult<Value> {
    let mut o = runmat_value::UnboundedObjectBuilder::new("Point");
    o.insert("x".to_string(), Value::Num(0.0));
    o.insert("y".to_string(), Value::Num(0.0));
    Ok(Value::Object(o.finish()))
}

pub(crate) async fn shape_area_method(_obj: Value) -> crate::BuiltinResult<Value> {
    Ok(Value::Num(0.0))
}

pub(crate) async fn circle_area_method(obj: Value) -> crate::BuiltinResult<Value> {
    match obj {
        Value::Object(o) => {
            let r = if let Some(Value::Num(v)) = o.property("r") {
                *v
            } else {
                0.0
            };
            Ok(Value::Num(std::f64::consts::PI * r * r))
        }
        other => Err(runtime_descriptor_error_with_detail(
            "Circle.area",
            &CIRCLE_AREA_ERROR_RECEIVER_INVALID,
            format!("got {other:?}"),
        )),
    }
}

// --- Test-only helpers to validate constructors and subsref/subsasgn ---

pub(crate) async fn ctor_ctor_method(x: f64) -> crate::BuiltinResult<Value> {
    // Construct object with property 'x' initialized
    let mut o = runmat_value::UnboundedObjectBuilder::new("Ctor");
    o.insert("x".to_string(), Value::Num(x));
    Ok(Value::Object(o.finish()))
}

// --- Test-only package functions to exercise import precedence ---

pub(crate) async fn pkgf_foo() -> crate::BuiltinResult<Value> {
    Ok(Value::Num(10.0))
}

pub(crate) async fn pkgg_foo() -> crate::BuiltinResult<Value> {
    Ok(Value::Num(20.0))
}

#[runmat_macros::runtime_builtin(
    name = "OverIdx.subsref",
    binding_variant = "default",
    builtin_path = "crate::builtins::introspection::test_methods"
)]
pub(crate) async fn overidx_subsref(
    obj: Value,
    subscript: Value,
) -> crate::BuiltinResult<RootedValue> {
    let path = crate::object::indexing::parse_standard_substruct(&subscript)?;
    crate::object::protocol::execute_owned_subsref(obj, path, overidx_read_step, None).await
}

async fn overidx_read_step(
    obj: Value,
    step: crate::object::indexing::ObjectSubscript,
) -> crate::BuiltinResult<RootedValue> {
    let payload = step.selector_value()?;
    // Simple sentinel implementation: return different values for '.' vs '()'
    let value = match (obj, step.kind(), payload) {
        (Value::Object(_), crate::object::indexing::ObjectIndexKind::Paren, Value::Cell(_)) => {
            Ok(Value::Num(99.0))
        }
        (Value::Object(o), crate::object::indexing::ObjectIndexKind::Brace, Value::Cell(_)) => {
            if let Some(v) = o.property("lastCell") {
                Ok(v.clone())
            } else {
                Ok(Value::Num(0.0))
            }
        }
        (
            Value::Object(o),
            crate::object::indexing::ObjectIndexKind::Member,
            Value::String(runmat_value::StringElement::Text(field)),
        ) => {
            // If field exists, return it; otherwise sentinel 77
            if let Some(v) = o.property(field.as_str()) {
                Ok(v.clone())
            } else {
                Ok(Value::Num(77.0))
            }
        }
        (
            Value::Object(o),
            crate::object::indexing::ObjectIndexKind::Member,
            Value::CharArray(ca),
        ) => {
            let field = super::text::decode_utf16_with_error(ca.data(), || {
                runtime_descriptor_error_with_detail(
                    "OverIdx.subsref",
                    &OVERIDX_ERROR_SUBSREF_PAYLOAD_UNSUPPORTED,
                    "member name is not valid UTF-16 text",
                )
            })?;
            if let Some(v) = o.property(&field) {
                Ok(v.clone())
            } else {
                Ok(Value::Num(77.0))
            }
        }
        _ => Err(runtime_descriptor_error(
            "OverIdx.subsref",
            &OVERIDX_ERROR_SUBSREF_PAYLOAD_UNSUPPORTED,
        )),
    }?;
    RootedValue::new(value).map_err(|error| error.into_runtime_error())
}

#[runmat_macros::runtime_builtin(
    name = "OverIdx.subsasgn",
    binding_variant = "default",
    builtin_path = "crate::builtins::introspection::test_methods"
)]
pub(crate) async fn overidx_subsasgn(
    obj: Value,
    subscript: Value,
    rhs: Value,
) -> crate::BuiltinResult<RootedValue> {
    let path = crate::object::indexing::parse_standard_substruct(&subscript)?;
    crate::object::protocol::execute_owned_subsasgn(
        obj,
        path,
        vec![rhs],
        overidx_read_step,
        |obj, step, mut values| async move {
            let rhs = values.pop().ok_or_else(|| {
                runtime_descriptor_error(
                    "OverIdx.subsasgn",
                    &OVERIDX_ERROR_SUBSASGN_PAYLOAD_UNSUPPORTED,
                )
            })?;
            overidx_write_step(obj, step, rhs).await
        },
        None,
    )
    .await
}

async fn overidx_write_step(
    obj: Value,
    step: crate::object::indexing::ObjectSubscript,
    rhs: Value,
) -> crate::BuiltinResult<RootedValue> {
    let payload = step.selector_value()?;
    let Value::Object(object) = obj else {
        return Err(runtime_descriptor_error(
            "OverIdx.subsasgn",
            &OVERIDX_ERROR_SUBSASGN_PAYLOAD_UNSUPPORTED,
        ));
    };
    let mut object = object.into_unbounded_builder().map_err(|_| {
        runtime_descriptor_error(
            "OverIdx.subsasgn",
            &OVERIDX_ERROR_SUBSASGN_PAYLOAD_UNSUPPORTED,
        )
    })?;
    match (step.kind(), payload) {
        (crate::object::indexing::ObjectIndexKind::Paren, Value::Cell(_)) => {
            // Store into 'last' property
            object.insert("last".to_string(), rhs);
            Ok(())
        }
        (crate::object::indexing::ObjectIndexKind::Brace, Value::Cell(_)) => {
            object.insert("lastCell".to_string(), rhs);
            Ok(())
        }
        (
            crate::object::indexing::ObjectIndexKind::Member,
            Value::String(runmat_value::StringElement::Text(field)),
        ) => {
            object.insert(field.as_str().to_owned(), rhs);
            Ok(())
        }
        (crate::object::indexing::ObjectIndexKind::Member, Value::CharArray(ca)) => {
            let field = super::text::decode_utf16_with_error(ca.data(), || {
                runtime_descriptor_error_with_detail(
                    "OverIdx.subsasgn",
                    &OVERIDX_ERROR_SUBSASGN_PAYLOAD_UNSUPPORTED,
                    "member name is not valid UTF-16 text",
                )
            })?;
            object.insert(field, rhs);
            Ok(())
        }
        _ => Err(runtime_descriptor_error(
            "OverIdx.subsasgn",
            &OVERIDX_ERROR_SUBSASGN_PAYLOAD_UNSUPPORTED,
        )),
    }?;
    RootedValue::new(Value::Object(object.finish())).map_err(|error| error.into_runtime_error())
}

pub(crate) async fn overidx_num_arguments_from_subscript(
    obj: Value,
    _subscript: Value,
    _indexing_context: Value,
) -> crate::BuiltinResult<Value> {
    let object = overidx_expect_object(obj, "OverIdx.numArgumentsFromSubscript")?;
    Ok(object.property("nargs").cloned().unwrap_or(Value::Num(3.0)))
}

pub(crate) async fn overidx_saveobj(obj: Value) -> crate::BuiltinResult<Value> {
    let object = overidx_expect_object(obj, "OverIdx.saveobj")?;
    let mut payload = runmat_value::UnboundedStructValueBuilder::new();
    for key in ["k", "nargs"] {
        if let Some(value) = object.property(key) {
            payload.insert(key.to_string(), value.clone());
        }
    }
    payload.insert(
        "saved_by".to_string(),
        Value::string("OverIdx.saveobj".to_string()),
    );
    Ok(Value::Struct(payload.finish()))
}

pub(crate) async fn overidx_loadobj(payload: Value) -> crate::BuiltinResult<Value> {
    let mut object = runmat_value::UnboundedObjectBuilder::new("OverIdx");
    if let Value::Struct(st) = payload {
        for (name, value) in st.into_fields_unbounded() {
            if name != "saved_by" {
                object.insert(name, value);
            }
        }
    }
    object.insert(
        "loaded_by".to_string(),
        Value::string("OverIdx.loadobj".to_string()),
    );
    Ok(Value::Object(object.finish()))
}

fn overidx_expect_object(
    obj: Value,
    method: &str,
) -> crate::BuiltinResult<runmat_value::ObjectInstance> {
    match obj {
        Value::Object(o) => Ok(o),
        other => Err(runtime_descriptor_error_with_detail(
            "OverIdx",
            &OVERIDX_ERROR_RECEIVER_INVALID,
            format!("{method}: got {other:?}"),
        )),
    }
}

fn overidx_binary_operands(
    lhs: Value,
    rhs: Value,
    method: &str,
) -> crate::BuiltinResult<(f64, f64)> {
    fn object_value(object: runmat_value::ObjectInstance) -> f64 {
        match object.property("k") {
            Some(Value::Num(value)) => *value,
            _ => 0.0,
        }
    }

    match (lhs, rhs) {
        (Value::Object(object), rhs) => {
            let rhs: f64 = (&rhs).try_into()?;
            Ok((object_value(object), rhs))
        }
        (lhs, Value::Object(object)) => {
            let lhs: f64 = (&lhs).try_into()?;
            Ok((lhs, object_value(object)))
        }
        (lhs, _) => Err(runtime_descriptor_error_with_detail(
            "OverIdx",
            &OVERIDX_ERROR_RECEIVER_INVALID,
            format!("{method}: got {lhs:?}"),
        )),
    }
}

// --- Operator overloading methods for OverIdx (test scaffolding) ---

pub(crate) async fn overidx_plus(obj: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    let (lhs, rhs) = overidx_binary_operands(obj, rhs, "OverIdx.plus")?;
    Ok(Value::Num(lhs + rhs))
}

pub(crate) async fn overidx_times(obj: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    let (lhs, rhs) = overidx_binary_operands(obj, rhs, "OverIdx.times")?;
    Ok(Value::Num(lhs * rhs))
}

pub(crate) async fn overidx_mtimes(obj: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    let (lhs, rhs) = overidx_binary_operands(obj, rhs, "OverIdx.mtimes")?;
    Ok(Value::Num(lhs * rhs))
}

pub(crate) async fn overidx_lt(obj: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    let (lhs, rhs) = overidx_binary_operands(obj, rhs, "OverIdx.lt")?;
    Ok(Value::Num(if lhs < rhs { 1.0 } else { 0.0 }))
}

pub(crate) async fn overidx_gt(obj: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    let (lhs, rhs) = overidx_binary_operands(obj, rhs, "OverIdx.gt")?;
    Ok(Value::Num(if lhs > rhs { 1.0 } else { 0.0 }))
}

pub(crate) async fn overidx_eq(obj: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    let (lhs, rhs) = overidx_binary_operands(obj, rhs, "OverIdx.eq")?;
    Ok(Value::Num(if (lhs - rhs).abs() < 1e-12 {
        1.0
    } else {
        0.0
    }))
}

pub(crate) async fn overidx_uplus(obj: Value) -> crate::BuiltinResult<Value> {
    // Identity
    Ok(obj)
}

pub(crate) async fn overidx_rdivide(obj: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    let (lhs, rhs) = overidx_binary_operands(obj, rhs, "OverIdx.rdivide")?;
    Ok(Value::Num(lhs / rhs))
}

pub(crate) async fn overidx_mrdivide(obj: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    overidx_rdivide(obj, rhs).await
}

pub(crate) async fn overidx_ldivide(obj: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    let (lhs, rhs) = overidx_binary_operands(obj, rhs, "OverIdx.ldivide")?;
    Ok(Value::Num(rhs / lhs))
}

pub(crate) async fn overidx_mldivide(obj: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    overidx_ldivide(obj, rhs).await
}

pub(crate) async fn overidx_and(obj: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    let (lhs, rhs) = overidx_binary_operands(obj, rhs, "OverIdx.and")?;
    Ok(Value::Num(if (lhs != 0.0) && (rhs != 0.0) {
        1.0
    } else {
        0.0
    }))
}

pub(crate) async fn overidx_or(obj: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    let (lhs, rhs) = overidx_binary_operands(obj, rhs, "OverIdx.or")?;
    Ok(Value::Num(if (lhs != 0.0) || (rhs != 0.0) {
        1.0
    } else {
        0.0
    }))
}

pub(crate) async fn overidx_xor(obj: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    let (lhs, rhs) = overidx_binary_operands(obj, rhs, "OverIdx.xor")?;
    let a = lhs != 0.0;
    let b = rhs != 0.0;
    Ok(Value::Num(if a ^ b { 1.0 } else { 0.0 }))
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod rooted_fixture_tests {
    use super::*;
    use std::cell::RefCell;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    thread_local! {
        static OBSERVED_HANDLE: RefCell<Option<runmat_gc::GcHandle>> = const { RefCell::new(None) };
    }

    #[test]
    fn managed_property_survives_producer_root_retirement_until_fixture_result_drops() {
        runmat_gc::gc_test_context(|| {
            let initial = runmat_gc::gc_allocate_rooted(Value::String("fixture payload".into()))
                .expect("allocate managed test property");
            let handle = initial.handle();
            let mut object = runmat_value::UnboundedObjectBuilder::new("OverIdx");
            object.insert(
                "last".into(),
                Value::HandleObject(runmat_value::HandleRef {
                    class_name: "FixturePayload".into(),
                    target: handle,
                    valid: true,
                }),
            );
            let subscript = crate::object::indexing::standard_substruct_fixture_from_parts(
                ".",
                Value::String("last".into()),
            )
            .expect("member path");
            let output = futures::executor::block_on(overidx_subsref(
                Value::Object(object.finish()),
                subscript,
            ))
            .expect("fixture member read");
            OBSERVED_HANDLE.with(|slot| *slot.borrow_mut() = Some(handle));
            let seen = Arc::new(AtomicBool::new(false));
            let observed = Arc::clone(&seen);
            let _observer = runmat_gc::RootRetirementObserver::register(Arc::new(move |event| {
                OBSERVED_HANDLE.with(|slot| {
                    let Some(handle) = *slot.borrow() else {
                        return;
                    };
                    if event != runmat_gc::RootRetirement::Explicit(handle) {
                        return;
                    }
                    runmat_gc::gc_collect_major().expect("collect at producer retirement");
                    assert!(runmat_gc::gc_clone_value(&handle).is_ok());
                    observed.store(true, Ordering::SeqCst);
                });
            }))
            .expect("observe producer retirement");
            initial.unroot().expect("retire producer root");
            assert!(seen.load(Ordering::SeqCst));
            assert_eq!(
                runmat_gc::gc_clone_value(&handle).expect("fixture result retains property"),
                Value::String("fixture payload".into())
            );
            drop(_observer);
            OBSERVED_HANDLE.with(|slot| *slot.borrow_mut() = None);
            drop(output);
            runmat_gc::gc_collect_major().expect("collect after result drop");
            assert!(runmat_gc::gc_clone_value(&handle).is_err());
        });
    }
}
