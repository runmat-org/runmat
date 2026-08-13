use super::*;
use runmat_builtins::{
    BuiltinIntegerAuditDescriptor, BuiltinIntegerAuditKind, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
};
use runmat_macros::runtime_builtin;

const HEIGHT_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "A",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Only array shape metadata is observed; numeric elements are not materialized.",
}];
pub const HEIGHT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor { form: "n = height(integer_A)", inputs: &HEIGHT_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::HostAndGpu, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "All integer classes share the array row-count contract; resident inputs are answered from handle shape without gather." }];
pub const ISCATEGORICAL_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "iscategorical is a universal object-type predicate; integer host or resident values return scalar false without gathering or converting numeric payload data.",
    };
pub const ISTABLE_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "istable is a universal object-type predicate; integer host or resident values return scalar false without gathering payload data.",
};
pub const ISTIMETABLE_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "istimetable is a universal object-type predicate; integer host or resident values return scalar false without gathering payload data.",
    };
pub const ISORDINAL_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "isordinal accepts any data type and returns true only for ordinal categorical arrays; integer host or resident values return scalar false without payload access.",
};

#[runtime_builtin(
    name = "height",
    category = "table",
    summary = "Return the number of rows in a table.",
    keywords = "height,table,rows",
    descriptor(crate::builtins::table::HEIGHT_DESCRIPTOR),
    integer_capabilities(
        crate::builtins::table::builtins::predicates::HEIGHT_INTEGER_CAPABILITIES
    ),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn height_builtin(value: Value) -> BuiltinResult<Value> {
    if let Value::GpuTensor(handle) = &value {
        let rows = handle.shape.first().copied().unwrap_or(1);
        return Ok(Value::Num(rows as f64));
    }
    let host = gather_if_needed_async(&value)
        .await
        .map_err(map_control_flow)?;
    if let Some(object) = table_object(&host) {
        return Ok(Value::Num(table_height(object)? as f64));
    }
    value_row_count(&host).map(|n| Value::Num(n as f64))
}

#[runtime_builtin(
    name = "width",
    category = "table",
    summary = "Return the number of variables in a table.",
    keywords = "width,table,variables",
    descriptor(crate::builtins::table::WIDTH_DESCRIPTOR),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn width_builtin(value: Value) -> BuiltinResult<Value> {
    let host = gather_if_needed_async(&value)
        .await
        .map_err(map_control_flow)?;
    if let Some(object) = table_object(&host) {
        return Ok(Value::Num(table_width(object)? as f64));
    }
    match host {
        Value::Tensor(t) => Ok(Value::Num(t.cols() as f64)),
        Value::ComplexTensor(t) => Ok(Value::Num(t.cols as f64)),
        Value::StringArray(sa) => Ok(Value::Num(sa.cols() as f64)),
        Value::LogicalArray(la) => Ok(Value::Num(la.shape.get(1).copied().unwrap_or(1) as f64)),
        Value::Cell(ca) => Ok(Value::Num(ca.cols as f64)),
        Value::CharArray(ca) => Ok(Value::Num(ca.cols as f64)),
        _ => Ok(Value::Num(1.0)),
    }
}

#[runtime_builtin(
    name = "istable",
    category = "table",
    summary = "Return true for table arrays.",
    keywords = "istable,table,predicate",
    descriptor(crate::builtins::table::TABLE_PREDICATE_DESCRIPTOR),
    integer_audit(crate::builtins::table::builtins::predicates::ISTABLE_INTEGER_AUDIT),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn istable_builtin(value: Value) -> BuiltinResult<Value> {
    Ok(Value::Bool(matches!(
        value,
        Value::Object(ref object) if object.is_class(TABLE_CLASS)
    )))
}

#[runtime_builtin(
    name = "istimetable",
    category = "table",
    summary = "Return true for timetable arrays.",
    keywords = "istimetable,timetable,predicate",
    descriptor(crate::builtins::table::TABLE_PREDICATE_DESCRIPTOR),
    integer_audit(crate::builtins::table::builtins::predicates::ISTIMETABLE_INTEGER_AUDIT),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn istimetable_builtin(value: Value) -> BuiltinResult<Value> {
    Ok(Value::Bool(matches!(
        value,
        Value::Object(ref object) if object.is_class(TIMETABLE_CLASS)
    )))
}

#[runtime_builtin(
    name = "iscategorical",
    category = "table",
    summary = "Return true for categorical arrays.",
    keywords = "iscategorical,categorical,predicate",
    descriptor(crate::builtins::table::TABLE_PREDICATE_DESCRIPTOR),
    integer_audit(crate::builtins::table::builtins::predicates::ISCATEGORICAL_INTEGER_AUDIT),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn iscategorical_builtin(value: Value) -> BuiltinResult<Value> {
    Ok(Value::Bool(matches!(
        value,
        Value::Object(ref object) if object.is_class(CATEGORICAL_CLASS)
    )))
}

#[runtime_builtin(
    name = "isordinal",
    category = "table",
    summary = "Return true for ordinal categorical arrays.",
    keywords = "isordinal,ordinal,categorical,predicate",
    descriptor(crate::builtins::table::TABLE_PREDICATE_DESCRIPTOR),
    integer_audit(crate::builtins::table::builtins::predicates::ISORDINAL_INTEGER_AUDIT),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn isordinal_builtin(value: Value) -> BuiltinResult<Value> {
    Ok(Value::Bool(matches!(
        value,
        Value::Object(ref object)
            if object.is_class(CATEGORICAL_CLASS)
                && matches!(object.properties.get("Ordinal"), Some(Value::Bool(true)))
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{IntValue, IntegerStorage, Tensor};

    #[test]
    fn universal_table_predicates_return_false_for_all_integer_classes() {
        for integer in [
            IntValue::I8(-1),
            IntValue::I16(-2),
            IntValue::I32(-3),
            IntValue::I64(i64::MIN),
            IntValue::U8(1),
            IntValue::U16(2),
            IntValue::U32(3),
            IntValue::U64(u64::MAX),
        ] {
            let value = Value::Int(integer);
            assert_eq!(
                block_on(istable_builtin(value.clone())).unwrap(),
                Value::Bool(false)
            );
            assert_eq!(
                block_on(istimetable_builtin(value.clone())).unwrap(),
                Value::Bool(false)
            );
            assert_eq!(
                block_on(isordinal_builtin(value)).unwrap(),
                Value::Bool(false)
            );
        }
    }

    #[test]
    fn universal_table_predicates_do_not_gather_resident_integer() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1])
                .expect("integer tensor");
            let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor)
                .expect("upload integer");
            let value = Value::GpuTensor(handle);
            assert_eq!(
                block_on(istable_builtin(value.clone())).unwrap(),
                Value::Bool(false)
            );
            assert_eq!(
                block_on(istimetable_builtin(value.clone())).unwrap(),
                Value::Bool(false)
            );
            assert_eq!(
                block_on(isordinal_builtin(value)).unwrap(),
                Value::Bool(false)
            );
        });
    }
}
