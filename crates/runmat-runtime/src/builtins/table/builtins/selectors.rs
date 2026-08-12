use super::*;
use runmat_builtins::{
    BuiltinExtensionDescriptor, BuiltinExtensionMode, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
};
use runmat_macros::runtime_builtin;

const HEAD_GPU_COUNT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "head-gpu-row-count",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "head with a GPU-resident row count is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:HeadGpuRowCountExtension"),
};
pub const HEAD_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [HEAD_GPU_COUNT_EXTENSION];

const HEAD_DATA_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "A",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Integer array rows are selected exactly and retain their input class.",
}];
const HEAD_COUNT_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "N",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "The positive row count is decoded exactly from authoritative scalar storage.",
}];
pub const HEAD_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor { form: "B = head(integer_A, ...)", inputs: &HEAD_DATA_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::PreserveInput, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::HostAndGpu, overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving, notes: "Row selection preserves exact integer storage and every trailing dimension; resident data is downloaded non-destructively and restored to its owner." },
    BuiltinIntegerCapabilityDescriptor { form: "B = head(A, integer_N)", inputs: &HEAD_COUNT_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "N is a positive scalar structural control; a resident N is a separately gated RunMat extension." },
];

#[runtime_builtin(
    name = "head",
    category = "table",
    summary = "Return the first rows of a table, timetable, or array.",
    keywords = "head,table,timetable,preview,rows",
    accel = "cpu",
    descriptor(crate::builtins::table::HEAD_DESCRIPTOR),
    extensions(crate::builtins::table::builtins::selectors::HEAD_EXTENSIONS),
    integer_capabilities(crate::builtins::table::builtins::selectors::HEAD_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn head_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(invalid_argument(
            "head: expected at most one row-count argument",
        ));
    }
    let n = match rest.first() {
        Some(Value::GpuTensor(handle)) => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &HEAD_GPU_COUNT_EXTENSION,
                "head",
            )?;
            let provider = runmat_accelerate_api::provider_for_handle(handle)
                .ok_or_else(|| invalid_argument("head: GPU row count has no owning provider"))?;
            let host =
                crate::builtins::common::gpu_helpers::download_value_preserving_residency_async(
                    provider, handle,
                )
                .await
                .map_err(map_control_flow)?;
            positive_usize(&host, "head row count")?
        }
        Some(value) => positive_usize(value, "head row count")?,
        None => 8,
    };
    if let Value::GpuTensor(handle) = value {
        let provider = runmat_accelerate_api::provider_for_handle(&handle)
            .ok_or_else(|| invalid_argument("head: GPU input has no owning provider"))?;
        let host = crate::builtins::common::gpu_helpers::download_value_preserving_residency_async(
            provider, &handle,
        )
        .await
        .map_err(map_control_flow)?;
        let rows = value_row_count(&host)?;
        let selected = (0..rows.min(n)).collect::<Vec<_>>();
        let selected_value = select_rows(&host, &selected)?;
        return crate::builtins::common::gpu_helpers::restore_class_preserving_value(
            &handle,
            selected_value,
            "head",
        );
    }
    let value = gather_if_needed_async(&value)
        .await
        .map_err(map_control_flow)?;
    let rows = value_row_count(&value)?;
    let selected = (0..rows.min(n)).collect::<Vec<_>>();
    if let Some(object) = table_object(&value) {
        let names = table_variable_names_from_object(object)?;
        let variables = table_variables(object)?;
        let mut columns = Vec::with_capacity(names.len());
        for name in &names {
            columns.push(select_rows(
                variables
                    .fields
                    .get(name)
                    .ok_or_else(|| invalid_variable(format!("head: missing variable '{name}'")))?,
                &selected,
            )?);
        }
        return subset_tabular_object(object, names, columns, &selected);
    }
    select_rows(&value, &selected)
}
