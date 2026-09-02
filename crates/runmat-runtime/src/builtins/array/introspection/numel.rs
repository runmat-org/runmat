//! `numel` execution binding and compatibility-gated selector extension.

use super::shape_query::{
    exact_double, parse_dimension_arguments, DimensionSelectorError, EmptySelectorPolicy,
    VisibleDimensions,
};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_builtins::{BuiltinErrorDescriptor, NUMEL_CATALOG_ENTRY};
use runmat_value::Value;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::introspection::numel")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "numel",
    op_kind: GpuOpKind::Custom("metadata"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Computes the element count from validated host, resident, or distributed shape metadata without provider dispatch or payload transfer.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::array::introspection::numel"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "numel",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Host scalar metadata query that forms a fusion boundary.",
};

#[runmat_macros::runtime_builtin(
    name = "numel",
    binding_variant = "default",
    builtin_path = "crate::builtins::array::introspection::numel"
)]
async fn numel_builtin(value: Value, selectors: Vec<Value>) -> BuiltinResult<Value> {
    reject_excess_outputs()?;
    let dimensions = VisibleDimensions::from_value(&value)
        .await
        .map_err(|detail| descriptor_error(&runmat_builtins::NUMEL_ERROR_INTERNAL, detail))?;
    let count = if selectors.is_empty() {
        dimensions.product()
    } else {
        crate::compatibility::ensure_builtin_extension_enabled(
            &runmat_builtins::NUMEL_DIMENSIONS_EXTENSION,
            NUMEL_CATALOG_ENTRY.identity.name,
        )?;
        let selectors = parse_dimension_arguments(&selectors, EmptySelectorPolicy::Reject)
            .map_err(selector_error)?;
        dimensions.selected_product(selectors.values())
    }
    .ok_or_else(|| {
        descriptor_error(
            &runmat_builtins::NUMEL_ERROR_RESULT_NOT_EXACT,
            "dimension product exceeds the structural range",
        )
    })?;
    exact_double(count).map(Value::Num).ok_or_else(|| {
        descriptor_error(
            &runmat_builtins::NUMEL_ERROR_RESULT_NOT_EXACT,
            format!("element count {count} cannot be represented exactly as double"),
        )
    })
}

fn reject_excess_outputs() -> BuiltinResult<()> {
    if matches!(crate::output_count::current_output_count(), Some(count) if count > 1) {
        return Err(descriptor_error(
            &runmat_builtins::NUMEL_ERROR_TOO_MANY_OUTPUTS,
            "only one output is defined",
        ));
    }
    Ok(())
}

fn selector_error(error: DimensionSelectorError) -> RuntimeError {
    let descriptor = match error {
        DimensionSelectorError::ArgumentType => &runmat_builtins::NUMEL_ERROR_DIM_ARG_TYPE,
        DimensionSelectorError::VectorShape => &runmat_builtins::NUMEL_ERROR_DIM_VECTOR_SHAPE,
        DimensionSelectorError::Empty => &runmat_builtins::NUMEL_ERROR_DIM_EMPTY,
        DimensionSelectorError::NonFinite => &runmat_builtins::NUMEL_ERROR_DIM_NON_FINITE,
        DimensionSelectorError::NonInteger => &runmat_builtins::NUMEL_ERROR_DIM_NON_INTEGER,
        DimensionSelectorError::LessThanOne => &runmat_builtins::NUMEL_ERROR_DIM_LT_ONE,
        DimensionSelectorError::OutOfRange => &runmat_builtins::NUMEL_ERROR_DIM_RANGE,
        DimensionSelectorError::VectorInScalarList => &runmat_builtins::NUMEL_ERROR_DIM_SCALAR_LIST,
    };
    descriptor_error(descriptor, descriptor.when)
}

fn descriptor_error(
    descriptor: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {detail}", descriptor.message))
        .with_builtin(NUMEL_CATALOG_ENTRY.identity.name);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_value::{CellArray, CharArray, IntegerStorage, Tensor};

    fn call(value: Value, selectors: Vec<Value>) -> BuiltinResult<Value> {
        block_on(numel_builtin(value, selectors))
    }

    #[test]
    fn counts_arrays_cells_characters_tables_and_resident_shapes() {
        let tensor = Tensor::new(vec![0.0; 24], vec![2, 3, 4]).unwrap();
        assert_eq!(
            call(Value::Tensor(tensor), vec![]).unwrap(),
            Value::Num(24.0)
        );
        let cell = CellArray::new(vec![Value::Num(0.0); 6], 2, 3).unwrap();
        assert_eq!(call(Value::Cell(cell), vec![]).unwrap(), Value::Num(6.0));
        let chars = CharArray::new("RunMat".chars().collect(), 1, 6).unwrap();
        assert_eq!(
            call(Value::CharArray(chars), vec![]).unwrap(),
            Value::Num(6.0)
        );

        crate::builtins::table::ensure_table_class_registered();
        let table = crate::builtins::table::table_from_columns(
            vec!["A".into(), "B".into()],
            vec![
                Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap()),
                Value::Tensor(Tensor::new(vec![4.0, 5.0, 6.0], vec![3, 1]).unwrap()),
            ],
        )
        .unwrap();
        assert_eq!(call(table, vec![]).unwrap(), Value::Num(6.0));

        let handle = runmat_accelerate_api::GpuTensorHandle {
            shape: vec![8, 9],
            device_id: u32::MAX,
            buffer_id: u64::MAX,
            descriptor: Default::default(),
        };
        assert_eq!(
            call(Value::GpuTensor(handle), vec![]).unwrap(),
            Value::Num(72.0)
        );
    }

    #[test]
    fn selector_extension_is_gated_and_exact() {
        let tensor = Tensor::new(vec![0.0; 24], vec![2, 3, 4]).unwrap();
        let strict = crate::compatibility::push_runmat_extensions_enabled(false);
        assert!(call(Value::Tensor(tensor.clone()), vec![Value::Num(1.0)]).is_err());
        drop(strict);

        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        assert_eq!(
            call(
                Value::Tensor(tensor.clone()),
                vec![Value::Num(1.0), Value::Num(3.0)],
            )
            .unwrap(),
            Value::Num(8.0)
        );
        let selectors = Tensor::new_integer(
            IntegerStorage::U64(vec![1, 9_007_199_254_740_993]),
            vec![1, 2],
        )
        .unwrap();
        assert_eq!(
            call(Value::Tensor(tensor), vec![Value::Tensor(selectors)]).unwrap(),
            Value::Num(2.0)
        );
    }

    #[test]
    fn rejects_empty_selector_and_excess_outputs() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let empty = Tensor::new(vec![], vec![1, 0]).unwrap();
        assert!(call(Value::Num(1.0), vec![Value::Tensor(empty)]).is_err());
        let _outputs = crate::output_count::push_output_count(Some(2));
        assert!(call(Value::Num(1.0), vec![]).is_err());
    }
}
