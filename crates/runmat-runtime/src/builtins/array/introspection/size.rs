//! MATLAB-compatible `size` execution binding.

use super::dimension_metadata::VisibleDimensions;
use super::shape_query::{parse_dimension_arguments, DimensionSelectorError, EmptySelectorPolicy};
use super::structural_output::{exact_double, row_vector, StructuralOutputError};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_builtins::{BuiltinErrorDescriptor, SIZE_CATALOG_ENTRY};
use runmat_value::Value;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::introspection::size")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "size",
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
    notes: "Reads validated host, resident, or distributed shape metadata without provider dispatch or payload transfer; returns host double values.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::introspection::size")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "size",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Host metadata query that forms a fusion boundary.",
};

#[runmat_macros::runtime_builtin(
    name = "size",
    binding_variant = "default",
    builtin_path = "crate::builtins::array::introspection::size"
)]
async fn size_builtin(value: Value, selectors: Vec<Value>) -> BuiltinResult<Value> {
    let dimensions = VisibleDimensions::from_value(&value)
        .await
        .map_err(|detail| descriptor_error(&runmat_builtins::SIZE_ERROR_INTERNAL, detail))?;
    if selectors.is_empty() {
        return unselected_size(&dimensions);
    }
    let selectors = parse_dimension_arguments(&selectors, EmptySelectorPolicy::Allow)
        .map_err(selector_error)?;
    let values = selectors
        .values()
        .iter()
        .map(|dimension| dimensions.extent(*dimension))
        .collect::<Vec<_>>();
    selected_size(&values)
}

fn unselected_size(dimensions: &VisibleDimensions) -> BuiltinResult<Value> {
    match crate::output_count::current_output_count() {
        Some(0) => Ok(Value::OutputList(Vec::new())),
        Some(count) if count > 1 => {
            let values = dimensions.collapsed_outputs(count).ok_or_else(|| {
                descriptor_error(
                    &runmat_builtins::SIZE_ERROR_RESULT_NOT_EXACT,
                    "collapsed dimension product exceeds the structural range",
                )
            })?;
            scalar_outputs(&values)
        }
        _ => structural_row(dimensions.reported_size()),
    }
}

fn selected_size(values: &[u64]) -> BuiltinResult<Value> {
    match crate::output_count::current_output_count() {
        Some(0) => Ok(Value::OutputList(Vec::new())),
        Some(count) if count > 1 => {
            if count != values.len() {
                return Err(descriptor_error(
                    &runmat_builtins::SIZE_ERROR_OUTPUT_COUNT,
                    format!(
                        "requested {count} outputs for {} queried dimensions",
                        values.len()
                    ),
                ));
            }
            scalar_outputs(values)
        }
        _ if values.len() == 1 => scalar_value(values[0]),
        _ => structural_row(values),
    }
}

fn scalar_outputs(values: &[u64]) -> BuiltinResult<Value> {
    values
        .iter()
        .copied()
        .map(scalar_value)
        .collect::<BuiltinResult<Vec<_>>>()
        .map(Value::OutputList)
}

fn scalar_value(value: u64) -> BuiltinResult<Value> {
    exact_double(value).map(Value::Num).ok_or_else(|| {
        descriptor_error(
            &runmat_builtins::SIZE_ERROR_RESULT_NOT_EXACT,
            format!("dimension extent {value} cannot be represented exactly as double"),
        )
    })
}

fn structural_row(values: &[u64]) -> BuiltinResult<Value> {
    row_vector(values).map_err(|error| match error {
        StructuralOutputError::NotExactlyRepresentable(value) => descriptor_error(
            &runmat_builtins::SIZE_ERROR_RESULT_NOT_EXACT,
            format!("dimension extent {value} cannot be represented exactly as double"),
        ),
        StructuralOutputError::Tensor(detail) => {
            descriptor_error(&runmat_builtins::SIZE_ERROR_INTERNAL, detail)
        }
    })
}

fn selector_error(error: DimensionSelectorError) -> RuntimeError {
    let descriptor = match error {
        DimensionSelectorError::ArgumentType | DimensionSelectorError::Empty => {
            &runmat_builtins::SIZE_ERROR_DIM_ARG_TYPE
        }
        DimensionSelectorError::VectorShape => &runmat_builtins::SIZE_ERROR_DIM_VECTOR_SHAPE,
        DimensionSelectorError::NonFinite => &runmat_builtins::SIZE_ERROR_DIM_NON_FINITE,
        DimensionSelectorError::NonInteger => &runmat_builtins::SIZE_ERROR_DIM_NON_INTEGER,
        DimensionSelectorError::LessThanOne => &runmat_builtins::SIZE_ERROR_DIM_LT_ONE,
        DimensionSelectorError::OutOfRange => &runmat_builtins::SIZE_ERROR_DIM_RANGE,
        DimensionSelectorError::VectorInScalarList => &runmat_builtins::SIZE_ERROR_DIM_SCALAR_LIST,
    };
    descriptor_error(descriptor, descriptor.when)
}

fn descriptor_error(
    descriptor: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {detail}", descriptor.message))
        .with_builtin(SIZE_CATALOG_ENTRY.identity.name);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_value::{CellArray, IntegerStorage, Tensor};

    fn call(value: Value, selectors: Vec<Value>) -> BuiltinResult<Value> {
        block_on(size_builtin(value, selectors))
    }

    fn assert_row(value: Value, expected: &[f64]) {
        let Value::Tensor(value) = value else {
            panic!("expected tensor")
        };
        assert_eq!(value.shape, vec![1, expected.len()]);
        assert_eq!(value.materialize_f64(), expected);
    }

    #[test]
    fn full_scalar_vector_and_variadic_queries_are_distinct() {
        let tensor = Tensor::new(vec![0.0; 24], vec![2, 3, 4]).unwrap();
        assert_row(
            call(Value::Tensor(tensor.clone()), vec![]).unwrap(),
            &[2.0, 3.0, 4.0],
        );
        assert_eq!(
            call(Value::Tensor(tensor.clone()), vec![Value::Num(2.0)]).unwrap(),
            Value::Num(3.0)
        );
        let dims = Tensor::new(vec![1.0, 3.0], vec![1, 2]).unwrap();
        assert_row(
            call(Value::Tensor(tensor.clone()), vec![Value::Tensor(dims)]).unwrap(),
            &[2.0, 4.0],
        );
        assert_row(
            call(
                Value::Tensor(tensor),
                vec![Value::Num(1.0), Value::Num(3.0)],
            )
            .unwrap(),
            &[2.0, 4.0],
        );
    }

    #[test]
    fn multiple_outputs_collapse_remaining_dimensions() {
        let _guard = crate::output_count::push_output_count(Some(2));
        let tensor = Tensor::new(vec![0.0; 60], vec![3, 4, 5]).unwrap();
        assert_eq!(
            call(Value::Tensor(tensor), vec![]).unwrap(),
            Value::OutputList(vec![Value::Num(3.0), Value::Num(20.0)])
        );
    }

    #[test]
    fn one_output_omits_trailing_singleton_dimensions() {
        let tensor = Tensor::new(vec![0.0; 6], vec![2, 3, 1, 1]).unwrap();
        assert_row(call(Value::Tensor(tensor), vec![]).unwrap(), &[2.0, 3.0]);
    }

    #[test]
    fn selected_multiple_outputs_must_match_query_count() {
        let _guard = crate::output_count::push_output_count(Some(2));
        let tensor = Tensor::new(vec![0.0; 24], vec![2, 3, 4]).unwrap();
        let dims = Tensor::new(vec![1.0, 3.0], vec![1, 2]).unwrap();
        assert_eq!(
            call(Value::Tensor(tensor.clone()), vec![Value::Tensor(dims)]).unwrap(),
            Value::OutputList(vec![Value::Num(2.0), Value::Num(4.0)])
        );
        assert!(call(Value::Tensor(tensor), vec![Value::Num(1.0)]).is_err());
    }

    #[test]
    fn empty_and_wide_integer_selectors_preserve_structural_semantics() {
        let tensor = Tensor::new(vec![0.0; 8], vec![2, 4]).unwrap();
        let empty = Tensor::new(vec![], vec![1, 0]).unwrap();
        assert_row(
            call(Value::Tensor(tensor.clone()), vec![Value::Tensor(empty)]).unwrap(),
            &[],
        );
        let wide =
            Tensor::new_integer(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1])
                .unwrap();
        assert_eq!(
            call(Value::Tensor(tensor), vec![Value::Tensor(wide)]).unwrap(),
            Value::Num(1.0)
        );
    }

    #[test]
    fn cell_table_and_resident_values_use_outer_metadata() {
        let cell = CellArray::new(vec![Value::Num(0.0); 6], 2, 3).unwrap();
        assert_row(call(Value::Cell(cell), vec![]).unwrap(), &[2.0, 3.0]);

        crate::builtins::table::ensure_table_class_registered();
        let table = crate::builtins::table::table_from_columns(
            vec!["A".into(), "B".into()],
            vec![
                Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap()),
                Value::Tensor(Tensor::new(vec![4.0, 5.0, 6.0], vec![3, 1]).unwrap()),
            ],
        )
        .unwrap();
        assert_row(call(table, vec![]).unwrap(), &[3.0, 2.0]);

        let handle = runmat_accelerate_api::GpuTensorHandle {
            shape: vec![7, 9],
            device_id: u32::MAX,
            buffer_id: u64::MAX,
            descriptor: Default::default(),
        };
        assert_row(call(Value::GpuTensor(handle), vec![]).unwrap(), &[7.0, 9.0]);
    }
}
