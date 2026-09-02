use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_builtins::{BuiltinCatalogEntry, BuiltinErrorDescriptor, ShapeScalarQuery};
use runmat_value::Value;

use super::dimension_metadata::{effective_rank, visible_dimensions};

pub(super) struct ShapeScalarQueryBoundary {
    entry: &'static BuiltinCatalogEntry,
    internal: &'static BuiltinErrorDescriptor,
    too_many_outputs: &'static BuiltinErrorDescriptor,
    result_not_exact: Option<&'static BuiltinErrorDescriptor>,
    unsupported_table: Option<&'static BuiltinErrorDescriptor>,
    query: ShapeScalarQuery,
}

impl ShapeScalarQueryBoundary {
    pub(super) const fn new(
        entry: &'static BuiltinCatalogEntry,
        internal: &'static BuiltinErrorDescriptor,
        too_many_outputs: &'static BuiltinErrorDescriptor,
        result_not_exact: Option<&'static BuiltinErrorDescriptor>,
        unsupported_table: Option<&'static BuiltinErrorDescriptor>,
        query: ShapeScalarQuery,
    ) -> Self {
        Self {
            entry,
            internal,
            too_many_outputs,
            result_not_exact,
            unsupported_table,
            query,
        }
    }

    pub(super) async fn execute(&self, value: Value) -> BuiltinResult<Value> {
        self.reject_excess_outputs()?;
        if self.query == ShapeScalarQuery::Length {
            if matches!(&value, Value::Object(object) if crate::builtins::table::is_tabular_object(object))
            {
                let descriptor = self
                    .unsupported_table
                    .expect("length declares its tabular-value diagnostic");
                return Err(self.error(descriptor, descriptor.when));
            }
            if let Some(count) = crate::builtins::containers::map::map_length(&value) {
                return self.result(u64::try_from(count).map_err(|error| self.internal(error))?);
            }
        }

        let dimensions = visible_dimensions(&value)
            .await
            .map_err(|error| self.internal(error))?;
        let result = match self.query {
            ShapeScalarQuery::Length => dimensions.into_iter().max().unwrap_or(0),
            ShapeScalarQuery::Rank => {
                u64::try_from(effective_rank(&dimensions)).map_err(|error| self.internal(error))?
            }
        };
        self.result(result)
    }

    fn result(&self, value: u64) -> BuiltinResult<Value> {
        if !is_exact_binary64_integer(value) {
            let descriptor = self.result_not_exact.unwrap_or(self.internal);
            return Err(self.error(
                descriptor,
                format!("structural result {value} cannot be represented exactly as double"),
            ));
        }
        Ok(Value::Num(value as f64))
    }

    fn reject_excess_outputs(&self) -> BuiltinResult<()> {
        if matches!(crate::output_count::current_output_count(), Some(count) if count > 1) {
            return Err(self.error(self.too_many_outputs, "only one output is defined"));
        }
        Ok(())
    }

    fn internal(&self, detail: impl std::fmt::Display) -> RuntimeError {
        self.error(self.internal, detail)
    }

    fn error(
        &self,
        descriptor: &'static BuiltinErrorDescriptor,
        detail: impl std::fmt::Display,
    ) -> RuntimeError {
        let mut builder = build_runtime_error(format!("{}: {detail}", descriptor.message))
            .with_builtin(self.entry.identity.name);
        if let Some(identifier) = descriptor.identifier {
            builder = builder.with_identifier(identifier);
        }
        builder.build()
    }
}

fn is_exact_binary64_integer(value: u64) -> bool {
    if value == 0 {
        return true;
    }
    let significant_bits = u64::BITS - value.leading_zeros();
    let discarded_bits = significant_bits.saturating_sub(f64::MANTISSA_DIGITS);
    value.trailing_zeros() >= discarded_bits
}

macro_rules! define_shape_scalar_query_runtime {
    (
        builtin_fn: $builtin_fn:ident,
        name: $name:literal,
        query: $query:expr,
        entry: $entry:path,
        internal_error: $internal_error:path,
        output_error: $output_error:path,
        exact_error: $exact_error:expr,
        table_error: $table_error:expr,
        builtin_path: $builtin_path:literal
    ) => {
        use crate::builtins::common::spec::{
            BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
            ReductionNaN, ResidencyPolicy, ShapeRequirements,
        };
        use runmat_value::Value;

        #[runmat_macros::register_gpu_spec(builtin_path = $builtin_path)]
        pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
            name: $name,
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
            notes: "Reads validated host, resident, or distributed shape metadata without provider dispatch or payload transfer; returns one host double scalar.",
        };
        #[runmat_macros::register_fusion_spec(builtin_path = $builtin_path)]
        pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
            name: $name,
            shape: ShapeRequirements::Any,
            constant_strategy: ConstantStrategy::InlineLiteral,
            elementwise: None,
            reduction: None,
            emits_nan: false,
            notes: "Host scalar metadata query that forms a fusion boundary.",
        };
        const BOUNDARY: super::shape_scalar_query::ShapeScalarQueryBoundary =
            super::shape_scalar_query::ShapeScalarQueryBoundary::new(
                &$entry,
                &$internal_error,
                &$output_error,
                $exact_error,
                $table_error,
                $query,
            );
        #[runmat_macros::runtime_builtin(
            name = $name,
            binding_variant = "default",
            builtin_path = $builtin_path
        )]
        async fn $builtin_fn(value: Value) -> crate::BuiltinResult<Value> {
            BOUNDARY.execute(value).await
        }
    };
}

pub(super) use define_shape_scalar_query_runtime;

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_execution::{
        DistributedObjectId, DistributedValueHandle, ExecutionScopeId, PoolHandle, PoolId,
    };
    use runmat_types::{
        DistributedOwner, DistributedValueId, DistributionScheme, LabCount, NumericClass,
        NumericDomain, NumericFact, ParallelRegionId, ProgramFunctionId, RegionId, ValueFact,
        ValueKindFact,
    };

    fn distributed_value(global_shape: Vec<u64>) -> Value {
        let scope_id = ExecutionScopeId::derive(&[b"shape-scalar-query"]);
        let function = ProgramFunctionId(9);
        let owner = ParallelRegionId(RegionId {
            function,
            ordinal: 1,
        });
        Value::Distributed(Box::new(DistributedValueHandle {
            id: DistributedObjectId::derive(&[b"shape-scalar-value"]),
            contract: DistributedValueId {
                function,
                ordinal: 2,
            },
            owner: DistributedOwner::Region(owner),
            scope_id,
            generation: 1,
            pool: PoolHandle {
                id: PoolId::derive(&[b"shape-scalar-pool"]),
                scope_id,
                generation: 1,
            },
            partition_count: LabCount(2),
            value: ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            })),
            global_shape,
            scheme: DistributionScheme::Replicated,
            materializable: true,
        }))
    }

    fn boundary(query: ShapeScalarQuery) -> ShapeScalarQueryBoundary {
        match query {
            ShapeScalarQuery::Length => ShapeScalarQueryBoundary::new(
                &runmat_builtins::LENGTH_CATALOG_ENTRY,
                &runmat_builtins::LENGTH_ERROR_INTERNAL,
                &runmat_builtins::LENGTH_ERROR_TOO_MANY_OUTPUTS,
                Some(&runmat_builtins::LENGTH_ERROR_RESULT_NOT_EXACT),
                Some(&runmat_builtins::LENGTH_ERROR_UNSUPPORTED_TABLE),
                query,
            ),
            ShapeScalarQuery::Rank => ShapeScalarQueryBoundary::new(
                &runmat_builtins::NDIMS_CATALOG_ENTRY,
                &runmat_builtins::NDIMS_ERROR_INTERNAL,
                &runmat_builtins::NDIMS_ERROR_TOO_MANY_OUTPUTS,
                None,
                None,
                query,
            ),
        }
    }

    #[test]
    fn exact_binary64_integer_check_accepts_only_representable_values() {
        assert!(is_exact_binary64_integer(1_u64 << 53));
        assert!(!is_exact_binary64_integer((1_u64 << 53) + 1));
        assert!(is_exact_binary64_integer((1_u64 << 53) + 2));
        assert!(!is_exact_binary64_integer(u64::MAX));
    }

    #[test]
    fn distributed_queries_use_validated_global_shape() {
        assert_eq!(
            block_on(boundary(ShapeScalarQuery::Length).execute(distributed_value(vec![8, 13])))
                .unwrap(),
            Value::Num(13.0)
        );
        assert_eq!(
            block_on(
                boundary(ShapeScalarQuery::Rank).execute(distributed_value(vec![8, 13, 4, 1]))
            )
            .unwrap(),
            Value::Num(3.0)
        );
    }

    #[test]
    fn length_rejects_an_inexact_distributed_extent() {
        let error = block_on(
            boundary(ShapeScalarQuery::Length)
                .execute(distributed_value(vec![(1_u64 << 53) + 1, 2])),
        )
        .expect_err("inexact double");
        assert_eq!(
            error.identifier(),
            Some("RunMat:length:ResultNotExactDouble")
        );
    }
}
