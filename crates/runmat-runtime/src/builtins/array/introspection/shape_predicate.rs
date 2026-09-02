use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_builtins::{BuiltinCatalogEntry, BuiltinErrorDescriptor, ShapePredicate};
use runmat_value::Value;

pub(super) struct ShapePredicateBoundary {
    entry: &'static BuiltinCatalogEntry,
    internal: &'static BuiltinErrorDescriptor,
    too_many_outputs: &'static BuiltinErrorDescriptor,
    predicate: ShapePredicate,
}

impl ShapePredicateBoundary {
    pub(super) const fn new(
        entry: &'static BuiltinCatalogEntry,
        internal: &'static BuiltinErrorDescriptor,
        too_many_outputs: &'static BuiltinErrorDescriptor,
        predicate: ShapePredicate,
    ) -> Self {
        Self {
            entry,
            internal,
            too_many_outputs,
            predicate,
        }
    }

    pub(super) async fn execute(&self, value: Value) -> BuiltinResult<Value> {
        self.reject_excess_outputs()?;
        let dimensions = self.dimensions(&value).await?;
        Ok(Value::Bool(self.classify(&dimensions)))
    }

    async fn dimensions(&self, value: &Value) -> BuiltinResult<Vec<u64>> {
        let dimensions = match value {
            Value::Distributed(handle) => {
                handle.validate().map_err(|error| self.internal(error))?;
                normalize_dimensions(&handle.global_shape)
            }
            other => {
                let dimensions = crate::builtins::common::shape::value_dimensions(other)
                    .await
                    .map_err(|error| self.internal(error.message()))?;
                dimensions.into_iter().map(|value| value as u64).collect()
            }
        };
        Ok(dimensions)
    }

    fn classify(&self, dimensions: &[u64]) -> bool {
        let rank = effective_rank(dimensions);
        let rows = dimensions.first().copied().unwrap_or(1);
        let columns = dimensions.get(1).copied().unwrap_or(1);
        match self.predicate {
            ShapePredicate::Empty => dimensions.contains(&0),
            ShapePredicate::Scalar => dimensions.iter().all(|dimension| *dimension == 1),
            ShapePredicate::Vector => rank <= 2 && (rows == 1 || columns == 1),
            ShapePredicate::Matrix => rank <= 2,
            ShapePredicate::Row => rank <= 2 && rows == 1,
            ShapePredicate::Column => rank <= 2 && columns == 1,
        }
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

fn normalize_dimensions(dimensions: &[u64]) -> Vec<u64> {
    match dimensions {
        [] | [1] | [1, 1] => vec![1, 1],
        [dimension] => vec![1, *dimension],
        _ => dimensions.to_vec(),
    }
}

fn effective_rank(dimensions: &[u64]) -> usize {
    dimensions
        .iter()
        .rposition(|dimension| *dimension != 1)
        .map_or(2, |index| (index + 1).max(2))
}

macro_rules! define_shape_predicate_runtime {
    (
        builtin_fn: $builtin_fn:ident,
        name: $name:literal,
        predicate: $predicate:expr,
        entry: $entry:path,
        internal_error: $internal_error:path,
        output_error: $output_error:path,
        builtin_path: $builtin_path:literal
    ) => {
        use crate::builtins::common::spec::{
            BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
            ReductionNaN, ResidencyPolicy, ShapeRequirements,
        };
        use runmat_macros::runtime_builtin;
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
            notes: "Reads validated host, resident, or distributed shape metadata without provider dispatch or payload transfer; returns one host logical scalar.",
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
        const BOUNDARY: super::shape_predicate::ShapePredicateBoundary =
            super::shape_predicate::ShapePredicateBoundary::new(
                &$entry,
                &$internal_error,
                &$output_error,
                $predicate,
            );
        #[runtime_builtin(
            name = $name,
            binding_variant = "default",
            builtin_path = $builtin_path
        )]
        async fn $builtin_fn(value: Value) -> crate::BuiltinResult<Value> {
            BOUNDARY.execute(value).await
        }
    };
}

pub(super) use define_shape_predicate_runtime;

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
        let scope_id = ExecutionScopeId::derive(&[b"shape-predicate"]);
        let function = ProgramFunctionId(7);
        let owner = ParallelRegionId(RegionId {
            function,
            ordinal: 2,
        });
        Value::Distributed(Box::new(DistributedValueHandle {
            id: DistributedObjectId::derive(&[b"shape-predicate-value"]),
            contract: DistributedValueId {
                function,
                ordinal: 1,
            },
            owner: DistributedOwner::Region(owner),
            scope_id,
            generation: 1,
            pool: PoolHandle {
                id: PoolId::derive(&[b"shape-predicate-pool"]),
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

    fn classify_distributed(predicate: ShapePredicate, global_shape: Vec<u64>) -> bool {
        let boundary = ShapePredicateBoundary::new(
            &runmat_builtins::ISEMPTY_CATALOG_ENTRY,
            &runmat_builtins::ISEMPTY_ERROR_INTERNAL,
            &runmat_builtins::ISEMPTY_ERROR_TOO_MANY_OUTPUTS,
            predicate,
        );
        match block_on(boundary.execute(distributed_value(global_shape))).expect("valid handle") {
            Value::Bool(value) => value,
            other => panic!("expected logical scalar, got {other:?}"),
        }
    }

    #[test]
    fn normalizes_scalar_and_vector_shapes() {
        assert_eq!(normalize_dimensions(&[]), vec![1, 1]);
        assert_eq!(normalize_dimensions(&[1]), vec![1, 1]);
        assert_eq!(normalize_dimensions(&[3]), vec![1, 3]);
    }

    #[test]
    fn effective_rank_ignores_only_trailing_singletons() {
        assert_eq!(effective_rank(&[2, 3, 1, 1]), 2);
        assert_eq!(effective_rank(&[1, 1, 3, 1]), 3);
    }

    #[test]
    fn distributed_values_use_validated_global_shape_without_materialization() {
        assert!(classify_distributed(ShapePredicate::Empty, vec![0, 4]));
        assert!(classify_distributed(ShapePredicate::Scalar, vec![1, 1]));
        assert!(classify_distributed(ShapePredicate::Vector, vec![8, 1, 1]));
        assert!(classify_distributed(ShapePredicate::Matrix, vec![8, 3, 1]));
        assert!(classify_distributed(ShapePredicate::Row, vec![1, 8]));
        assert!(classify_distributed(ShapePredicate::Column, vec![8, 1]));
        assert!(!classify_distributed(ShapePredicate::Matrix, vec![8, 3, 2]));
    }

    #[test]
    fn invalid_distributed_metadata_uses_the_catalog_error() {
        let boundary = ShapePredicateBoundary::new(
            &runmat_builtins::ISEMPTY_CATALOG_ENTRY,
            &runmat_builtins::ISEMPTY_ERROR_INTERNAL,
            &runmat_builtins::ISEMPTY_ERROR_TOO_MANY_OUTPUTS,
            ShapePredicate::Empty,
        );
        let Value::Distributed(mut handle) = distributed_value(vec![2, 2]) else {
            unreachable!();
        };
        handle.generation = 0;
        let error = block_on(boundary.execute(Value::Distributed(handle))).expect_err("invalid");
        assert_eq!(error.identifier(), Some("RunMat:isempty:InternalError"));
    }
}
