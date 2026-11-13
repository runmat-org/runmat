//! Scoped, validated pipeline ownership for migrated fusion and tagged routes.
use super::backend_types::WgpuProvider;
use anyhow::{ensure, Result};
use std::sync::Arc;

/// A driver-validated BGL/pipeline pair carried into numerical emission;
/// legacy shared-cache writers cannot substitute an unvalidated pipeline.
pub(super) struct ValidatedFusionPipeline {
    pub(super) key: u64,
    kind: ValidatedPipelineKind,
    recipe: super::pipeline_recipe::PipelineRecipe,
    pub(super) layout: Arc<wgpu::BindGroupLayout>,
    pub(super) pipeline: super::pipeline_cache::PipelineOwner,
}

/// A successful scoped driver receipt is not yet a semantic runtime pipeline.
/// The disk label never supplies `ValidatedPipelineKind`; runtime admission does.
pub(super) struct WarmedPipelineReceipt {
    pub(super) recipe: super::pipeline_recipe::PipelineRecipe,
    pub(super) layout: Arc<wgpu::BindGroupLayout>,
    pub(super) pipeline: super::pipeline_cache::PipelineOwner,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ValidatedPipelineKind {
    Fusion {
        inputs: usize,
        layout: super::pipeline_recipe::FusionPipelineLayout,
    },
    Modulation,
    SoftmaxRows,
    TriangularSolve,
    IntegerSumCast,
    DimensionNSum,
    MatmulEpilogue,
    Indexing(IndexingPipelineKind),
    IntegerComparison,
    IntegerBinary(IntegerBinaryPipelineKind),
    ReductionExtrema(ReductionExtremaPipelineKind),
    StdPrimitive(StdPrimitivePipelineKind),
    IntegerCast {
        source: crate::backend::wgpu::types::NumericPrecision,
    },
    FloatingCast {
        source: crate::backend::wgpu::types::NumericPrecision,
        target: crate::backend::wgpu::types::NumericPrecision,
    },
    RoundDigits {
        significant: bool,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum IndexingPipelineKind {
    ScatterColumn,
    ScatterRow,
    Sub2Ind { inputs: usize },
    Ind2Sub { outputs: usize },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum IntegerBinaryPipelineKind {
    Arithmetic,
    MinMax,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ReductionExtremaPipelineKind {
    Global,
    Dimensional,
    IntegerDimensional,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum StdPrimitivePipelineKind {
    DimensionalSum,
    Binary,
    Fill,
}

impl StdPrimitivePipelineKind {
    fn matches_recipe(
        self,
        selected: &super::pipeline_recipe::PipelineRecipe,
        canonical: &super::pipeline_recipe::PipelineRecipe,
    ) -> bool {
        selected.layout == self.layout_tag()
            && selected.layout == canonical.layout
            && selected.shader == canonical.shader
            && selected.workgroup == canonical.workgroup
    }

    fn layout_tag(self) -> &'static str {
        match self {
            Self::DimensionalSum => "runmat-std-dim-sum-validated-bgl",
            Self::Binary => "runmat-std-binary-validated-bgl",
            Self::Fill => "runmat-std-fill-validated-bgl",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::DimensionalSum => "runmat-std-dim-sum-validated",
            Self::Binary => "runmat-std-binary-validated",
            Self::Fill => "runmat-std-fill-validated",
        }
    }

    pub(super) fn recipe(self, provider: &WgpuProvider) -> super::pipeline_recipe::PipelineRecipe {
        let (shader, workgroup) = match self {
            Self::DimensionalSum => (
                provider
                    .pipelines
                    .reduce_dim_sum_mean
                    .frozen_shader_source(),
                provider.linear_launch.workgroup(),
            ),
            Self::Binary => (
                provider.pipelines.binary.frozen_shader_source(),
                crate::backend::wgpu::config::WORKGROUP_SIZE,
            ),
            Self::Fill => (
                provider.pipelines.fill.frozen_shader_source(),
                provider.linear_launch.workgroup(),
            ),
        };
        super::pipeline_recipe::PipelineRecipe {
            shader: shader.to_owned(),
            layout: self.layout_tag().into(),
            workgroup,
        }
    }
}

impl ReductionExtremaPipelineKind {
    fn matches_recipe(
        self,
        selected: &super::pipeline_recipe::PipelineRecipe,
        canonical: &super::pipeline_recipe::PipelineRecipe,
    ) -> bool {
        selected.layout == self.layout_tag()
            && selected.layout == canonical.layout
            && selected.shader == canonical.shader
            && selected.workgroup == canonical.workgroup
    }

    pub(super) fn layout_tag(self) -> &'static str {
        match self {
            Self::Global => "runmat-reduce-global-validated-bgl",
            Self::Dimensional => "runmat-reduce-dim-minmax-validated-bgl",
            Self::IntegerDimensional => "runmat-integer-reduce-dim-extrema-validated-bgl",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Global => "runmat-reduce-global-validated",
            Self::Dimensional => "runmat-reduce-dim-minmax-validated",
            Self::IntegerDimensional => "runmat-integer-reduce-dim-extrema-validated",
        }
    }

    pub(super) fn recipe(self, provider: &WgpuProvider) -> super::pipeline_recipe::PipelineRecipe {
        let (source, workgroup) = match self {
            Self::Global => (
                provider
                    .pipelines
                    .reduce_global
                    .frozen_shader_source()
                    .to_owned(),
                provider.global_reduce_launch.workgroup(),
            ),
            Self::Dimensional => (
                provider
                    .pipelines
                    .reduce_dim_minmax
                    .frozen_shader_source()
                    .to_owned(),
                provider.linear_launch.workgroup(),
            ),
            Self::IntegerDimensional => {
                let scalar = match provider.precision {
                    crate::backend::wgpu::types::NumericPrecision::F32 => "f32",
                    crate::backend::wgpu::types::NumericPrecision::F64 => "f64",
                };
                (
                    crate::backend::wgpu::shaders::integer::extrema_dim_shader(
                        scalar,
                        provider.workgroup_config.scalar,
                    ),
                    provider.workgroup_config.scalar,
                )
            }
        };
        super::pipeline_recipe::PipelineRecipe {
            shader: source,
            layout: self.layout_tag().into(),
            workgroup,
        }
    }
}

impl IntegerBinaryPipelineKind {
    pub(super) fn layout_tag(self) -> &'static str {
        match self {
            Self::Arithmetic => "runmat-integer-arithmetic-bgl",
            Self::MinMax => "runmat-integer-minmax-bgl",
        }
    }

    pub(super) fn label(self) -> &'static str {
        match self {
            Self::Arithmetic => "runmat-integer-arithmetic",
            Self::MinMax => "runmat-integer-minmax",
        }
    }

    pub(super) fn shader(self, workgroup: u32) -> String {
        match self {
            Self::Arithmetic => {
                crate::backend::wgpu::shaders::integer::arithmetic_shader(workgroup)
            }
            Self::MinMax => crate::backend::wgpu::shaders::integer::minmax_shader(workgroup),
        }
    }
}

impl IndexingPipelineKind {
    fn layout_tag(self) -> String {
        match self {
            Self::ScatterColumn => "runmat-scatter-col-bgl".into(),
            Self::ScatterRow => "runmat-scatter-row-bgl".into(),
            Self::Sub2Ind { inputs } => format!("runmat-sub2ind-layout-{inputs}"),
            Self::Ind2Sub { outputs } => format!("runmat-ind2sub-layout-{outputs}"),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::ScatterColumn => "runmat-scatter-col",
            Self::ScatterRow => "runmat-scatter-row",
            Self::Sub2Ind { .. } => "runmat-sub2ind",
            Self::Ind2Sub { .. } => "runmat-ind2sub",
        }
    }
}

impl ValidatedFusionPipeline {
    fn require_kind(
        &self,
        recipe: &super::pipeline_recipe::PipelineRecipe,
        kind: ValidatedPipelineKind,
    ) -> Result<()> {
        ensure!(
            self.key == recipe.key() && self.recipe == *recipe && self.kind == kind,
            "validated pipeline route does not match its physical recipe"
        );
        Ok(())
    }

    pub(super) fn require_recipe(
        &self,
        recipe: &super::pipeline_recipe::PipelineRecipe,
        inputs: usize,
        layout_kind: super::pipeline_recipe::FusionPipelineLayout,
    ) -> Result<()> {
        ensure!(
            self.key == recipe.key()
                && self.recipe == *recipe
                && self.kind
                    == ValidatedPipelineKind::Fusion {
                        inputs,
                        layout: layout_kind,
                    },
            "validated fused pipeline does not match physical recipe"
        );
        Ok(())
    }

    pub(super) fn require_modulation(
        &self,
        recipe: &super::pipeline_recipe::PipelineRecipe,
    ) -> Result<()> {
        ensure!(
            self.key == recipe.key()
                && self.recipe == *recipe
                && recipe.layout == crate::backend::wgpu::bindings::MODULATION_LAYOUT_TAG
                && self.kind == ValidatedPipelineKind::Modulation,
            "validated modulation pipeline does not match physical recipe"
        );
        Ok(())
    }

    pub(super) fn require_softmax_rows(
        &self,
        recipe: &super::pipeline_recipe::PipelineRecipe,
    ) -> Result<()> {
        ensure!(
            self.key == recipe.key()
                && self.recipe == *recipe
                && recipe.layout == crate::backend::wgpu::bindings::SOFTMAX_ROWS_LAYOUT_TAG
                && self.kind == ValidatedPipelineKind::SoftmaxRows,
            "validated softmax rows pipeline does not match physical recipe"
        );
        Ok(())
    }

    pub(super) fn require_triangular_solve(
        &self,
        recipe: &super::pipeline_recipe::PipelineRecipe,
    ) -> Result<()> {
        ensure!(
            self.key == recipe.key()
                && self.recipe == *recipe
                && recipe.layout == crate::backend::wgpu::bindings::TRIANGULAR_SOLVE_LAYOUT_TAG
                && self.kind == ValidatedPipelineKind::TriangularSolve,
            "validated triangular solve pipeline does not match physical recipe"
        );
        Ok(())
    }

    pub(super) fn require_integer_comparison(
        &self,
        recipe: &super::pipeline_recipe::PipelineRecipe,
    ) -> Result<()> {
        ensure!(
            recipe.layout == crate::backend::wgpu::bindings::INTEGER_COMPARE_LAYOUT_TAG,
            "integer comparison physical binding recipe changed"
        );
        self.require_kind(recipe, ValidatedPipelineKind::IntegerComparison)
    }

    pub(super) fn require_integer_binary(
        &self,
        recipe: &super::pipeline_recipe::PipelineRecipe,
        kind: IntegerBinaryPipelineKind,
    ) -> Result<()> {
        ensure!(
            recipe.layout == kind.layout_tag() && recipe.shader == kind.shader(recipe.workgroup),
            "integer binary physical shader or layout changed"
        );
        self.require_kind(recipe, ValidatedPipelineKind::IntegerBinary(kind))
    }

    pub(super) fn require_reduction_extrema(
        &self,
        recipe: &super::pipeline_recipe::PipelineRecipe,
        kind: ReductionExtremaPipelineKind,
    ) -> Result<()> {
        ensure!(
            recipe.layout == kind.layout_tag(),
            "reduction extrema physical layout changed"
        );
        self.require_kind(recipe, ValidatedPipelineKind::ReductionExtrema(kind))
    }

    pub(super) fn require_std_primitive(
        &self,
        recipe: &super::pipeline_recipe::PipelineRecipe,
        kind: StdPrimitivePipelineKind,
    ) -> Result<()> {
        ensure!(
            recipe.layout == kind.layout_tag(),
            "std primitive layout changed"
        );
        self.require_kind(recipe, ValidatedPipelineKind::StdPrimitive(kind))
    }

    pub(super) fn require_integer_cast(
        &self,
        recipe: &super::pipeline_recipe::PipelineRecipe,
        source: crate::backend::wgpu::types::NumericPrecision,
    ) -> Result<()> {
        let canonical =
            super::pipeline_recipe::PipelineRecipe::integer_cast(source, recipe.workgroup);
        ensure!(
            recipe.layout == canonical.layout && recipe.shader == canonical.shader,
            "integer cast recipe differs from selected source"
        );
        self.require_kind(recipe, ValidatedPipelineKind::IntegerCast { source })
    }

    pub(super) fn require_floating_cast(
        &self,
        recipe: &super::pipeline_recipe::PipelineRecipe,
        source: crate::backend::wgpu::types::NumericPrecision,
        target: crate::backend::wgpu::types::NumericPrecision,
    ) -> Result<()> {
        let canonical =
            super::pipeline_recipe::PipelineRecipe::floating_cast(source, target, recipe.workgroup);
        ensure!(
            recipe.layout == canonical.layout && recipe.shader == canonical.shader,
            "floating cast recipe differs from selected source and target"
        );
        self.require_kind(
            recipe,
            ValidatedPipelineKind::FloatingCast { source, target },
        )
    }

    pub(super) fn require_integer_sum_cast(
        &self,
        recipe: &super::pipeline_recipe::PipelineRecipe,
    ) -> Result<()> {
        ensure!(
            self.key == recipe.key()
                && self.recipe == *recipe
                && recipe.layout == super::pipeline_recipe::INTEGER_SUM_CAST_LAYOUT_TAG
                && recipe.shader
                    == super::pipeline_recipe::PipelineRecipe::integer_sum_cast(recipe.workgroup)
                        .shader
                && self.kind == ValidatedPipelineKind::IntegerSumCast,
            "validated integer sum cast pipeline differs from its selected recipe"
        );
        Ok(())
    }

    pub(super) fn require_dimension_n_sum(
        &self,
        recipe: &super::pipeline_recipe::PipelineRecipe,
    ) -> Result<()> {
        ensure!(
            self.key == recipe.key()
                && self.recipe == *recipe
                && recipe.layout == super::pipeline_recipe::DIMENSION_N_SUM_F64_LAYOUT_TAG
                && recipe.shader
                    == super::pipeline_recipe::PipelineRecipe::dimension_n_sum_f64(
                        recipe.workgroup
                    )
                    .shader
                && self.kind == ValidatedPipelineKind::DimensionNSum,
            "validated N-D sum pipeline differs from its selected recipe"
        );
        Ok(())
    }

    pub(super) fn require_matmul_epilogue(
        &self,
        recipe: &super::pipeline_recipe::PipelineRecipe,
    ) -> Result<()> {
        ensure!(
            self.key == recipe.key()
                && self.recipe == *recipe
                && recipe.layout == crate::backend::wgpu::bindings::MATMUL_EPILOGUE_LAYOUT_TAG
                && self.kind == ValidatedPipelineKind::MatmulEpilogue,
            "validated matmul epilogue pipeline does not match physical recipe"
        );
        Ok(())
    }

    pub(super) fn require_indexing(
        &self,
        recipe: &super::pipeline_recipe::PipelineRecipe,
        indexing: IndexingPipelineKind,
    ) -> Result<()> {
        ensure!(
            self.key == recipe.key()
                && self.recipe == *recipe
                && self.kind == ValidatedPipelineKind::Indexing(indexing)
                && recipe.layout == indexing.layout_tag(),
            "validated indexing pipeline does not match physical recipe"
        );
        Ok(())
    }

    pub(super) fn require_round_digits(
        &self,
        recipe: &super::pipeline_recipe::PipelineRecipe,
        significant: bool,
    ) -> Result<()> {
        let layout = if significant {
            crate::backend::wgpu::bindings::ROUND_SIGNIFICANT_LAYOUT_TAG
        } else {
            crate::backend::wgpu::bindings::ROUND_DECIMALS_LAYOUT_TAG
        };
        ensure!(
            recipe.layout == layout,
            "round digits physical binding recipe changed"
        );
        self.require_kind(recipe, ValidatedPipelineKind::RoundDigits { significant })
    }
}

fn select_validated_winner<T>(
    cache: &std::sync::Mutex<std::collections::HashMap<u64, Arc<T>>>,
    key: u64,
    candidate: Arc<T>,
) -> Result<(Arc<T>, bool)> {
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("validated fusion cache poisoned"))?;
    Ok(match cache.get(&key) {
        Some(winner) => (winner.clone(), false),
        None => {
            cache.insert(key, candidate.clone());
            (candidate, true)
        }
    })
}

#[cfg(test)]
#[path = "validated_fusion/reduction_extrema_tests.rs"]
mod reduction_extrema_tests;
#[cfg(test)]
#[path = "validated_fusion/reduction_std_tests.rs"]
mod reduction_std_tests;
#[cfg(test)]
#[path = "validated_fusion/tests.rs"]
mod tests;

impl WgpuProvider {
    pub(super) async fn prepare_validated_fusion_pipeline(
        &self,
        recipe: &super::pipeline_recipe::PipelineRecipe,
        inputs: usize,
        layout_kind: super::pipeline_recipe::FusionPipelineLayout,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Arc<ValidatedFusionPipeline>> {
        let label = match layout_kind {
            super::pipeline_recipe::FusionPipelineLayout::Single => "runmat-fusion-pipeline",
            super::pipeline_recipe::FusionPipelineLayout::Multiple(_) => {
                "runmat-fusion-multi-pipeline"
            }
        };
        self.prepare_validated_pipeline(
            recipe,
            ValidatedPipelineKind::Fusion {
                inputs,
                layout: layout_kind,
            },
            label,
        )
        .await
    }

    pub(super) async fn prepare_validated_std_primitive_pipeline(
        &self,
        recipe: &super::pipeline_recipe::PipelineRecipe,
        kind: StdPrimitivePipelineKind,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Arc<ValidatedFusionPipeline>> {
        self.prepare_validated_pipeline(
            recipe,
            ValidatedPipelineKind::StdPrimitive(kind),
            kind.label(),
        )
        .await
    }

    pub(super) async fn prepare_validated_modulation_pipeline(
        &self,
        recipe: &super::pipeline_recipe::PipelineRecipe,
        label: &'static str,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Arc<ValidatedFusionPipeline>> {
        self.prepare_validated_pipeline(recipe, ValidatedPipelineKind::Modulation, label)
            .await
    }

    pub(super) async fn prepare_validated_softmax_rows_pipeline(
        &self,
        recipe: &super::pipeline_recipe::PipelineRecipe,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Arc<ValidatedFusionPipeline>> {
        self.prepare_validated_pipeline(
            recipe,
            ValidatedPipelineKind::SoftmaxRows,
            "runmat-softmax-rows-pipeline",
        )
        .await
    }

    pub(super) async fn prepare_validated_triangular_solve_pipeline(
        &self,
        recipe: &super::pipeline_recipe::PipelineRecipe,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Arc<ValidatedFusionPipeline>> {
        self.prepare_validated_pipeline(
            recipe,
            ValidatedPipelineKind::TriangularSolve,
            "runmat-triangular-solve-pipeline",
        )
        .await
    }

    pub(super) async fn prepare_validated_integer_comparison_pipeline(
        &self,
        recipe: &super::pipeline_recipe::PipelineRecipe,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Arc<ValidatedFusionPipeline>> {
        self.prepare_validated_pipeline(
            recipe,
            ValidatedPipelineKind::IntegerComparison,
            "runmat-integer-compare",
        )
        .await
    }

    pub(super) async fn prepare_validated_integer_binary_pipeline(
        &self,
        recipe: &super::pipeline_recipe::PipelineRecipe,
        kind: IntegerBinaryPipelineKind,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Arc<ValidatedFusionPipeline>> {
        self.prepare_validated_pipeline(
            recipe,
            ValidatedPipelineKind::IntegerBinary(kind),
            kind.label(),
        )
        .await
    }

    pub(super) async fn prepare_validated_reduction_extrema_pipeline(
        &self,
        recipe: &super::pipeline_recipe::PipelineRecipe,
        kind: ReductionExtremaPipelineKind,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Arc<ValidatedFusionPipeline>> {
        self.prepare_validated_pipeline(
            recipe,
            ValidatedPipelineKind::ReductionExtrema(kind),
            kind.label(),
        )
        .await
    }

    pub(super) async fn prepare_validated_integer_cast_pipeline(
        &self,
        recipe: &super::pipeline_recipe::PipelineRecipe,
        source: crate::backend::wgpu::types::NumericPrecision,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Arc<ValidatedFusionPipeline>> {
        self.prepare_validated_pipeline(
            recipe,
            ValidatedPipelineKind::IntegerCast { source },
            "runmat-integer-cast",
        )
        .await
    }

    pub(super) async fn prepare_validated_floating_cast_pipeline(
        &self,
        recipe: &super::pipeline_recipe::PipelineRecipe,
        source: crate::backend::wgpu::types::NumericPrecision,
        target: crate::backend::wgpu::types::NumericPrecision,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Arc<ValidatedFusionPipeline>> {
        self.prepare_validated_pipeline(
            recipe,
            ValidatedPipelineKind::FloatingCast { source, target },
            "runmat-floating-cast",
        )
        .await
    }

    pub(super) async fn prepare_validated_integer_sum_cast_pipeline(
        &self,
        recipe: &super::pipeline_recipe::PipelineRecipe,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Arc<ValidatedFusionPipeline>> {
        self.prepare_validated_pipeline(
            recipe,
            ValidatedPipelineKind::IntegerSumCast,
            "runmat-integer-sum-floating-cast",
        )
        .await
    }

    pub(super) async fn prepare_validated_dimension_n_sum_pipeline(
        &self,
        recipe: &super::pipeline_recipe::PipelineRecipe,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Arc<ValidatedFusionPipeline>> {
        self.prepare_validated_pipeline(
            recipe,
            ValidatedPipelineKind::DimensionNSum,
            "runmat-nd-sum-f64",
        )
        .await
    }

    pub(super) async fn prepare_validated_matmul_epilogue_pipeline(
        &self,
        recipe: &super::pipeline_recipe::PipelineRecipe,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Arc<ValidatedFusionPipeline>> {
        self.prepare_validated_pipeline(
            recipe,
            ValidatedPipelineKind::MatmulEpilogue,
            "runmat-matmul-epilogue",
        )
        .await
    }

    pub(super) async fn prepare_validated_round_digits_pipeline(
        &self,
        recipe: &super::pipeline_recipe::PipelineRecipe,
        significant: bool,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Arc<ValidatedFusionPipeline>> {
        self.prepare_validated_pipeline(
            recipe,
            ValidatedPipelineKind::RoundDigits { significant },
            "runmat-round-digits-pipeline",
        )
        .await
    }

    pub(super) async fn prepare_validated_indexing_pipeline(
        &self,
        recipe: &super::pipeline_recipe::PipelineRecipe,
        indexing: IndexingPipelineKind,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Arc<ValidatedFusionPipeline>> {
        self.prepare_validated_pipeline(
            recipe,
            ValidatedPipelineKind::Indexing(indexing),
            indexing.label(),
        )
        .await
    }

    async fn prepare_validated_pipeline(
        &self,
        recipe: &super::pipeline_recipe::PipelineRecipe,
        kind: ValidatedPipelineKind,
        label: &'static str,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Arc<ValidatedFusionPipeline>> {
        use runmat_accelerate_api::ProviderExecutionError;
        let layout_matches = match kind {
            ValidatedPipelineKind::Fusion { inputs, layout } => {
                super::pipeline_recipe::PipelineRecipe::fusion(
                    &recipe.shader,
                    inputs,
                    layout,
                    recipe.workgroup,
                )
                .layout
                    == recipe.layout
            }
            ValidatedPipelineKind::Modulation => {
                recipe.layout == crate::backend::wgpu::bindings::MODULATION_LAYOUT_TAG
            }
            ValidatedPipelineKind::SoftmaxRows => {
                recipe.layout == crate::backend::wgpu::bindings::SOFTMAX_ROWS_LAYOUT_TAG
            }
            ValidatedPipelineKind::TriangularSolve => {
                recipe.layout == crate::backend::wgpu::bindings::TRIANGULAR_SOLVE_LAYOUT_TAG
            }
            ValidatedPipelineKind::IntegerSumCast => {
                let canonical =
                    super::pipeline_recipe::PipelineRecipe::integer_sum_cast(recipe.workgroup);
                recipe.layout == canonical.layout && recipe.shader == canonical.shader
            }
            ValidatedPipelineKind::IntegerCast { source } => {
                let canonical =
                    super::pipeline_recipe::PipelineRecipe::integer_cast(source, recipe.workgroup);
                recipe.layout == canonical.layout && recipe.shader == canonical.shader
            }
            ValidatedPipelineKind::FloatingCast { source, target } => {
                let canonical = super::pipeline_recipe::PipelineRecipe::floating_cast(
                    source,
                    target,
                    recipe.workgroup,
                );
                recipe.layout == canonical.layout && recipe.shader == canonical.shader
            }
            ValidatedPipelineKind::DimensionNSum => {
                let canonical =
                    super::pipeline_recipe::PipelineRecipe::dimension_n_sum_f64(recipe.workgroup);
                recipe.layout == canonical.layout && recipe.shader == canonical.shader
            }
            ValidatedPipelineKind::Indexing(indexing) => recipe.layout == indexing.layout_tag(),
            ValidatedPipelineKind::MatmulEpilogue => {
                recipe.layout == crate::backend::wgpu::bindings::MATMUL_EPILOGUE_LAYOUT_TAG
            }
            ValidatedPipelineKind::IntegerComparison => {
                recipe.layout == crate::backend::wgpu::bindings::INTEGER_COMPARE_LAYOUT_TAG
            }
            ValidatedPipelineKind::IntegerBinary(binary) => {
                recipe.layout == binary.layout_tag()
                    && recipe.shader == binary.shader(recipe.workgroup)
            }
            ValidatedPipelineKind::ReductionExtrema(reduction) => {
                let canonical = reduction.recipe(self);
                reduction.matches_recipe(recipe, &canonical)
            }
            ValidatedPipelineKind::StdPrimitive(primitive) => {
                let canonical = primitive.recipe(self);
                primitive.matches_recipe(recipe, &canonical)
            }
            ValidatedPipelineKind::RoundDigits { significant } => {
                recipe.layout
                    == if significant {
                        crate::backend::wgpu::bindings::ROUND_SIGNIFICANT_LAYOUT_TAG
                    } else {
                        crate::backend::wgpu::bindings::ROUND_DECIMALS_LAYOUT_TAG
                    }
            }
        };
        if !layout_matches {
            return Err(ProviderExecutionError::from(anyhow::anyhow!(
                "validated pipeline layout identity changed before preparation"
            )));
        }
        if crate::backend::wgpu::cache::key::has_unresolved_launch_tokens(recipe.shader.as_bytes())
        {
            return Err(ProviderExecutionError::from(anyhow::anyhow!(
                "validated pipeline source has unresolved launch tokens"
            )));
        }
        let key = recipe.key();
        let observation = super::pipeline_observation::PipelineObservation::begin();
        let cached = self
            .validated_fusion_cache
            .lock()
            .map_err(|_| anyhow::anyhow!("validated fusion cache poisoned"))?
            .get(&key)
            .cloned();
        if let Some(cached) = cached {
            cached
                .require_kind(recipe, kind)
                .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?;
            self.metrics.inc_hit();
            observation.finish(runmat_execution::CandidatePreparationState::Warm);
            return Ok(cached);
        }
        let warmed = self
            .warmed_pipeline_cache
            .lock()
            .map_err(|_| anyhow::anyhow!("warmed pipeline cache poisoned"))?
            .get(&key)
            .cloned();
        if let Some(warmed) = warmed.filter(|receipt| receipt.recipe == *recipe) {
            // The disk record supplied only a physical recipe and a scoped
            // driver receipt. This typed route has now admitted its kind and
            // exact source/layout/workgroup before the pair can be published.
            let candidate = Arc::new(ValidatedFusionPipeline {
                key,
                kind,
                recipe: recipe.clone(),
                layout: warmed.layout.clone(),
                pipeline: warmed.pipeline.clone(),
            });
            let (selected, _) =
                select_validated_winner(&self.validated_fusion_cache, key, candidate)
                    .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?;
            selected
                .require_kind(recipe, kind)
                .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?;
            if let Ok(mut tags) = self.bind_group_layout_tags.lock() {
                let ptr = selected.layout.as_ref() as *const wgpu::BindGroupLayout as usize;
                tags.entry(ptr).or_insert_with(|| recipe.layout.clone());
            }
            self.metrics.inc_hit();
            observation.finish(runmat_execution::CandidatePreparationState::Warm);
            return Ok(selected);
        }
        let prepared_shader = crate::backend::wgpu::shader_compilation::prepare(
            &recipe.shader,
            self.access.enabled_features(),
            None,
        )
        .map_err(ProviderExecutionError::from)?;
        let receipt = self
            .access
            .with_profiled_scoped(&self.emission_authority, |emission| -> Result<_> {
                let layout = Arc::new(match kind {
                    ValidatedPipelineKind::Fusion {
                        inputs,
                        layout: super::pipeline_recipe::FusionPipelineLayout::Single,
                    } => crate::backend::wgpu::bindings::build_fusion_bgl(emission, inputs),
                    ValidatedPipelineKind::Fusion {
                        inputs,
                        layout: super::pipeline_recipe::FusionPipelineLayout::Multiple(outputs),
                    } => crate::backend::wgpu::bindings::build_fusion_multi_bgl(
                        emission, inputs, outputs,
                    ),
                    ValidatedPipelineKind::Modulation => {
                        crate::backend::wgpu::bindings::build_modulation_bgl(emission)
                    }
                    ValidatedPipelineKind::SoftmaxRows => {
                        crate::backend::wgpu::bindings::build_softmax_rows_bgl(emission)
                    }
                    ValidatedPipelineKind::TriangularSolve => {
                        crate::backend::wgpu::bindings::build_triangular_solve_bgl(emission)
                    }
                    ValidatedPipelineKind::IntegerSumCast
                    | ValidatedPipelineKind::DimensionNSum
                    | ValidatedPipelineKind::IntegerCast { .. }
                    | ValidatedPipelineKind::FloatingCast { .. } => emission.device().create_bind_group_layout(
                        &wgpu::BindGroupLayoutDescriptor {
                            label: Some(label),
                            entries: &[
                                crate::backend::wgpu::bindings::storage_read_entry(0),
                                crate::backend::wgpu::bindings::storage_read_write_entry(1),
                                crate::backend::wgpu::bindings::uniform_entry(2),
                            ],
                        },
                    ),
                    ValidatedPipelineKind::ReductionExtrema(reduction) => emission.device().create_bind_group_layout(
                        &wgpu::BindGroupLayoutDescriptor {
                            label: Some(label),
                            entries: &match reduction {
                                ReductionExtremaPipelineKind::Global => vec![
                                    crate::backend::wgpu::bindings::storage_read_entry(0),
                                    crate::backend::wgpu::bindings::storage_read_write_entry(1),
                                    crate::backend::wgpu::bindings::uniform_entry(2),
                                ],
                                ReductionExtremaPipelineKind::Dimensional | ReductionExtremaPipelineKind::IntegerDimensional => vec![
                                    crate::backend::wgpu::bindings::storage_read_entry(0),
                                    crate::backend::wgpu::bindings::storage_read_write_entry(1),
                                    crate::backend::wgpu::bindings::storage_read_write_entry(2),
                                    crate::backend::wgpu::bindings::uniform_entry(3),
                                ],
                            },
                        },
                    ),
                    ValidatedPipelineKind::StdPrimitive(primitive) => {
                        let entries = match primitive {
                            StdPrimitivePipelineKind::DimensionalSum => vec![
                                crate::backend::wgpu::bindings::storage_read_entry(0),
                                crate::backend::wgpu::bindings::storage_read_write_entry(1),
                                crate::backend::wgpu::bindings::uniform_entry(2),
                            ],
                            StdPrimitivePipelineKind::Binary => vec![
                                crate::backend::wgpu::bindings::storage_read_entry(0),
                                crate::backend::wgpu::bindings::storage_read_entry(1),
                                crate::backend::wgpu::bindings::storage_read_write_entry(2),
                                crate::backend::wgpu::bindings::uniform_entry(3),
                            ],
                            StdPrimitivePipelineKind::Fill => vec![
                                crate::backend::wgpu::bindings::storage_read_write_entry(0),
                                crate::backend::wgpu::bindings::uniform_entry(1),
                            ],
                        };
                        emission.device().create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                            label: Some(label),
                            entries: &entries,
                        })
                    }
                    ValidatedPipelineKind::Indexing(indexing) => match indexing {
                        IndexingPipelineKind::ScatterColumn => crate::backend::wgpu::bindings::build_scatter_col_bgl(emission),
                        IndexingPipelineKind::ScatterRow => crate::backend::wgpu::bindings::build_scatter_row_bgl(emission),
                        IndexingPipelineKind::Sub2Ind { inputs } => crate::backend::wgpu::bindings::build_sub2ind_bgl(emission, inputs),
                        IndexingPipelineKind::Ind2Sub { outputs } => crate::backend::wgpu::bindings::build_ind2sub_bgl(emission, outputs),
                    },
                    ValidatedPipelineKind::MatmulEpilogue => {
                        crate::backend::wgpu::bindings::build_matmul_epilogue_bgl(emission)
                    }
                    ValidatedPipelineKind::IntegerComparison => {
                        crate::backend::wgpu::bindings::build_integer_compare_bgl(emission)
                    }
                    ValidatedPipelineKind::IntegerBinary(_) => emission.device().create_bind_group_layout(
                        &wgpu::BindGroupLayoutDescriptor {
                            label: Some(label),
                            entries: &[
                                crate::backend::wgpu::bindings::storage_read_entry(0),
                                crate::backend::wgpu::bindings::storage_read_entry(1),
                                crate::backend::wgpu::bindings::storage_read_write_entry(2),
                                crate::backend::wgpu::bindings::uniform_entry(3),
                            ],
                        },
                    ),
                    ValidatedPipelineKind::RoundDigits { .. } => {
                        crate::backend::wgpu::bindings::build_round_digits_bgl(emission)
                    }
                });
                let pipeline_layout = crate::backend::wgpu::pipelines::create_pipeline_layout(
                    emission,
                    label,
                    layout.as_ref(),
                );
                let module = crate::backend::wgpu::shader_compilation::create_prepared(
                    emission,
                    label,
                    &prepared_shader,
                )?;
                let pipeline = super::pipeline_cache::PipelineOwner::unbounded(emission.device().create_compute_pipeline(
                    &crate::backend::wgpu::compat::wgpu_compute_pipeline_descriptor! {
                        label: Some(label), layout: Some(&pipeline_layout), module: &module, entry_point: "main",
                    },
                ));
                Ok(ValidatedFusionPipeline {
                    key,
                    kind,
                    recipe: recipe.clone(),
                    layout,
                    pipeline,
                })
            })
            .map_err(|error| ProviderExecutionError::from_source(error).with_unproven_effects())?;
        let candidate = Arc::new(
            receipt
                .resolve()
                .await
                .map_err(|error| {
                    ProviderExecutionError::from_source(error).with_unproven_effects()
                })?
                .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?,
        );
        // Race-safe publication. A losing validated candidate is discarded;
        // neither a cache lock nor device gate spans the scope resolution.
        let (selected, cold) =
            select_validated_winner(&self.validated_fusion_cache, key, candidate)
                .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?;
        selected
            .require_kind(recipe, kind)
            .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?;
        if cold {
            self.metrics.inc_miss();
            if let Ok(mut tags) = self.bind_group_layout_tags.lock() {
                let ptr = selected.layout.as_ref() as *const wgpu::BindGroupLayout as usize;
                tags.entry(ptr).or_insert_with(|| recipe.layout.clone());
            }
            self.persist_pipeline_meta(
                key,
                label,
                Some(&recipe.layout),
                Some(recipe.workgroup),
                Some(recipe.shader.as_bytes()),
            );
        } else {
            self.metrics.inc_hit();
        }
        observation.finish(if cold {
            runmat_execution::CandidatePreparationState::Cold
        } else {
            runmat_execution::CandidatePreparationState::Warm
        });
        Ok(selected)
    }
}
