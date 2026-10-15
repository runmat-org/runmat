//! Exact physical probe and real continuation for one native unary invocation.
//! Planning is not placement, resource admission, or numerical evidence. The
//! opaque domain witness is minted only by consuming the retained probe recipes.
use super::compound_execution::{check_cancelled, retire_output_after_failure, with_prior_effect};
use crate::{
    AccelProvider, GpuTensorHandle, ProviderExecutedRealElementwise, ProviderExecutionError,
    ProviderExecutionResult, ProviderExtremaKind, ProviderFeasibilityQuery,
    ProviderGlobalExtremaRecipe, ProviderRealElementwiseOperation, ProviderRealElementwiseRecipe,
    ProviderRealElementwiseRequest,
};
use runmat_execution::{cancellation::CancellationToken, Digest};
use runmat_types::{RealUnaryClassification, RealUnaryPredicate, ScalarUnaryMath};
mod continuation;
pub use continuation::{
    ProviderRealUnaryContinuation, ProviderRealUnaryContinuationKind,
    ProviderRealUnaryContinuations, ProviderRealUnaryExecuted, ProviderRealUnaryExecution,
    ProviderRealUnaryExecutionEvidence, ProviderRealUnaryPublication,
};

/// Immutable physical stages on the exact retained source view. Host/complex
/// continuations and their storage are admitted by the original runtime owner;
/// this physical recipe cannot grant those continuations or a completion right.
#[derive(Clone)]
pub struct ProviderRealUnaryCompound {
    source: GpuTensorHandle,
    predicate: RealUnaryPredicate,
    minimum: ProviderGlobalExtremaRecipe,
    maximum: Option<ProviderGlobalExtremaRecipe>,
    real: ProviderRealElementwiseRecipe,
}

impl std::fmt::Debug for ProviderRealUnaryCompound {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ProviderRealUnaryCompound")
            .field("source", &self.source.view_identity())
            .field("predicate", &self.predicate)
            .finish_non_exhaustive()
    }
}

/// Nonserializable result of actual scalar readbacks. It is not interchangeable
/// with a caller-supplied NumericalDomainEvidence or a provider feasibility quote.
pub struct ProviderRealUnaryDomain {
    source: crate::allocation::TensorViewIdentity,
    recipe: Digest,
    classification: RealUnaryClassification,
}

impl ProviderRealUnaryDomain {
    pub fn classification(&self) -> RealUnaryClassification {
        self.classification
    }
}

impl ProviderRealUnaryCompound {
    pub fn plan(
        owner: &'static dyn AccelProvider,
        source: &GpuTensorHandle,
        operation: ScalarUnaryMath,
    ) -> ProviderExecutionResult<Option<Self>> {
        let Some(predicate) = operation.real_predicate() else {
            return Ok(None);
        };
        let Some(minimum) =
            ProviderGlobalExtremaRecipe::plan(owner, source, ProviderExtremaKind::Minimum)?
        else {
            return Ok(None);
        };
        let maximum = if predicate == RealUnaryPredicate::ClosedUnitInterval {
            let Some(maximum) =
                ProviderGlobalExtremaRecipe::plan(owner, source, ProviderExtremaKind::Maximum)?
            else {
                return Ok(None);
            };
            Some(maximum)
        } else {
            None
        };
        let Some(real) = ProviderRealElementwiseRecipe::plan(
            owner,
            ProviderRealElementwiseRequest {
                operation: ProviderRealElementwiseOperation::UnaryMath(operation),
                source: minimum.request().source.clone(),
                source_buffer_bytes: source.physical().requested_bytes(),
            },
        )?
        else {
            return Ok(None);
        };
        Ok(Some(Self {
            source: source.clone(),
            predicate,
            minimum,
            maximum,
            real,
        }))
    }

    pub fn source(&self) -> &GpuTensorHandle {
        &self.source
    }
    pub fn predicate(&self) -> RealUnaryPredicate {
        self.predicate
    }
    pub fn probes(&self) -> impl Iterator<Item = &ProviderGlobalExtremaRecipe> {
        std::iter::once(&self.minimum).chain(self.maximum.iter())
    }
    pub fn real(&self) -> &ProviderRealElementwiseRecipe {
        &self.real
    }
    pub fn is_owned_by(&self, owner: &dyn AccelProvider) -> bool {
        self.minimum.is_owned_by(owner)
            && self
                .maximum
                .as_ref()
                .is_none_or(|recipe| recipe.is_owned_by(owner))
            && self.real.is_owned_by(owner)
    }
    pub fn identity(&self) -> anyhow::Result<Digest> {
        Ok(Digest::sha256_serialized(&(
            "real-unary-compound-v1",
            self.predicate,
            self.minimum.identity()?,
            self.maximum
                .as_ref()
                .map(ProviderGlobalExtremaRecipe::identity)
                .transpose()?,
            self.real.identity()?,
        ))?)
    }

    /// Call only inside the complete route's admitted resource/completion
    /// scope. Cancellation checks bracket every effectful child await. They
    /// never substitute for the child's actual submission/publication fence.
    pub async fn probe(
        &self,
        cancellation: Option<&CancellationToken>,
    ) -> ProviderExecutionResult<ProviderRealUnaryDomain> {
        self.validate_source()?;
        let mut extrema = [0.0; 2];
        let mut count = 0;
        let mut effected = false;
        for recipe in self.probes() {
            check_cancelled(cancellation, effected)?;
            let executed = with_prior_effect(recipe.execute_checked().await, effected)?;
            effected = true;
            check_cancelled(cancellation, effected)?;
            let scalar = with_prior_effect(executed.read_scalar().await, effected)?;
            check_cancelled(cancellation, effected)?;
            with_prior_effect(self.validate_source(), effected)?;
            if scalar.source() != &self.source.view_identity()
                || scalar.kind() != recipe.request().kind
                || scalar.recipe_identity()
                    != with_prior_effect(
                        recipe.identity().map_err(ProviderExecutionError::from),
                        effected,
                    )?
            {
                return Err(unproven().after_effect());
            }
            extrema[count] = match scalar.value() {
                crate::scalar_program::ScalarLiteral::F32Bits(bits) => {
                    f64::from(f32::from_bits(bits))
                }
                crate::scalar_program::ScalarLiteral::F64Bits(bits) => f64::from_bits(bits),
            };
            count += 1;
        }
        // Global recipes include NaNs rather than omitting them. A NaN probe
        // therefore remains indeterminate and must take the owning fallback.
        let classification = self.predicate.classify(extrema[..count].iter().copied());
        Ok(ProviderRealUnaryDomain {
            source: self.source.view_identity(),
            recipe: with_prior_effect(
                self.identity().map_err(ProviderExecutionError::from),
                effected,
            )?,
            classification,
        })
    }

    pub fn authenticates(&self, evidence: &ProviderRealUnaryDomain) -> bool {
        evidence.source == self.source.view_identity()
            && self.identity().ok() == Some(evidence.recipe)
            && self.validate_source().is_ok()
    }

    /// No caller can replace a consumed probe with a static Inside promise.
    /// Outside/indeterminate belong to the runtime's distinct typed host/error
    /// continuation and are never reported as Unsupported provider execution.
    pub async fn execute_real(
        &self,
        evidence: &ProviderRealUnaryDomain,
        cancellation: Option<&CancellationToken>,
    ) -> ProviderExecutionResult<ProviderExecutedRealElementwise> {
        // This public operation belongs to an already-consumed probe. Child
        // applicability cannot erase effects of that earlier stage.
        with_prior_effect(self.execute_real_child(evidence, cancellation).await, true)
    }

    // Only `consume` may interpret an actual child's pre-effect Unsupported as
    // a typed continuation. It never exposes that disposition as compound
    // pre-effect failure to a caller.
    async fn execute_real_child(
        &self,
        evidence: &ProviderRealUnaryDomain,
        cancellation: Option<&CancellationToken>,
    ) -> ProviderExecutionResult<ProviderExecutedRealElementwise> {
        check_cancelled(cancellation, true)?;
        if !self.authenticates(evidence)
            || evidence.classification != RealUnaryClassification::Inside
        {
            return Err(unproven().after_effect());
        }
        let output = self.real.execute(&self.source).await?;
        if let Err(primary) = check_cancelled(cancellation, true) {
            return Err(retire_output_after_failure(output.output, primary));
        }
        Ok(output)
    }

    fn validate_source(&self) -> ProviderExecutionResult<()> {
        self.source
            .allocation_lease()
            .validate_live()
            .map_err(ProviderExecutionError::from_source)?;
        let owner = crate::provider_for_handle(&self.source).ok_or_else(unproven)?;
        if !self.is_owned_by(owner)
            || ProviderFeasibilityQuery::exact_numeric_download(&self.source)
                .and_then(|query| query.inputs.into_iter().next())
                .as_ref()
                != Some(&self.minimum.request().source)
        {
            return Err(unproven());
        }
        Ok(())
    }
}

fn unproven() -> ProviderExecutionError {
    ProviderExecutionError::from(anyhow::anyhow!(
        "compound unary source or domain witness changed"
    ))
}
