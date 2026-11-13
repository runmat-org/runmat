//! Optional selected reduction sign refinement. The reduction declaration,
//! not this resolver, owns control parsing and ordered stage semantics.

use super::*;
use runmat_runtime::builtins::common::spec::{
    FusionAdmissionWork, FusionProofStage, FusionReductionContext, FusionSignEvidence,
};

fn control_materialization(controls: &[&Value]) -> Option<FusionProofStage> {
    let mut shape_dimensions = 0_u64;
    let mut string_bytes = 0_u64;
    let mut char_elements = 0_u64;
    for value in controls {
        let (shape, text, chars) = match value {
            Value::Num(_) | Value::Int(_) => (0, 0, 0),
            Value::String(value) => (
                0,
                u64::try_from(value.as_str().map_or(0, str::len)).ok()?,
                0,
            ),
            Value::Tensor(value) => (u64::try_from(value.shape().len()).ok()?, 0, 0),
            Value::StringArray(value) => (
                u64::try_from(value.shape.len()).ok()?,
                value.data().iter().try_fold(0_u64, |total, string| {
                    total.checked_add(u64::try_from(string.as_str().map_or(0, str::len)).ok()?)
                })?,
                0,
            ),
            Value::CharArray(value) => (
                u64::try_from(value.shape.len()).ok()?,
                0,
                u64::try_from(value.data().len()).ok()?,
            ),
            _ => return None,
        };
        shape_dimensions = shape_dimensions.checked_add(shape)?;
        string_bytes = string_bytes.checked_add(text)?;
        char_elements = char_elements.checked_add(chars)?;
    }
    Some(FusionProofStage::ControlMaterialization {
        values: u64::try_from(controls.len()).ok()?,
        shape_dimensions,
        string_bytes,
        char_elements,
    })
}

pub(super) async fn resolved_sign(
    program: &ResolvedProgram,
    inferred: &assembly::InferredOperation<'_>,
    values: &[Option<&Value>],
    snapshot: Option<SemanticBinding>,
    work: &mut FusionAdmissionWork,
) -> std::result::Result<
    Option<(
        FusionSignEvidence,
        runmat_types::ValueKindFact,
        runmat_types::StorageFact,
    )>,
    FusionResolveFailure,
> {
    let Some(binding) = snapshot else {
        return Ok(None);
    };
    let Some(reduction) = binding.declaration.reduction else {
        return Ok(None);
    };
    let Some((&input_id, control_ids)) = inferred.input_ids.split_first() else {
        return Ok(None);
    };
    let Some((input, control_values)) = inferred.arguments.split_first() else {
        return Ok(None);
    };
    let Some(input_numeric) = input.numeric() else {
        return Ok(None);
    };
    if program.sign.get(&input_id).copied()
        != Some(FusionSignEvidence::NonnegativeOrNaN {
            class: input_numeric.class,
        })
        || control_ids.len() != control_values.len()
        || values.len() != inferred.input_ids.len()
    {
        return Ok(None);
    }
    let Some(controls) = values[1..].iter().copied().collect::<Option<Vec<_>>>() else {
        return Ok(None);
    };
    let Some(materialization) = control_materialization(&controls) else {
        return Ok(None);
    };
    *work = FusionAdmissionWork::DomainProof {
        binding: binding.declaration.binding,
        provider_precision: program.provider_precision,
        stages: vec![
            materialization,
            FusionProofStage::RangePropagation {
                operands: u64::try_from(inferred.input_ids.len()).unwrap_or(u64::MAX),
            },
        ],
    };
    let controls: Vec<Value> = controls.into_iter().cloned().collect();
    let Some(projection) = reduction
        .prove_nonnegative_or_nan(
            FusionReductionContext {
                input,
                controls: &controls,
                requested_outputs: 1,
            },
            &inferred.fact,
        )
        .await?
    else {
        return Ok(None);
    };
    if inferred.fact.numeric().is_some_and(|numeric| {
        numeric.class != projection.class || numeric.domain != NumericDomain::Real
    }) {
        return Ok(None);
    }
    Ok(Some((
        FusionSignEvidence::NonnegativeOrNaN {
            class: projection.class,
        },
        runmat_types::ValueKindFact::Numeric(runmat_types::NumericFact {
            class: projection.class,
            domain: NumericDomain::Real,
        }),
        projection.storage,
    )))
}
