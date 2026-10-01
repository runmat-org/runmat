//! Exact capabilities of the logical hooks, not admission by operation family.
use super::*;
use runmat_builtins::{LogicalBinaryOperator, RelationalOperator};

#[cfg(test)]
mod tests;

#[derive(Clone, Copy)]
enum Operation {
    Cast,
    Binary(LogicalBinaryOperator),
    Compare(RelationalOperator),
}

impl Operation {
    fn identity(self) -> &'static str {
        use runmat_builtins::*;
        match self {
            Self::Cast => LOGICAL_CATALOG_ENTRY.identity.name,
            Self::Binary(LogicalBinaryOperator::And) => AND_CATALOG_ENTRY.identity.name,
            Self::Binary(LogicalBinaryOperator::Or) => OR_CATALOG_ENTRY.identity.name,
            Self::Binary(LogicalBinaryOperator::Xor) => XOR_CATALOG_ENTRY.identity.name,
            Self::Compare(RelationalOperator::Equal) => EQ_CATALOG_ENTRY.identity.name,
            Self::Compare(RelationalOperator::NotEqual) => NE_CATALOG_ENTRY.identity.name,
            Self::Compare(RelationalOperator::LessThan) => LT_CATALOG_ENTRY.identity.name,
            Self::Compare(RelationalOperator::LessThanOrEqual) => LE_CATALOG_ENTRY.identity.name,
            Self::Compare(RelationalOperator::GreaterThan) => GT_CATALOG_ENTRY.identity.name,
            Self::Compare(RelationalOperator::GreaterThanOrEqual) => GE_CATALOG_ENTRY.identity.name,
        }
    }

    fn arity(self) -> usize {
        if matches!(self, Self::Cast) {
            1
        } else {
            2
        }
    }
}

const OPERATIONS: &[Operation] = &[
    Operation::Cast,
    Operation::Binary(LogicalBinaryOperator::And),
    Operation::Binary(LogicalBinaryOperator::Or),
    Operation::Binary(LogicalBinaryOperator::Xor),
    Operation::Compare(RelationalOperator::Equal),
    Operation::Compare(RelationalOperator::NotEqual),
    Operation::Compare(RelationalOperator::LessThan),
    Operation::Compare(RelationalOperator::LessThanOrEqual),
    Operation::Compare(RelationalOperator::GreaterThan),
    Operation::Compare(RelationalOperator::GreaterThanOrEqual),
];

#[derive(Clone, Copy)]
pub(super) enum HookProfile {
    InProcess,
    #[cfg(feature = "wgpu")]
    Wgpu,
}

pub(super) fn append(snapshot: &mut ProviderCapabilitySnapshot) {
    snapshot.operations.extend(
        OPERATIONS
            .iter()
            .map(|operation| ProviderCapabilityOperation {
                identity: ProviderOperationIdentity::new(operation.identity()),
                family: ProviderOperationFamily::Elementwise,
            }),
    );
    // all/any need resolved reduction axes/NaN controls in the query before
    // their full-vs-dimensional, provider-specific ND support can be declared.
}

pub(super) fn validate(
    profile: HookProfile,
    _precision: ProviderPrecision,
    query: &ProviderFeasibilityQuery,
) -> Option<ProviderFeasibility> {
    let operation = OPERATIONS
        .iter()
        .copied()
        .find(|operation| operation.identity() == query.operation.0)?;
    let reject = |code, reason| Some(ProviderFeasibility::rejected(code, reason));
    if query.family != ProviderOperationFamily::Elementwise {
        return reject(
            ProviderRejectionCode::UnsupportedOperation,
            "logical.family.mismatch",
        );
    }
    if query.inputs.len() != operation.arity() || query.outputs.len() > 1 {
        return reject(ProviderRejectionCode::InvalidShape, "logical.operand_count");
    }
    if query
        .inputs
        .iter()
        .chain(&query.outputs)
        .any(|value| value.storage != ProviderStorage::DenseReal)
    {
        return reject(
            ProviderRejectionCode::UnsupportedStorage,
            "logical.real_storage.required",
        );
    }
    #[cfg(feature = "wgpu")]
    let float = ProviderElementType::from(_precision);
    let floating = match profile {
        HookProfile::InProcess => query.inputs.iter().all(|value| {
            matches!(
                value.element_type,
                ProviderElementType::F32 | ProviderElementType::F64 | ProviderElementType::Logical
            )
        }),
        #[cfg(feature = "wgpu")]
        HookProfile::Wgpu => query.inputs.iter().all(|value| value.element_type == float),
    };
    #[cfg(feature = "wgpu")]
    let integer = matches!(profile, HookProfile::Wgpu)
        && matches!(operation, Operation::Compare(_))
        && query.inputs[0].element_type == query.inputs[1].element_type
        && matches!(
            query.inputs[0].element_type,
            ProviderElementType::I8
                | ProviderElementType::I16
                | ProviderElementType::I32
                | ProviderElementType::I64
                | ProviderElementType::U8
                | ProviderElementType::U16
                | ProviderElementType::U32
                | ProviderElementType::U64
        );
    #[cfg(not(feature = "wgpu"))]
    let integer = false;
    if !floating && !integer {
        // Logical class alone does not prove physical floating storage.
        return reject(
            ProviderRejectionCode::UnsupportedElementType,
            "logical.input_type.unsupported",
        );
    }
    if query
        .inputs
        .iter()
        .chain(&query.outputs)
        .any(|value| value.layout != ProviderLayout::ColumnMajorContiguous)
    {
        return reject(
            ProviderRejectionCode::UnsupportedLayout,
            "logical.layout.contiguous_required",
        );
    }
    if query.outputs.iter().any(|value| match profile {
        HookProfile::InProcess => value.element_type != ProviderElementType::Logical,
        #[cfg(feature = "wgpu")]
        HookProfile::Wgpu => {
            value.element_type != float && value.element_type != ProviderElementType::Logical
        }
    }) {
        return reject(
            ProviderRejectionCode::UnsupportedElementType,
            "logical.output_type.unsupported",
        );
    }
    let output_shape = if query.inputs.len() == 1 {
        query.inputs[0].shape.clone()
    } else {
        let left = &query.inputs[0].shape;
        let right = &query.inputs[1].shape;
        match profile {
            HookProfile::InProcess => {}
            #[cfg(feature = "wgpu")]
            HookProfile::Wgpu if integer => {
                if left.len().max(right.len()) > crate::backend::wgpu::params::BCAST_MAX_RANK {
                    return reject(
                        ProviderRejectionCode::UnsupportedRank,
                        "logical.integer_broadcast.rank",
                    );
                }
            }
            #[cfg(feature = "wgpu")]
            HookProfile::Wgpu if left != right => {
                return reject(
                    ProviderRejectionCode::InvalidShape,
                    "logical.floating_shapes.must_match",
                );
            }
            #[cfg(feature = "wgpu")]
            HookProfile::Wgpu => {}
        }
        let convert = |shape: &[u64]| {
            shape
                .iter()
                .copied()
                .map(usize::try_from)
                .collect::<Result<Vec<_>, _>>()
        };
        let (Ok(left), Ok(right)) = (convert(left), convert(right)) else {
            return reject(
                ProviderRejectionCode::InvalidShape,
                "logical.broadcast.dimension_overflow",
            );
        };
        let Ok(shape) = runmat_types::geometry::broadcast_concrete_shape(&left, &right) else {
            return reject(
                ProviderRejectionCode::InvalidShape,
                "logical.broadcast.invalid",
            );
        };
        let Ok(shape) = shape
            .into_iter()
            .map(u64::try_from)
            .collect::<Result<Vec<_>, _>>()
        else {
            return reject(
                ProviderRejectionCode::InvalidShape,
                "logical.broadcast.dimension_overflow",
            );
        };
        shape
    };
    if query
        .outputs
        .iter()
        .any(|value| value.shape != output_shape)
    {
        return reject(
            ProviderRejectionCode::InvalidShape,
            "logical.output_shape.mismatch",
        );
    }
    let elements = if output_shape.contains(&0) {
        Some(0)
    } else {
        output_shape
            .iter()
            .try_fold(1_u64, |total, extent| total.checked_mul(*extent))
    };
    let Some(elements) = elements else {
        return reject(
            ProviderRejectionCode::InvalidShape,
            "logical.output_shape.overflow",
        );
    };
    #[cfg(feature = "wgpu")]
    if matches!(profile, HookProfile::Wgpu) && elements > u64::from(u32::MAX) {
        return reject(
            ProviderRejectionCode::ResourceLimit,
            "logical.dispatch.length",
        );
    }
    if matches!(profile, HookProfile::InProcess) {
        // InProcess logical producers own real F64 payloads even when their
        // public class is logical. Representation alone cannot bound all live
        // inputs, so only the distinct output allocation is estimated here.
        let Some(output_bytes) = elements.checked_mul(ProviderElementType::F64.byte_width()) else {
            return reject(
                ProviderRejectionCode::ResourceLimit,
                "logical.output_bytes.overflow",
            );
        };
        return Some(ProviderFeasibility::supported(ProviderResourceEstimate {
            transient_bytes: None,
            output_bytes: Some(output_bytes),
            dispatches: Some(u32::from(elements != 0)),
        }));
    }
    None
}
