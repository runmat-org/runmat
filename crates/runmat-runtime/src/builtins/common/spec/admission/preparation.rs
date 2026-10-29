//! Selected operation-owned bounds for admission itself, not a pure-operation
//! classification or permission to change numerical/domain dispatch.
use super::{FusionAdmissionContext, FusionAdmissionFunction, FusionAdmissionFuture};
use runmat_value::Value;

pub struct FusionAdmissionPreparationQuery<'a> {
    /// Authentic request leaves/constants; computed operands stay absent.
    pub operands: &'a [Option<&'a Value>],
    pub max_rank: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FusionAdmissionResources {
    pub scratch_bytes: u64,
    pub retained_bytes: u64,
}

type Resources =
    for<'a> fn(FusionAdmissionPreparationQuery<'a>) -> Option<FusionAdmissionResources>;

/// The selected declaration owns both its unchanged invocation and any proven
/// finite preparation recipe. Unknown admissions have no finite bound.
#[derive(Clone, Copy)]
pub struct FusionAdmission {
    invoke: FusionAdmissionFunction,
    preparation: Option<(usize, Resources)>,
}

impl FusionAdmission {
    pub const fn unbounded(invoke: FusionAdmissionFunction) -> Self {
        Self {
            invoke,
            preparation: None,
        }
    }

    pub const fn with_preparation(
        invoke: FusionAdmissionFunction,
        arity: usize,
        resources: Resources,
    ) -> Self {
        Self {
            invoke,
            preparation: Some((arity, resources)),
        }
    }

    pub fn call<'a>(self, context: FusionAdmissionContext<'a>) -> FusionAdmissionFuture<'a> {
        (self.invoke)(context)
    }

    pub fn preparation_arity(self) -> Option<usize> {
        self.preparation.map(|(arity, _)| arity)
    }

    pub fn preparation_resources(
        self,
        query: FusionAdmissionPreparationQuery<'_>,
    ) -> Option<FusionAdmissionResources> {
        let (arity, resources) = self.preparation?;
        if query.operands.len() != arity {
            return None;
        }
        resources(query)
    }
}
