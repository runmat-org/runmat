//! Compact original-row dependence, independent of rounded coefficients.
//! All/none are allocation-free. Mixed arrays retain only dependent row IDs.
use super::*;
#[derive(Clone, PartialEq, Eq)]
pub(in crate::builtins::math::optim::optimmodel::lowering) enum Degree {
    Constant,
    Affine,
    Rows(std::rc::Rc<Vec<usize>>),
}
pub(in crate::builtins::math::optim::optimmodel::lowering) enum DependentRows<'a> {
    None,
    All(std::ops::Range<usize>),
    Rows(std::slice::Iter<'a, usize>),
}
impl Iterator for DependentRows<'_> {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        match self {
            Self::None => None,
            Self::All(rows) => rows.next(),
            Self::Rows(rows) => rows.next().copied(),
        }
    }
}
impl Degree {
    pub(in crate::builtins::math::optim::optimmodel::lowering) fn rows(
        &self,
        count: usize,
    ) -> DependentRows<'_> {
        match self {
            Self::Constant => DependentRows::None,
            Self::Affine => DependentRows::All(0..count),
            Self::Rows(rows) => DependentRows::Rows(rows.iter()),
        }
    }
    pub(in crate::builtins::math::optim::optimmodel::lowering) fn at(&self, row: usize) -> bool {
        match self {
            Self::Constant => false,
            Self::Affine => true,
            Self::Rows(rows) => rows.binary_search(&row).is_ok(),
        }
    }
    pub(in crate::builtins::math::optim::optimmodel::lowering) async fn collect(
        count: usize,
        mut depends: impl FnMut(usize) -> BuiltinResult<bool>,
        policy: Policy,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let mut rows = Vec::new();
        let mut all = true;
        for row in 0..count {
            if depends(row)? {
                if !all {
                    rows.try_reserve(1).map_err(|e| policy.reserve(e))?;
                    rows.push(row);
                }
            } else if all {
                all = false;
                rows.try_reserve(row).map_err(|e| policy.reserve(e))?;
                for preceding in 0..row {
                    rows.push(preceding);
                    c.completed(1).await?;
                }
            }
            c.completed(1).await?;
        }
        Ok(if count == 0 || (!all && rows.is_empty()) {
            Self::Constant
        } else if all {
            Self::Affine
        } else {
            Self::Rows(std::rc::Rc::new(rows))
        })
    }
}
