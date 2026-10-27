//! Exact two-output projection transfers registrations, never retraces a child.
use super::{iteration::RootedValuesIteration, RootedResult, RootedValue, ValueSequence};
impl RootedResult<ValueSequence> {
    pub(crate) fn try_into_pair(self) -> Result<(RootedValue, RootedValue), crate::RuntimeError> {
        if self.value.len() != 2 {
            return Err(crate::runtime_error::semantic_error(
                "RunMat:Call:OutputCount",
                format!(
                    "this call requires two values, but returned {}",
                    self.value.len()
                ),
            ));
        }
        let Self { value, _roots } = self;
        let mut values = RootedValuesIteration::from_sequence(value);
        let first = values.next().expect("checked first result");
        let second = values.next().expect("checked second result");
        drop(values);
        Ok((
            RootedResult {
                value: first,
                _roots: _roots.clone(),
            },
            RootedResult {
                value: second,
                _roots,
            },
        ))
    }
}
