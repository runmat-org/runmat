//! A unique owner moves slots; aliases project one canonical Value at a time.
//! Both routes retain the original physical allocation and its claims. There
//! is no replacement Vec, iterator Box, or second root registration.
use runmat_value::{
    OutputListIntoIter, OutputListStorage, Value, ValueSequence, ValueSequenceStorage,
};
#[derive(Debug)]
pub(crate) enum RootedValuesIteration {
    Single(Option<Value>),
    Unique(OutputListIntoIter),
    Retained {
        values: OutputListStorage,
        front: usize,
        back: usize,
    },
}
impl RootedValuesIteration {
    /// The caller retains the original sequence's root lease. Only physical
    /// storage moves here; an inline single value needs no collection.
    pub(crate) fn from_sequence(values: ValueSequence) -> Self {
        match values.into_storage() {
            ValueSequenceStorage::CommaSeparated(values) => Self::new(values),
            ValueSequenceStorage::Single(value) => Self::Single(Some(value)),
        }
    }

    pub(super) fn new(values: OutputListStorage) -> Self {
        match values.try_into_owned() {
            Ok(values) => Self::Unique(values.into_iter()),
            Err(values) => {
                let back = values.len();
                Self::Retained {
                    values,
                    front: 0,
                    back,
                }
            }
        }
    }
}
impl Iterator for RootedValuesIteration {
    type Item = Value;
    fn next(&mut self) -> Option<Value> {
        match self {
            Self::Single(value) => value.take(),
            Self::Unique(values) => values.next(),
            Self::Retained {
                values,
                front,
                back,
            } => {
                if front == back {
                    return None;
                }
                let index = *front;
                *front += 1;
                Some(values[index].clone())
            }
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = match self {
            Self::Single(value) => usize::from(value.is_some()),
            Self::Unique(values) => values.len(),
            Self::Retained { front, back, .. } => back - front,
        };
        (n, Some(n))
    }
}
impl DoubleEndedIterator for RootedValuesIteration {
    fn next_back(&mut self) -> Option<Value> {
        match self {
            Self::Single(value) => value.take(),
            Self::Unique(values) => values.next_back(),
            Self::Retained {
                values,
                front,
                back,
            } => {
                if front == back {
                    return None;
                }
                *back -= 1;
                Some(values[*back].clone())
            }
        }
    }
}
impl ExactSizeIterator for RootedValuesIteration {}
impl std::iter::FusedIterator for RootedValuesIteration {}
