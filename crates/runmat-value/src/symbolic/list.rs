//! Published symbolic child lists retain the one original collection owner.
use crate::{ExactArrayStorage, OwnedExactArrayValues};
#[derive(Debug, PartialEq)]
pub struct SymbolicList<T>(ExactArrayStorage<T>);
impl<T> From<Vec<T>> for SymbolicList<T> {
    fn from(values: Vec<T>) -> Self {
        Self(if values.capacity() == 0 {
            ExactArrayStorage::empty()
        } else {
            ExactArrayStorage::shared_unbounded(values)
        })
    }
}
impl<T> From<ExactArrayStorage<T>> for SymbolicList<T> {
    fn from(values: ExactArrayStorage<T>) -> Self {
        if values.is_shallow_clone() {
            return Self(values);
        }
        let values = values
            .try_into_owned()
            .unwrap_or_else(|_| unreachable!("unpublished symbolic list is unique"))
            .try_into_unbounded_vec()
            .unwrap_or_else(|_| unreachable!("unpublished symbolic list has no claim"));
        values.into()
    }
}
impl<T: Clone> Clone for SymbolicList<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl<T> SymbolicList<T> {
    pub fn storage(&self) -> &ExactArrayStorage<T> {
        &self.0
    }
    pub fn into_storage(self) -> ExactArrayStorage<T> {
        self.0
    }
    pub fn try_into_owned(self) -> Result<OwnedExactArrayValues<T>, Self> {
        self.0.try_into_owned().map_err(Self)
    }
}
impl<T> std::ops::Deref for SymbolicList<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        &self.0
    }
}
impl<'a, T> IntoIterator for &'a SymbolicList<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}
impl<T> FromIterator<T> for SymbolicList<T> {
    fn from_iter<I: IntoIterator<Item = T>>(values: I) -> Self {
        values.into_iter().collect::<Vec<_>>().into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn publication_moves_unpublished_slots_and_aliases_original_allocation() {
        let values = vec![String::from("parameter")];
        let pointer = values.as_ptr();
        let list = SymbolicList::from(ExactArrayStorage::unbounded(values));
        assert_eq!(list.as_ptr(), pointer);
        let alias = list.clone();
        assert_eq!(alias.as_ptr(), pointer);
        assert!(list.try_into_owned().is_err());
        assert_eq!(alias[0], "parameter");
    }
}
