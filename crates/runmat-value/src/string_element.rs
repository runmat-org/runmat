//! One language string element: immutable text or an explicit missing value.
//!
//! Missing has no UTF-8 allocation. Text retains the existing physical owner;
//! aliases and exact promotion never create a second byte representation.
use std::fmt;

use crate::{
    ExactHostAllocationPermit, ExactHostNumericAllocationError, Utf8Text, Utf8TextAdmissionError,
};

/// Text sorts before missing in the storage's total ordering. Public language
/// comparison and missing-data policies are owned by their operations.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StringElement {
    Text(Utf8Text),
    Missing,
}

impl Default for StringElement {
    fn default() -> Self {
        Self::Text(Utf8Text::default())
    }
}

#[derive(Debug)]
pub struct StringElementAdmissionError(Utf8TextAdmissionError);

impl StringElementAdmissionError {
    pub fn source(&self) -> ExactHostNumericAllocationError {
        self.0.source()
    }

    /// Recovery does not detach a claim added through an alias.
    pub fn try_into_unclaimed_element(self) -> Result<StringElement, Self> {
        self.0
            .try_into_unclaimed_text()
            .map(StringElement::Text)
            .map_err(Self)
    }
}

impl fmt::Display for StringElementAdmissionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl std::error::Error for StringElementAdmissionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.0)
    }
}

impl StringElement {
    /// Every text constructor is literal, including the text `<missing>`.
    pub fn text(text: impl Into<Utf8Text>) -> Self {
        Self::Text(text.into())
    }

    pub const fn missing() -> Self {
        Self::Missing
    }

    pub fn is_missing(&self) -> bool {
        matches!(self, Self::Missing)
    }

    pub fn as_str(&self) -> Option<&str> {
        self.text_ref().map(Utf8Text::as_str)
    }

    pub fn text_ref(&self) -> Option<&Utf8Text> {
        match self {
            Self::Text(text) => Some(text),
            Self::Missing => None,
        }
    }

    pub fn into_text(self) -> Option<Utf8Text> {
        match self {
            Self::Text(text) => Some(text),
            Self::Missing => None,
        }
    }

    /// Explicit rendering view. Semantic conversions must not use display
    /// spelling to reconstruct an element's presence.
    pub fn display(&self) -> StringElementDisplay<'_> {
        StringElementDisplay(self)
    }

    pub fn capacity(&self) -> usize {
        self.text_ref().map_or(0, Utf8Text::capacity)
    }

    pub fn has_retained_claim(&self) -> bool {
        self.text_ref().is_some_and(Utf8Text::has_retained_claim)
    }

    pub fn shared_storage_identity(
        &self,
    ) -> Option<crate::exact_array_storage::SharedStorageIdentity> {
        self.text_ref().and_then(Utf8Text::shared_storage_identity)
    }

    pub fn unclaimed_bytes(&self) -> Result<Option<usize>, ExactHostNumericAllocationError> {
        match self {
            Self::Text(text) => text.unclaimed_bytes(),
            Self::Missing => Ok(None),
        }
    }

    pub fn requested_bytes_for_capacity(
        capacity: usize,
    ) -> Result<usize, ExactHostNumericAllocationError> {
        Utf8Text::requested_bytes_for_capacity(capacity)
    }

    pub fn try_adopt_string_exact(
        text: String,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self, crate::ExactArrayAdoptionError<u8>> {
        Utf8Text::try_adopt_string_exact(text, permit).map(Self::Text)
    }

    pub fn try_admit_exact(
        self,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self, StringElementAdmissionError> {
        match self {
            Self::Text(text) => text
                .try_admit_exact(permit)
                .map(Self::Text)
                .map_err(StringElementAdmissionError),
            Self::Missing => Ok(Self::Missing),
        }
    }

    pub fn try_copy_exact(
        &self,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self, ExactHostNumericAllocationError> {
        match self {
            Self::Text(text) => text.try_copy_exact(permit).map(Self::Text),
            Self::Missing => Ok(Self::Missing),
        }
    }

    pub fn try_copy_exact_with<E>(
        &self,
        permit: &dyn ExactHostAllocationPermit,
        mut checkpoint: impl FnMut(usize) -> Result<(), E>,
    ) -> Result<Self, crate::ExactArrayInitializationError<E>> {
        match self {
            Self::Text(text) => text.try_copy_exact_with(permit, checkpoint).map(Self::Text),
            Self::Missing => {
                checkpoint(0).map_err(crate::ExactArrayInitializationError::Element)?;
                Ok(Self::Missing)
            }
        }
    }
}

pub struct StringElementDisplay<'a>(&'a StringElement);

impl fmt::Display for StringElementDisplay<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            StringElement::Text(text) => formatter.write_str(text.as_str()),
            StringElement::Missing => formatter.write_str("<missing>"),
        }
    }
}

impl From<String> for StringElement {
    fn from(text: String) -> Self {
        Self::Text(text.into())
    }
}
impl From<&str> for StringElement {
    fn from(text: &str) -> Self {
        Self::Text(text.into())
    }
}
impl From<Utf8Text> for StringElement {
    fn from(text: Utf8Text) -> Self {
        Self::Text(text)
    }
}
impl From<&Utf8Text> for StringElement {
    fn from(text: &Utf8Text) -> Self {
        Self::Text(text.clone())
    }
}
impl From<&StringElement> for StringElement {
    fn from(element: &StringElement) -> Self {
        element.clone()
    }
}
impl From<&String> for StringElement {
    fn from(text: &String) -> Self {
        Self::from(text.as_str())
    }
}

impl PartialEq<str> for StringElement {
    fn eq(&self, other: &str) -> bool {
        self.as_str() == Some(other)
    }
}
impl PartialEq<&str> for StringElement {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == Some(*other)
    }
}
impl PartialEq<String> for StringElement {
    fn eq(&self, other: &String) -> bool {
        self.as_str() == Some(other.as_str())
    }
}
impl PartialEq<StringElement> for String {
    fn eq(&self, other: &StringElement) -> bool {
        other.as_str() == Some(self.as_str())
    }
}

impl serde::Serialize for StringElement {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Text(text) => serializer.serialize_str(text.as_str()),
            Self::Missing => serializer.serialize_none(),
        }
    }
}
impl<'de> serde::Deserialize<'de> for StringElement {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let text = <Option<String> as serde::Deserialize>::deserialize(deserializer)?;
        Ok(text.map_or(Self::Missing, Self::from))
    }
}
