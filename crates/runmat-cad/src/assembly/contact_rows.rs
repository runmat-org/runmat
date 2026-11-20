//! The top assembly's contact regions as Outline rows (`contact:<name>`): the two components,
//! the faces their parts publish, whether the region is bonded, and what is wrong with it.

use crate::assembly::model::{AssemblyDocument, ContactFace};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContactRow {
    pub name: String,
    /// The components, by path, and the faces their parts publish.
    pub a: String,
    pub a_face: String,
    pub b: String,
    pub b_face: String,
    pub bonded: bool,
    /// What is wrong with it: a component is gone.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub problem: Option<String>,
}

/// One row per contact region the top assembly declares, in document order.
pub fn contact_rows(top: Option<&AssemblyDocument>) -> Vec<ContactRow> {
    let Some(doc) = top else {
        return Vec::new();
    };
    let missing = |side: &ContactFace| {
        let first = side.occurrence.0.first()?;
        (!doc.occurrences.iter().any(|o| o.name == first.name))
            .then(|| format!("its component `{}` isn't in the assembly", side.occurrence))
    };
    doc.contacts
        .iter()
        .map(|c| ContactRow {
            name: c.name.clone(),
            a: c.a.occurrence.to_string(),
            a_face: c.a.face.as_str().to_owned(),
            b: c.b.occurrence.to_string(),
            b_face: c.b.face.as_str().to_owned(),
            bonded: c.bonded,
            problem: missing(&c.a).or_else(|| missing(&c.b)),
        })
        .collect()
}
