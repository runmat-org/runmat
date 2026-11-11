//! STEP (ISO 10303) beyond what the geometry kernel writes: the Part 21 exchange structure
//! ([`p21`]) and AP242 semantic PMI ([`pmi`]), written into and read from the kernel's AP242
//! files. Clean-room, from ISO 10303-21, the AP242 schema (`ap242_managed_model_based_3d_engineering`)
//! and the CAx-IF Recommended Practices for the Representation and Presentation of PMI (AP242).

pub mod p21;
pub mod pmi;
