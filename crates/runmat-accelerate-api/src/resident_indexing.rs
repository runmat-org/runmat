/// Logical RHS element mapping at the existing functional scatter boundary.
/// Storage lane expansion belongs to the provider, after this mapping.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProviderScatterValueMapping {
    Linear,
    Scalar,
}
