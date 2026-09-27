//! Shared matrix-product evaluation for scalar and matrix/page consumers.
//! Public identities own admission and error descriptors; numerical/provider
//! semantics and temporary ownership have one implementation here.
use runmat_value::Value;
mod errors;
mod host;
mod integer;
mod provider;
pub(crate) use errors::ProductError;
type ProductResult<T> = Result<T, ProductError>;
// Mathematical operation label retained for established helper error messages.
const NAME: &str = "mtimes";
pub(crate) async fn evaluate(lhs: &Value, rhs: &Value) -> ProductResult<Value> {
    if integer::contains(lhs) || integer::contains(rhs) {
        return integer::evaluate(lhs, rhs).await;
    }
    if let Some(result) = provider::try_product(lhs, rhs).await? {
        return Ok(result);
    }
    host::evaluate(lhs.clone(), rhs.clone()).await
}
