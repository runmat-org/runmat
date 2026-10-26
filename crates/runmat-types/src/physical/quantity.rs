use super::{AdditiveQuantity, QuantityKind, QuantityMarker, QuantitySpec};
use serde::{
    de::{Error, MapAccess, Visitor},
    ser::SerializeMap,
    Deserialize, Deserializer, Serialize, Serializer,
};
use std::{
    fmt,
    marker::PhantomData,
    ops::{Add, Div, Mul, Neg, Sub},
};

/// Canonical physical value with a semantic kind and owned storage. Storage is
/// private: numeric kernels must explicitly project canonical values.
///
/// The canonical field wire name remains stable (e.g. `amperes`) so this type
/// migration does not silently reinterpret persisted analysis-model quantities.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quantity<K: QuantitySpec, S = f64> {
    pub(super) value: S,
    pub(super) spec: K,
}

impl<K: QuantityMarker, S: Default> Default for Quantity<K, S> {
    fn default() -> Self {
        Self::new(S::default())
    }
}

pub trait FiniteQuantity {
    fn is_finite(&self) -> bool;
}
impl<Q: FiniteQuantity, const N: usize> FiniteQuantity for [Q; N] {
    fn is_finite(&self) -> bool {
        self.iter().all(FiniteQuantity::is_finite)
    }
}
pub trait CanonicalQuantity: FiniteQuantity + Copy {
    fn from_canonical_value(value: f64) -> Self;
    fn canonical_value(self) -> f64;
}
pub trait QuantityStorage {
    fn all_finite(&self) -> bool;
}
impl QuantityStorage for f64 {
    fn all_finite(&self) -> bool {
        self.is_finite()
    }
}
impl<S: QuantityStorage, const N: usize> QuantityStorage for [S; N] {
    fn all_finite(&self) -> bool {
        self.as_slice().all_finite()
    }
}
impl<S: QuantityStorage> QuantityStorage for [S] {
    fn all_finite(&self) -> bool {
        self.iter().all(QuantityStorage::all_finite)
    }
}
impl<S: QuantityStorage> QuantityStorage for Vec<S> {
    fn all_finite(&self) -> bool {
        self.as_slice().all_finite()
    }
}
impl<S: QuantityStorage + ?Sized> QuantityStorage for &S {
    fn all_finite(&self) -> bool {
        (*self).all_finite()
    }
}

impl<K: QuantitySpec, S> Quantity<K, S> {
    pub fn dimension(&self) -> crate::PhysicalDimension {
        self.spec.physical_dimension()
    }
    pub fn semantic_kind(&self) -> Option<QuantityKind> {
        self.spec.semantic_kind()
    }
    pub fn angle_power(&self) -> i8 {
        self.spec.nominal_angle_power()
    }
    pub fn descriptor(&self) -> K {
        self.spec
    }
    pub fn canonical_values(&self) -> &S {
        &self.value
    }
    /// Run an explicit numerical storage transform while retaining the exact
    /// static/dynamic descriptor. Nonfinite output never enters the new quantity.
    pub fn try_map_canonical_storage<T: QuantityStorage>(
        self,
        numerical: impl FnOnce(S) -> T,
    ) -> Result<Quantity<K, T>, QuantityValueError> {
        let value = numerical(self.value);
        if !value.all_finite() {
            return Err(QuantityValueError::NonFinite);
        }
        Ok(Quantity {
            value,
            spec: self.spec,
        })
    }
    pub fn into_canonical_values(self) -> S {
        self.value
    }
}
impl<K: QuantityMarker, S> Quantity<K, S> {
    /// Construct canonical storage inside a reviewed producer. External values use
    /// checked constructors or validating serde; this also admits rejection fixtures.
    pub const fn new(value: S) -> Self {
        Self {
            value,
            spec: K::MARKER,
        }
    }
    pub const fn kind(&self) -> QuantityKind {
        K::KIND
    }
}
impl<K: QuantitySpec, S: Copy> Quantity<K, S> {
    pub const fn canonical_value(self) -> S {
        self.value
    }
}
impl<K: QuantityMarker, S: QuantityStorage> Quantity<K, S> {
    pub fn try_new(value: S) -> Result<Self, QuantityValueError> {
        if !value.all_finite() {
            return Err(QuantityValueError::NonFinite);
        }
        Ok(Self::new(value))
    }
}
impl<K: QuantitySpec, S: QuantityStorage + Copy> Quantity<K, S> {
    pub fn is_finite(self) -> bool {
        self.value.all_finite()
    }
}
impl<K: QuantitySpec, S: QuantityStorage> FiniteQuantity for Quantity<K, S> {
    fn is_finite(&self) -> bool {
        self.value.all_finite()
    }
}
impl<K: QuantityMarker> CanonicalQuantity for Quantity<K> {
    fn from_canonical_value(value: f64) -> Self {
        Self::new(value)
    }
    fn canonical_value(self) -> f64 {
        self.value
    }
}
// Ordering is admitted only for the same statically known semantic kind.
// Dynamic descriptors require an explicit compatibility check first.
impl<K: QuantityMarker, S: PartialOrd> PartialOrd for Quantity<K, S> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl<K: AdditiveQuantity> Add for Quantity<K> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.value + rhs.value)
    }
}
impl<K: AdditiveQuantity> Sub for Quantity<K> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.value - rhs.value)
    }
}
impl<K: AdditiveQuantity> Neg for Quantity<K> {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.value)
    }
}
impl<K: AdditiveQuantity> Mul<f64> for Quantity<K> {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self::new(self.value * rhs)
    }
}
impl<K: AdditiveQuantity> Div<f64> for Quantity<K> {
    type Output = Self;
    fn div(self, rhs: f64) -> Self {
        Self::new(self.value / rhs)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum QuantityValueError {
    #[error("physical quantity must contain finite canonical values")]
    NonFinite,
}
impl<K: QuantityMarker, S: Serialize + QuantityStorage> Serialize for Quantity<K, S> {
    fn serialize<T: Serializer>(&self, s: T) -> Result<T::Ok, T::Error> {
        if !self.value.all_finite() {
            return Err(serde::ser::Error::custom(QuantityValueError::NonFinite));
        }
        let mut map = s.serialize_map(Some(1))?;
        map.serialize_entry(K::CANONICAL_FIELD, &self.value)?;
        map.end()
    }
}
impl<'de, K: QuantityMarker, S: Deserialize<'de> + QuantityStorage> Deserialize<'de>
    for Quantity<K, S>
{
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct QuantityVisitor<K, S>(PhantomData<(K, S)>);
        impl<'de, K: QuantityMarker, S: Deserialize<'de> + QuantityStorage> Visitor<'de>
            for QuantityVisitor<K, S>
        {
            type Value = Quantity<K, S>;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "a finite {} quantity", K::CANONICAL_FIELD)
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut value = None;
                while let Some(key) = map.next_key::<String>()? {
                    if key != K::CANONICAL_FIELD {
                        return Err(A::Error::custom(format!(
                            "unexpected quantity field `{key}`; expected `{}`",
                            K::CANONICAL_FIELD
                        )));
                    }
                    if value.is_some() {
                        return Err(A::Error::duplicate_field(K::CANONICAL_FIELD));
                    }
                    value = Some(map.next_value::<S>()?);
                }
                Quantity::try_new(value.ok_or_else(|| A::Error::missing_field(K::CANONICAL_FIELD))?)
                    .map_err(A::Error::custom)
            }
        }
        d.deserialize_map(QuantityVisitor::<K, S>(PhantomData))
    }
}
#[cfg(feature = "tool-schema")]
impl<K: QuantityMarker, S: schemars::JsonSchema> schemars::JsonSchema for Quantity<K, S> {
    fn schema_name() -> String {
        format!("{:?}Quantity{}", K::KIND, S::schema_name())
    }
    fn json_schema(g: &mut schemars::gen::SchemaGenerator) -> schemars::schema::Schema {
        use schemars::schema::*;
        let mut object = ObjectValidation::default();
        object.required.insert(K::CANONICAL_FIELD.into());
        object
            .properties
            .insert(K::CANONICAL_FIELD.into(), g.subschema_for::<S>());
        object.additional_properties = Some(Box::new(Schema::Bool(false)));
        Schema::Object(SchemaObject {
            instance_type: Some(InstanceType::Object.into()),
            object: Some(Box::new(object)),
            ..Default::default()
        })
    }
}

/// These examples deliberately fail compilation: same storage and even equal
/// dimensions do not make different physical kinds interchangeable.
/// ```compile_fail
/// use runmat_types::physical::{ElectricCurrent, ElectricPotential};
/// fn current(_:ElectricCurrent){}
/// current(ElectricPotential::new(12.0));
/// ```
/// ```compile_fail
/// use runmat_types::physical::{Moment, Energy};
/// let moment:Moment=Energy::new(2.0);
/// ```
/// ```compile_fail
/// use runmat_types::physical::Temperature;
/// let invalid=Temperature::new(300.0)+Temperature::new(300.0);
/// ```
/// ```compile_fail
/// use runmat_types::physical::{CoordinateUnits,KernelCoordinates,PreviewCoordinates,LengthUnit};
/// fn exact(_:CoordinateUnits<KernelCoordinates>){}
/// exact(CoordinateUnits::<PreviewCoordinates>::new(LengthUnit::Millimeter));
/// ```
pub struct QuantityTypeContracts;

impl<S: QuantityStorage> QuantityStorage for Option<S> {
    fn all_finite(&self) -> bool {
        self.as_ref().is_none_or(QuantityStorage::all_finite)
    }
}
impl<K: QuantityMarker> Quantity<K> {
    /// Ordering retains the known semantic role, including absolute temperature.
    pub fn min(self, other: Self) -> Self {
        Self::new(self.value.min(other.value))
    }
    pub fn max(self, other: Self) -> Self {
        Self::new(self.value.max(other.value))
    }
}
impl<K: AdditiveQuantity> Quantity<K> {
    pub fn abs(self) -> Self {
        Self::new(self.value.abs())
    }
}
impl<K: AdditiveQuantity> Mul<Quantity<K>> for f64 {
    type Output = Quantity<K>;
    fn mul(self, rhs: Quantity<K>) -> Self::Output {
        rhs * self
    }
}

impl<K: AdditiveQuantity> std::ops::AddAssign for Quantity<K> {
    fn add_assign(&mut self, rhs: Self) {
        self.value += rhs.value;
    }
}
impl<K: AdditiveQuantity> std::ops::SubAssign for Quantity<K> {
    fn sub_assign(&mut self, rhs: Self) {
        self.value -= rhs.value;
    }
}
impl<K: QuantityMarker, T: QuantityStorage> Quantity<K, Vec<T>> {
    /// Append a finite value of the same static kind without exposing mutable
    /// raw storage or rescanning/copying an already admitted history buffer.
    pub fn try_push(&mut self, value: Quantity<K, T>) -> Result<(), QuantityValueError> {
        if !value.value.all_finite() {
            return Err(QuantityValueError::NonFinite);
        }
        self.value.push(value.value);
        Ok(())
    }
}

/// Random access to an interleaved scalar coordinate buffer, without allocating
/// a duplicate dense array merely to attach its uniform quantity descriptor.
pub trait ScalarQuantityStorage: QuantityStorage {
    fn scalar_count(&self) -> usize;
    fn scalar_value(&self, index: usize) -> Option<f64>;
}
impl ScalarQuantityStorage for [f64] {
    fn scalar_count(&self) -> usize {
        self.len()
    }
    fn scalar_value(&self, index: usize) -> Option<f64> {
        self.get(index).copied()
    }
}
impl ScalarQuantityStorage for Vec<f64> {
    fn scalar_count(&self) -> usize {
        self.len()
    }
    fn scalar_value(&self, index: usize) -> Option<f64> {
        self.get(index).copied()
    }
}
impl<S: ScalarQuantityStorage + ?Sized> ScalarQuantityStorage for &S {
    fn scalar_count(&self) -> usize {
        (**self).scalar_count()
    }
    fn scalar_value(&self, index: usize) -> Option<f64> {
        (**self).scalar_value(index)
    }
}
impl<K: QuantitySpec, S: ScalarQuantityStorage> Quantity<K, S> {
    pub fn scalar_count(&self) -> usize {
        self.value.scalar_count()
    }
    pub fn scalar_at(&self, index: usize) -> Option<Quantity<K>> {
        let value = self.value.scalar_value(index)?;
        value.is_finite().then_some(Quantity {
            value,
            spec: self.spec,
        })
    }
}

impl<K: QuantityMarker, T: QuantityStorage + Clone> Quantity<K, Vec<T>> {
    /// Resize a uniformly typed series with a finite value of the same kind.
    /// The owner's separate topology/cardinality validation still applies.
    pub fn try_resize(
        &mut self,
        length: usize,
        fill: Quantity<K, T>,
    ) -> Result<(), QuantityValueError> {
        if !fill.value.all_finite() {
            return Err(QuantityValueError::NonFinite);
        }
        self.value.resize(length, fill.value);
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum QuantityVectorEditError {
    #[error("physical quantity replacement must contain finite values")]
    NonFinite,
    #[error("physical quantity vector index is out of bounds")]
    IndexOutOfBounds,
    #[error("physical quantity vector descriptor differs from its replacement")]
    DescriptorMismatch,
}
impl<K: QuantitySpec + PartialEq, T: QuantityStorage> Quantity<K, Vec<T>> {
    /// Replace one finite, identically described component without releasing a
    /// mutable raw buffer or rescanning the admitted vector.
    pub fn try_replace(
        &mut self,
        index: usize,
        value: Quantity<K, T>,
    ) -> Result<(), QuantityVectorEditError> {
        if value.spec != self.spec {
            return Err(QuantityVectorEditError::DescriptorMismatch);
        }
        if !value.value.all_finite() {
            return Err(QuantityVectorEditError::NonFinite);
        }
        let destination = self
            .value
            .get_mut(index)
            .ok_or(QuantityVectorEditError::IndexOutOfBounds)?;
        *destination = value.value;
        Ok(())
    }
}

impl<K: QuantitySpec, T> Quantity<K, Vec<T>> {
    /// Borrow a uniform physical buffer while retaining its descriptor. Forwarding
    /// between physical owners does not release a raw slice or copy the buffer.
    pub fn as_quantity_slice(&self) -> Quantity<K, &[T]> {
        Quantity {
            spec: self.spec,
            value: self.value.as_slice(),
        }
    }
}
impl<K: QuantitySpec, T, const N: usize> Quantity<K, [T; N]> {
    pub fn as_quantity_slice(&self) -> Quantity<K, &[T]> {
        Quantity {
            spec: self.spec,
            value: self.value.as_slice(),
        }
    }
}
