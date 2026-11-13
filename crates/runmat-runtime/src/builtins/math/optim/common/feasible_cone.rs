//! Feasible polyhedral generators with explicit lineality and pointed rays.
//! A cut through lineality needs one pivot, not all +/-coordinate pairs.
//! Pointed cuts use exact facet-incidence adjacency before adding boundary rays.
//! This avoids an artificial O(n^2) ray explosion for one active halfspace.
mod independent_axes;
pub(in crate::builtins::math::optim) use independent_axes::IndependentAxes;
mod normal;
mod policy;
pub(in crate::builtins::math::optim) use normal::{Diagonal, Normal};
pub(in crate::builtins::math::optim) use policy::Policy;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::builtins::math::optim) enum Basis {
    Maximal,
    Minimal,
}
use crate::builtins::math::optim::common::ScalarCooperation;
use crate::BuiltinResult;
use num_rational::BigRational;
use num_traits::{Signed, ToPrimitive, Zero};
mod algebra;
use algebra::{combine, normalize, product};

type Ray = Vec<(usize, BigRational)>;
struct Generator {
    vector: Ray,
    faces: Vec<usize>,
}
pub(in crate::builtins::math::optim) struct Cone {
    policy: Policy,
    dimension: usize,
    lineality: Vec<Ray>,
    rays: Vec<Generator>,
    facets: usize,
}

impl Cone {
    #[cfg(test)]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub async fn full(
        policy: Policy,
        dimension: usize,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        dimension
            .checked_mul(2)
            .ok_or_else(|| policy.invalid("direction count overflow"))?;
        let mut lineality = policy.buffer(dimension)?;
        for coordinate in 0..dimension {
            let mut vector = policy.buffer(1)?;
            vector.push((coordinate, BigRational::from_integer(1.into())));
            lineality.push(vector);
            cooperation.completed(1).await?;
        }
        Ok(Self {
            policy,
            dimension,
            lineality,
            rays: Vec::new(),
            facets: 0,
        })
    }

    pub async fn intersect_normal(
        &mut self,
        normal: &Normal,
        reversed: bool,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        let policy = self.policy;
        if normal.dimension != self.dimension {
            return Err(policy.invalid("direction normal dimension mismatch"));
        }
        let mut free_products = policy.buffer(self.lineality.len())?;
        let mut pivot = None;
        for (index, vector) in self.lineality.iter().enumerate() {
            let value = product(&normal.values, vector, cooperation).await?;
            let value = if reversed { -value } else { value };
            if pivot.is_none() && !value.is_zero() {
                pivot = Some(index);
            }
            free_products.push(value);
        }
        if let Some(pivot) = pivot {
            return self
                .cut_lineality(normal, reversed, pivot, free_products, cooperation)
                .await;
        }
        self.cut_pointed(normal, reversed, cooperation).await
    }

    async fn cut_lineality(
        &mut self,
        normal: &Normal,
        reversed: bool,
        pivot: usize,
        mut products: Vec<BigRational>,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        let policy = self.policy;
        let facet = self.facets;
        self.facets = facet
            .checked_add(1)
            .ok_or_else(|| policy.invalid("facet count overflow"))?;
        let mut pivot_vector = self.lineality.remove(pivot);
        let pivot_product = products.remove(pivot);
        let mut next_free = policy.buffer(self.lineality.len())?;
        for (vector, value) in std::mem::take(&mut self.lineality)
            .into_iter()
            .zip(products)
        {
            next_free.push(if value.is_zero() {
                vector
            } else {
                projected(
                    &vector,
                    &value,
                    &pivot_vector,
                    &pivot_product,
                    cooperation,
                    policy,
                )
                .await?
            });
            cooperation.completed(1).await?;
        }
        let mut next_rays = policy.buffer(
            self.rays
                .len()
                .checked_add(1)
                .ok_or_else(|| policy.invalid("ray count overflow"))?,
        )?;
        for generator in std::mem::take(&mut self.rays) {
            let value = product(&normal.values, &generator.vector, cooperation).await?;
            let value = if reversed { -value } else { value };
            let vector = if value.is_zero() {
                generator.vector
            } else {
                projected(
                    &generator.vector,
                    &value,
                    &pivot_vector,
                    &pivot_product,
                    cooperation,
                    policy,
                )
                .await?
            };
            let mut faces = generator.faces;
            policy.push(&mut faces, facet)?;
            next_rays.push(Generator { vector, faces });
            cooperation.completed(1).await?;
        }
        // The inward part of the eliminated free line is a new pointed ray.
        // It annihilates all preceding normals because it was lineality.
        if pivot_product.is_positive() {
            for (_, value) in &mut pivot_vector {
                *value = -std::mem::take(value);
                cooperation.completed(1).await?;
            }
        }
        let mut faces = policy.buffer(facet)?;
        for prior in 0..facet {
            faces.push(prior);
            cooperation.completed(1).await?;
        }
        next_rays.push(Generator {
            vector: pivot_vector,
            faces,
        });
        self.lineality = next_free;
        self.rays = next_rays;
        Ok(())
    }

    async fn cut_pointed(
        &mut self,
        normal: &Normal,
        reversed: bool,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        let policy = self.policy;
        let mut products = policy.buffer(self.rays.len())?;
        let mut inward = policy.buffer(self.rays.len())?;
        let mut outward = policy.buffer(self.rays.len())?;
        for (index, generator) in self.rays.iter().enumerate() {
            let value = product(&normal.values, &generator.vector, cooperation).await?;
            let value = if reversed { -value } else { value };
            if value.is_negative() {
                inward.push(index);
            }
            if value.is_positive() {
                outward.push(index);
            }
            products.push(value);
        }
        // This row does not alter the cone. Retaining redundant facets is not
        // necessary for the current exact face-incidence representation.
        if outward.is_empty() {
            return Ok(());
        }
        let facet = self.facets;
        self.facets = facet
            .checked_add(1)
            .ok_or_else(|| policy.invalid("facet count overflow"))?;
        let mut boundary = policy.buffer(0)?;
        for &inside in &inward {
            for &outside in &outward {
                let mut faces = intersection(
                    &self.rays[inside].faces,
                    &self.rays[outside].faces,
                    cooperation,
                    policy,
                )
                .await?;
                if !adjacent(&self.rays, inside, outside, &faces, cooperation).await? {
                    continue;
                }
                let mut vector = combine(
                    &self.rays[inside].vector,
                    &products[outside],
                    &self.rays[outside].vector,
                    &products[inside],
                    cooperation,
                    policy,
                )
                .await?;
                if !normalize(&mut vector, cooperation).await? {
                    return Err(policy.invalid("pointed boundary generator vanished"));
                }
                if !contains(&boundary, &vector, cooperation).await? {
                    policy.push(&mut faces, facet)?;
                    policy.push(&mut boundary, Generator { vector, faces })?;
                }
            }
        }
        let mut retained = policy.buffer(self.rays.len())?;
        for (mut generator, value) in std::mem::take(&mut self.rays).into_iter().zip(products) {
            if value.is_positive() {
                continue;
            }
            if value.is_zero() {
                policy.push(&mut generator.faces, facet)?;
            }
            retained.push(generator);
            cooperation.completed(1).await?;
        }
        for generator in boundary {
            if !contains(&retained, &generator.vector, cooperation).await? {
                policy.push(&mut retained, generator)?;
            }
        }
        self.rays = retained;
        Ok(())
    }

    pub fn len(&self, basis: Basis) -> BuiltinResult<usize> {
        let policy = self.policy;
        let free = match basis {
            Basis::Maximal => self.lineality.len().checked_mul(2),
            Basis::Minimal if self.lineality.is_empty() => Some(0),
            Basis::Minimal => self.lineality.len().checked_add(1),
        }
        .ok_or_else(|| policy.invalid("direction count overflow"))?;
        free.checked_add(self.rays.len())
            .ok_or_else(|| policy.invalid("direction count overflow"))
    }

    pub fn rotatable(&self) -> bool {
        !self.lineality.is_empty()
    }

    #[cfg(test)]
    pub async fn direction(
        &self,
        index: usize,
        basis: Basis,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Vec<f64>>> {
        self.oriented_direction(index, basis, false, cooperation)
            .await
    }

    pub async fn oriented_direction(
        &self,
        index: usize,
        basis: Basis,
        reversed: bool,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Vec<f64>>> {
        let policy = self.policy;
        enum Selected<'a> {
            Borrowed(&'a Ray, bool),
            Sum(Ray),
        }
        let free = self.len(basis)? - self.rays.len();
        let selected = if index < free {
            match basis {
                Basis::Maximal => Selected::Borrowed(
                    &self.lineality[index % self.lineality.len()],
                    index >= self.lineality.len(),
                ),
                Basis::Minimal if index < self.lineality.len() => {
                    Selected::Borrowed(&self.lineality[index], false)
                }
                Basis::Minimal => {
                    let mut sum = policy.buffer(0)?;
                    let one = BigRational::from_integer(1.into());
                    for vector in &self.lineality {
                        sum = combine(&sum, &one, vector, &one, cooperation, policy).await?;
                    }
                    Selected::Sum(sum)
                }
            }
        } else {
            let generator = self
                .rays
                .get(index - free)
                .ok_or_else(|| policy.invalid("poll direction index out of range"))?;
            Selected::Borrowed(&generator.vector, false)
        };
        let (vector, negative) = match &selected {
            Selected::Borrowed(vector, negative) => (*vector, *negative),
            // combine(left,1,right,1) computes left-right, so this vector
            // is already the NEGATIVE sum of the free-basis vectors.
            Selected::Sum(vector) => (vector, false),
        };
        // Lineality can be traversed in either direction. Inward pointed
        // rays cannot be negated merely to rotate a minimal free basis.
        let negative = negative ^ (reversed && index < free);
        let mut result = policy.buffer(self.dimension)?;
        let mut entry = 0;
        for coordinate in 0..self.dimension {
            if vector
                .get(entry)
                .is_none_or(|(column, _)| *column != coordinate)
            {
                result.push(0.0);
            } else {
                let exact = &vector[entry].1;
                let Some(number) = exact.to_f64() else {
                    return Ok(None);
                };
                if !number.is_finite() || number == 0.0 && !exact.is_zero() {
                    return Ok(None);
                }
                result.push(if negative { -number } else { number });
                entry += 1;
            }
            cooperation.completed(1).await?;
        }
        Ok(Some(result))
    }

    #[cfg(test)]
    pub fn stored_nonzeros(&self) -> usize {
        self.lineality.iter().map(Vec::len).sum::<usize>()
            + self.rays.iter().map(|ray| ray.vector.len()).sum::<usize>()
    }
}

async fn projected(
    vector: &Ray,
    value: &BigRational,
    pivot: &Ray,
    pivot_product: &BigRational,
    cooperation: &mut ScalarCooperation,
    policy: Policy,
) -> BuiltinResult<Ray> {
    let mut result = combine(vector, pivot_product, pivot, value, cooperation, policy).await?;
    for (_, coordinate) in &mut result {
        *coordinate /= pivot_product;
        cooperation.completed(1).await?;
    }
    if !normalize(&mut result, cooperation).await? {
        return Err(policy.invalid("free cone basis lost rank"));
    }
    Ok(result)
}
async fn intersection(
    left: &[usize],
    right: &[usize],
    cooperation: &mut ScalarCooperation,
    policy: Policy,
) -> BuiltinResult<Vec<usize>> {
    let mut result = policy.buffer(left.len().min(right.len()))?;
    let (mut i, mut j) = (0, 0);
    while i < left.len() && j < right.len() {
        match left[i].cmp(&right[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                result.push(left[i]);
                i += 1;
                j += 1;
            }
        }
        cooperation.completed(1).await?;
    }
    Ok(result)
}
async fn adjacent(
    rays: &[Generator],
    left: usize,
    right: usize,
    common: &[usize],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<bool> {
    for (index, generator) in rays.iter().enumerate() {
        if index == left || index == right {
            continue;
        }
        let mut contains = true;
        for face in common {
            if generator.faces.binary_search(face).is_err() {
                contains = false;
                break;
            }
            cooperation
                .completed((generator.faces.len().max(1).ilog2() + 1) as u64)
                .await?;
        }
        cooperation.completed(1).await?;
        if contains {
            return Ok(false);
        }
    }
    Ok(true)
}
async fn contains(
    rays: &[Generator],
    candidate: &Ray,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<bool> {
    for generator in rays {
        if generator.vector.len() != candidate.len() {
            continue;
        }
        let mut same = true;
        for (left, right) in generator.vector.iter().zip(candidate) {
            if left != right {
                same = false;
                break;
            }
            cooperation.completed(1).await?;
        }
        cooperation.completed(1).await?;
        if same {
            return Ok(true);
        }
    }
    Ok(false)
}
