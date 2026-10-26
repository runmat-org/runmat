#!/usr/bin/env python3
"""Retain explicit, narrow reviewed domain dispositions; never approve whole physics crates."""
import json
import re
from pathlib import Path
root=Path(__file__).resolve().parents[2]
p=root/'quantity-migration/reviews.json'
ledger=json.loads(p.read_text())
records=json.loads((root/'quantity-migration/inventory.json').read_text())['records']
# Static quantity aliases resolve to the same implementation; classifying these
# does not approve adjacent raw numeric tuple members or another local type.
catalog=(root/'crates/runmat-types/src/physical/catalog.rs').read_text()
shared_aliases=set(re.findall(r'pub type (\w+) = Quantity<kind::', catalog))
shared_aliases.add('DimensionedScalar')
def shared_alias_type(typ):
    tokens=set(re.findall(r'\b[A-Za-z_]\w*\b', typ))
    return bool(tokens & shared_aliases) and not (tokens & {'f64','f32','u64','usize'})

# Generic numerical algorithms own no physical policy and never introduce a unit
# conversion. Their dimension-bearing callers are audited separately, not exempted.
numerical_modules={
 'crates/runmat-analysis/fea/src/solve/krylov.rs': 'matrix-free GMRES mathematical recurrence; physical operators/row layouts remain in calling owner, original residual policy is relative and no dimensional magnitude floor or implicit source scaling is introduced',
 'crates/runmat-analysis/fea/src/solve/krylov/right_ilu.rs': 'generic fixed-support nonsymmetric numerical factorization; no physical matrix dimension asserted, no dimensional pivot floor or pivot replacement; caller retains original physical operator and validates original equation after right preconditioning',
 'crates/runmat-analysis/fea/src/solve/krylov/right_diagonal.rs': 'generic invertible right scaling; original physical equation and typed solution coordinates retained by caller; no diagonal floor/sign replacement or invented physical magnitude',
 'crates/runmat-analysis/fea/src/solve/fixed_point.rs': 'physics-independent affine Anderson proposal; coefficient is a ratio, image remains a mathematical vector until calling owner accepts it against original physical equations; mixed-unit norm is not supplied by this helper',
 'crates/runmat-meshing/tetrahedron/src/cdt/volume_mesh/frame.rs': 'private canonical insertion-order frame derived exclusively from DelaunayConstraints typed Position nodes; origin/scale are temporary extracted canonical metres, axes and quantized ordering coordinates are ratios; all actual predicates and published mesh nodes preserve original typed physical coordinates',

 'crates/runmat-analysis/fea/src/algebra/matrix.rs': 'generic real CSR numerical container and numerical content identity; coefficients have caller-owned output/input dimensions, retained by QuantityLinearOperator/StructuralPhysicalOperator/PhysicalScalarSystem; container does not advertise a uniform physical dimension',
 'crates/runmat-analysis/fea/src/algebra/matrix/assembler.rs': 'generic real sparse triplet accumulation and compensated duplicate reduction; physical law/row/basis descriptors belong to checked operator owners, not triplet entries',
 'crates/runmat-analysis/fea/src/algebra/compensated.rs': 'unit-polymorphic Neumaier scalar accumulation; caller supplies scaling and physical result admission, no independent physical policy or conversion',
 'crates/runmat-analysis/fea/src/algebra/normalized_vector.rs': 'generic affine vector normalization and directional differential kernel; output orientation is unitless and derivative units follow supplied perturbation parameters, not an invented Position unit',
 'crates/runmat-analysis/fea/src/algebra/right_inverse.rs': 'unit-polymorphic real CSR inverse certificate; product A*B is mathematical identity, no independently stored physical quantity or unit conversion',
 'crates/runmat-analysis/fea/src/algebra/coercivity/symbolic/numeric_support.rs': 'symbolic sparsity/numeric support arithmetic, not dimensional physical fields',

 'crates/runmat-analysis/fea/src/algebra/reference.rs': 'deterministic sparse/vector algebra; StructuralPhysicalOperator and StructuralSolveLayout retain physical lanes at structural calls; other physics owner admission remains separately inventoried',
 'crates/runmat-analysis/fea/src/algebra/anchored_energy/outward.rs': 'outward-rounded interval arithmetic, not an independently dimensioned energy field',
 'crates/runmat-meshing/core/src/quality/predicate/vector.rs': 'generic affine-vector dot/cross/norm arithmetic; no imported physical scale or persisted quantity authority',
 'crates/runmat-meshing/core/src/quality/predicate/exact_dyadic.rs': 'exact dyadic significand/exponent arithmetic for robust predicates, not physical quantities',
}
count=0
for record in records:
 rid=record['review_id']; typ=record['type']; path=record['path']
 if record.get('test_context'):
  review={'disposition':'test_fixture','reason':'AST test scope; fixture values do not define a production representation. Tests remain required verification.'}
 elif shared_alias_type(typ):
  review={'disposition':'typed_quantity','reason':'Static quantity alias resolves to shared Quantity<K>; the type contains no independently raw numeric storage outside that alias.', 'basis':'crates/runmat-types/src/physical/catalog.rs static aliases, reexported by analysis/core/problem/quantity.rs; no alternate implementation', 'validation':'Alias consumers passed aggregate library compilation; final combined qualification pending.'}
 elif 'runmat_types :: physical ::' in typ:
  unit=any(n in typ for n in ('CoordinateUnits','UnitEvidence','UnitDefinition','SourceQuantity','SourceMeasure','Unit <'))
  review={'disposition':'typed_unit_or_provenance' if unit else 'typed_quantity','reason':'Declaration directly carries the shared physical implementation; scalar/storage serde adapters validate external canonical values. Mixed structural/modal lanes retain their actual DOF/basis metadata.','validation':'Representation reviewed; downstream aggregate compilation and final combined qualification pending.'}
 elif typ in {'PhysicalDimension','runmat_types :: PhysicalDimension','QuantityKind','PhysicalQuantityKind','QuantityDimension','DynamicDimension','LengthUnit','LengthUnitEvidence','UnitDefinition','UnitAttribution','ObservationQuantity'}:
  review={'disposition':'typed_unit_or_provenance','reason':'Shared dimension/unit/kind metadata (or a concrete observation selector), not an independently stored numerical physical value. PhysicalQuantityKind and QuantityDimension resolve to shared catalog/descriptor aliases; observation field values are separately admitted through their definition and QuantityField layout.','basis':'shared physical authority and results/field/physical_storage.rs; legacy metadata aliases retained only as reexports'}
 elif path in numerical_modules:
  review={'disposition':'approved_raw_numeric_boundary','reason':numerical_modules[path], 'basis':'quantity-migration/boundary-reviews.md#generic-numerical-kernels', 'validation':'Pure algorithm scope reviewed independently of physical owner migration; final compiler campaign pending.'}
 elif path=='crates/runmat-analysis/fea/src/physics/structural/topology/identity.rs' and record['owner']=='NodeContract':
  review={'disposition':'not_physical','reason':'u64 IEEE bit-pattern identity encoding of StructuralNode typed Position; serializer preserves the existing coordinates_m wire name. Bits are not meters or numerical solver inputs.'}
 else:
  continue
 ledger['rust'][rid]=review;count+=1
reference_count=0
for record in json.loads((root/'quantity-migration/references.json').read_text())['records']:
 path=record['path']; source=record['source']
 if source.startswith(('//','/*','* ')):
  review={'disposition':'not_physical','reason':'Source comment/documentation reference; executable declarations and consumers are reviewed separately.'}
 elif path in {'bindings/ts/src/generated/fea-semantic-contracts.ts','bindings/ts/src/generated/fea-analysis-contracts.ts','bindings/ts/src/generated/fea-workspace-contracts.ts','bindings/ts/src/generated/fea-assessment-guards.ts'}:
  review={'disposition':'typed_unit_or_provenance','reason':'Generated projection/guard of authoritative Rust model and shared Quantity/Unit serde wire; generation fixtures and bindings tsc passed after canonical storage cutover. No independently chosen numeric scale.'}
 elif set(record['representations']) <= {'ObservationQuantity','PhysicalQuantityKind','QuantityKind','PhysicalDimension','CanonicalQuantity','FiniteQuantity'}:
  review={'disposition':'typed_unit_or_provenance','reason':'Reference to the shared dimension/kind authority or an observation selector/quantity trait; this lexical site defines no separately owned numeric physical storage. Actual fields, numerical producers and quantity admission are classified in the AST ledger independently.','basis':'runmat-types physical catalog/PhysicalDimension and analysis observation/field quantity definitions'}
 elif 'Quantity' in record['representations'] and ('physical::' in source or 'physical ::' in source):
  review={'disposition':'typed_quantity','reason':'Explicit reference to shared Quantity implementation or construction/extraction API, not a parallel numeric/unit representation. This lexical reference classification does not approve a surrounding raw physical field or numerical algorithm; those have separate AST reviews.','basis':'crates/runmat-types/src/physical/quantity.rs and checked source/dynamic/vector wrappers'}
 elif source.startswith('pub use ') and 'physical' in source:
  review={'disposition':'typed_unit_or_provenance','reason':'Stable semantic reexport of the shared physical authority; does not define another representation.'}
 else:
  continue
 ledger['references'][record['review_id']]=review;reference_count+=1
p.write_text(json.dumps(ledger,indent=2)+'\n')
# Publish review application without rescanning or dropping historical dispositions.
from review_ledger import apply
active_counts={}
for name, reference in [('inventory',False),('references',True)]:
 destination=root/f'quantity-migration/{name}.json'
 document=json.loads(destination.read_text())
 apply(document['records'],ledger['references' if reference else 'rust'],reference)
 destination.write_text(json.dumps(document,indent=2)+'\n')
 active_counts[name]=sum(record['disposition']=='unclassified' for record in document['records'])
summary_path=root/'quantity-migration/summary.json'
summary=json.loads(summary_path.read_text())
summary['unclassified_rust_candidates']=active_counts['inventory']
summary['unclassified_reference_sites']=active_counts['references']
summary_path.write_text(json.dumps(summary,indent=2)+'\n')
print(f'{count} Rust and {reference_count} reference reviewed records retained; all other records unchanged/unclassified')
