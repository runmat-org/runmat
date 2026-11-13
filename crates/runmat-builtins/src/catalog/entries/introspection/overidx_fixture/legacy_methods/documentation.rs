use crate::*;
const FIXTURE_SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Fixture execution boundary",
        paragraphs: &[
            "This is an executable MethodOnly fixture compiled in ordinary native/WASM runtimes, not a public completion or standalone API. Its canonical default binding retains original arguments and selected cancellation authority; no additional registry or executor is introduced.",
            "Scalar-only arithmetic routes use no new payload backing and retain their existing double results. Object, serialization and dynamic alias-output routes lack a physical finite-storage certificate and refuse Managed invocations before projection or effects. There is no accelerator provider, implicit gather, distributed route or numeric fusion contract."
        ]
    }
];
pub(super) const OVERIDX_NUM_ARGUMENTS_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("OverIdx.numArgumentsFromSubscript"),
    keywords: &["OverIdx.numArgumentsFromSubscript", "indexing overload", "nargs property", "test fixture"],
    slug: Some("overidx.numargumentsfromsubscript"),
    summary: "Executable class-method fixture.",
    description: "Requires a value object of any class and ignores the subscript and indexing-context arguments. Returns the nargs property of any class when present, otherwise scalar double three.",
    sections: FIXTURE_SECTIONS,
    example_exemption: Some("MethodOnly executable fixture exercised by existing semantic/VM class tests, not a public documentation example."),
    status: Some(BuiltinDocumentationStatus::Partial),
    ..BuiltinDocumentation::EMPTY
};
pub(super) const OVERIDX_SAVEOBJ_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("OverIdx.saveobj"),
    keywords: &["OverIdx.saveobj", "serialization", "struct payload", "property preservation", "test fixture"],
    slug: Some("overidx.saveobj"),
    summary: "Executable class-method fixture.",
    description: "Requires a value object, copies present k/nargs properties to a scalar struct and adds string saved_by=OverIdx.saveobj. Arbitrary property payload classes are preserved.",
    sections: FIXTURE_SECTIONS,
    example_exemption: Some("MethodOnly executable fixture exercised by existing semantic/VM class tests, not a public documentation example."),
    status: Some(BuiltinDocumentationStatus::Partial),
    ..BuiltinDocumentation::EMPTY
};
pub(super) const OVERIDX_LOADOBJ_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("OverIdx.loadobj"),
    keywords: &["OverIdx.loadobj", "deserialization", "value object", "property restoration", "test fixture"],
    slug: Some("overidx.loadobj"),
    summary: "Executable class-method fixture.",
    description: "Accepts any input. A struct contributes fields except saved_by; nonstruct input contributes no fields. Returns a value object of class OverIdx and adds string loaded_by=OverIdx.loadobj.",
    sections: FIXTURE_SECTIONS,
    example_exemption: Some("MethodOnly executable fixture exercised by existing semantic/VM class tests, not a public documentation example."),
    status: Some(BuiltinDocumentationStatus::Partial),
    ..BuiltinDocumentation::EMPTY
};
pub(super) const OVERIDX_PLUS_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("OverIdx.plus"),
    keywords: &["OverIdx.plus", "operator overload", "scalar addition", "test fixture"],
    slug: Some("overidx.plus"),
    summary: "Executable class-method fixture.",
    description: "Accepts a value object on either side and one scalar double/native integer on the other. The object contributes double property k, or zero when absent/non-double; native integers convert via legacy f64, which may round. Returns lhs+rhs.",
    sections: FIXTURE_SECTIONS,
    example_exemption: Some("MethodOnly executable fixture exercised by existing semantic/VM class tests, not a public documentation example."),
    status: Some(BuiltinDocumentationStatus::Partial),
    ..BuiltinDocumentation::EMPTY
};
pub(super) const OVERIDX_TIMES_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("OverIdx.times"),
    keywords: &["OverIdx.times", "operator overload", "scalar product", "test fixture"],
    slug: Some("overidx.times"),
    summary: "Executable class-method fixture.",
    description: "Accepts a value object on either side and one scalar double/native integer on the other. The object contributes double property k, or zero when absent/non-double; native integers convert via legacy f64, which may round. Returns lhs*rhs.",
    sections: FIXTURE_SECTIONS,
    example_exemption: Some("MethodOnly executable fixture exercised by existing semantic/VM class tests, not a public documentation example."),
    status: Some(BuiltinDocumentationStatus::Partial),
    ..BuiltinDocumentation::EMPTY
};
pub(super) const OVERIDX_MTIMES_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("OverIdx.mtimes"),
    keywords: &["OverIdx.mtimes", "operator overload", "scalar multiplication", "test fixture"],
    slug: Some("overidx.mtimes"),
    summary: "Executable class-method fixture.",
    description: "Accepts a value object on either side and one scalar double/native integer on the other. The object contributes double property k, or zero when absent/non-double; native integers convert via legacy f64, which may round. Returns lhs*rhs, not matrix multiplication.",
    sections: FIXTURE_SECTIONS,
    example_exemption: Some("MethodOnly executable fixture exercised by existing semantic/VM class tests, not a public documentation example."),
    status: Some(BuiltinDocumentationStatus::Partial),
    ..BuiltinDocumentation::EMPTY
};
pub(super) const OVERIDX_LT_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("OverIdx.lt"),
    keywords: &["OverIdx.lt", "operator overload", "less-than comparison", "double result", "test fixture"],
    slug: Some("overidx.lt"),
    summary: "Executable class-method fixture.",
    description: "Accepts a value object on either side and one scalar double/native integer on the other. The object contributes double property k, or zero when absent/non-double; native integers convert via legacy f64, which may round. Returns double one/zero for lhs<rhs, not logical.",
    sections: FIXTURE_SECTIONS,
    example_exemption: Some("MethodOnly executable fixture exercised by existing semantic/VM class tests, not a public documentation example."),
    status: Some(BuiltinDocumentationStatus::Partial),
    ..BuiltinDocumentation::EMPTY
};
pub(super) const OVERIDX_GT_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("OverIdx.gt"),
    keywords: &["OverIdx.gt", "operator overload", "greater-than comparison", "double result", "test fixture"],
    slug: Some("overidx.gt"),
    summary: "Executable class-method fixture.",
    description: "Accepts a value object on either side and one scalar double/native integer on the other. The object contributes double property k, or zero when absent/non-double; native integers convert via legacy f64, which may round. Returns double one/zero for lhs>rhs, not logical.",
    sections: FIXTURE_SECTIONS,
    example_exemption: Some("MethodOnly executable fixture exercised by existing semantic/VM class tests, not a public documentation example."),
    status: Some(BuiltinDocumentationStatus::Partial),
    ..BuiltinDocumentation::EMPTY
};
pub(super) const OVERIDX_EQ_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("OverIdx.eq"),
    keywords: &["OverIdx.eq", "operator overload", "tolerant equality", "double result", "test fixture"],
    slug: Some("overidx.eq"),
    summary: "Executable class-method fixture.",
    description: "Accepts a value object on either side and one scalar double/native integer on the other. The object contributes double property k, or zero when absent/non-double; native integers convert via legacy f64, which may round. Returns double one when abs(lhs-rhs)<1e-12, otherwise double zero.",
    sections: FIXTURE_SECTIONS,
    example_exemption: Some("MethodOnly executable fixture exercised by existing semantic/VM class tests, not a public documentation example."),
    status: Some(BuiltinDocumentationStatus::Partial),
    ..BuiltinDocumentation::EMPTY
};
pub(super) const OVERIDX_UPLUS_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("OverIdx.uplus"),
    keywords: &["OverIdx.uplus", "operator overload", "unary identity", "alias preservation", "test fixture"],
    slug: Some("overidx.uplus"),
    summary: "Executable class-method fixture.",
    description: "Returns its input unchanged, of any class, preserving alias and residency. It does not validate an object receiver.",
    sections: FIXTURE_SECTIONS,
    example_exemption: Some("MethodOnly executable fixture exercised by existing semantic/VM class tests, not a public documentation example."),
    status: Some(BuiltinDocumentationStatus::Partial),
    ..BuiltinDocumentation::EMPTY
};
pub(super) const OVERIDX_RDIVIDE_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("OverIdx.rdivide"),
    keywords: &["OverIdx.rdivide", "operator overload", "right division", "scalar arithmetic", "test fixture"],
    slug: Some("overidx.rdivide"),
    summary: "Executable class-method fixture.",
    description: "Accepts a value object on either side and one scalar double/native integer on the other. The object contributes double property k, or zero when absent/non-double; native integers convert via legacy f64, which may round. Returns lhs/rhs.",
    sections: FIXTURE_SECTIONS,
    example_exemption: Some("MethodOnly executable fixture exercised by existing semantic/VM class tests, not a public documentation example."),
    status: Some(BuiltinDocumentationStatus::Partial),
    ..BuiltinDocumentation::EMPTY
};
pub(super) const OVERIDX_MRDIVIDE_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("OverIdx.mrdivide"),
    keywords: &["OverIdx.mrdivide", "operator overload", "scalar right division", "test fixture"],
    slug: Some("overidx.mrdivide"),
    summary: "Executable class-method fixture.",
    description: "Accepts a value object on either side and one scalar double/native integer on the other. The object contributes double property k, or zero when absent/non-double; native integers convert via legacy f64, which may round. Uses the same scalar lhs/rhs route as rdivide.",
    sections: FIXTURE_SECTIONS,
    example_exemption: Some("MethodOnly executable fixture exercised by existing semantic/VM class tests, not a public documentation example."),
    status: Some(BuiltinDocumentationStatus::Partial),
    ..BuiltinDocumentation::EMPTY
};
pub(super) const OVERIDX_LDIVIDE_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("OverIdx.ldivide"),
    keywords: &["OverIdx.ldivide", "operator overload", "left division", "scalar arithmetic", "test fixture"],
    slug: Some("overidx.ldivide"),
    summary: "Executable class-method fixture.",
    description: "Accepts a value object on either side and one scalar double/native integer on the other. The object contributes double property k, or zero when absent/non-double; native integers convert via legacy f64, which may round. Returns rhs/lhs.",
    sections: FIXTURE_SECTIONS,
    example_exemption: Some("MethodOnly executable fixture exercised by existing semantic/VM class tests, not a public documentation example."),
    status: Some(BuiltinDocumentationStatus::Partial),
    ..BuiltinDocumentation::EMPTY
};
pub(super) const OVERIDX_MLDIVIDE_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("OverIdx.mldivide"),
    keywords: &["OverIdx.mldivide", "operator overload", "scalar left division", "test fixture"],
    slug: Some("overidx.mldivide"),
    summary: "Executable class-method fixture.",
    description: "Accepts a value object on either side and one scalar double/native integer on the other. The object contributes double property k, or zero when absent/non-double; native integers convert via legacy f64, which may round. Uses the same scalar rhs/lhs route as ldivide.",
    sections: FIXTURE_SECTIONS,
    example_exemption: Some("MethodOnly executable fixture exercised by existing semantic/VM class tests, not a public documentation example."),
    status: Some(BuiltinDocumentationStatus::Partial),
    ..BuiltinDocumentation::EMPTY
};
pub(super) const OVERIDX_AND_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("OverIdx.and"),
    keywords: &["OverIdx.and", "operator overload", "nonzero conjunction", "double result", "test fixture"],
    slug: Some("overidx.and"),
    summary: "Executable class-method fixture.",
    description: "Accepts a value object on either side and one scalar double/native integer on the other. The object contributes double property k, or zero when absent/non-double; native integers convert via legacy f64, which may round. Returns double one/zero for nonzero(lhs) AND nonzero(rhs).",
    sections: FIXTURE_SECTIONS,
    example_exemption: Some("MethodOnly executable fixture exercised by existing semantic/VM class tests, not a public documentation example."),
    status: Some(BuiltinDocumentationStatus::Partial),
    ..BuiltinDocumentation::EMPTY
};
pub(super) const OVERIDX_OR_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("OverIdx.or"),
    keywords: &["OverIdx.or", "operator overload", "nonzero disjunction", "double result", "test fixture"],
    slug: Some("overidx.or"),
    summary: "Executable class-method fixture.",
    description: "Accepts a value object on either side and one scalar double/native integer on the other. The object contributes double property k, or zero when absent/non-double; native integers convert via legacy f64, which may round. Returns double one/zero for nonzero(lhs) OR nonzero(rhs).",
    sections: FIXTURE_SECTIONS,
    example_exemption: Some("MethodOnly executable fixture exercised by existing semantic/VM class tests, not a public documentation example."),
    status: Some(BuiltinDocumentationStatus::Partial),
    ..BuiltinDocumentation::EMPTY
};
pub(super) const OVERIDX_XOR_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("OverIdx.xor"),
    keywords: &["OverIdx.xor", "operator overload", "nonzero exclusive disjunction", "double result", "test fixture"],
    slug: Some("overidx.xor"),
    summary: "Executable class-method fixture.",
    description: "Accepts a value object on either side and one scalar double/native integer on the other. The object contributes double property k, or zero when absent/non-double; native integers convert via legacy f64, which may round. Returns double one/zero for exclusive nonzero(lhs)/nonzero(rhs).",
    sections: FIXTURE_SECTIONS,
    example_exemption: Some("MethodOnly executable fixture exercised by existing semantic/VM class tests, not a public documentation example."),
    status: Some(BuiltinDocumentationStatus::Partial),
    ..BuiltinDocumentation::EMPTY
};
