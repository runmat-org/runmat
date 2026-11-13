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
pub(super) const POINT_MOVE_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("Point.move"),
    keywords: &["Point.move", "classdef", "point coordinates", "translation", "test fixture"],
    slug: Some("point.move"),
    summary: "Executable class-method fixture.",
    description: "Requires a value object (not necessarily class Point), reads numeric x/y properties or defaults each to zero, then returns an updated value object with double x+dx and y+dy. Scalar integer dx/dy convert through the legacy f64 conversion; uint64 values can round.",
    sections: FIXTURE_SECTIONS,
    example_exemption: Some("MethodOnly executable fixture exercised by existing semantic/VM class tests, not a public documentation example."),
    status: Some(BuiltinDocumentationStatus::Partial),
    ..BuiltinDocumentation::EMPTY
};
pub(super) const POINT_ORIGIN_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("Point.origin"),
    keywords: &["Point.origin", "classdef", "coordinate origin", "value object", "test fixture"],
    slug: Some("point.origin"),
    summary: "Executable class-method fixture.",
    description: "Creates a value object of class Point with exactly double properties x=0 and y=0.",
    sections: FIXTURE_SECTIONS,
    example_exemption: Some("MethodOnly executable fixture exercised by existing semantic/VM class tests, not a public documentation example."),
    status: Some(BuiltinDocumentationStatus::Partial),
    ..BuiltinDocumentation::EMPTY
};
pub(super) const SHAPE_AREA_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("Shape.area"),
    keywords: &["Shape.area", "classdef", "area", "base method", "test fixture"],
    slug: Some("shape.area"),
    summary: "Executable class-method fixture.",
    description: "Ignores its argument of any class and returns scalar double zero. It does not require an object receiver.",
    sections: FIXTURE_SECTIONS,
    example_exemption: Some("MethodOnly executable fixture exercised by existing semantic/VM class tests, not a public documentation example."),
    status: Some(BuiltinDocumentationStatus::Partial),
    ..BuiltinDocumentation::EMPTY
};
pub(super) const CIRCLE_AREA_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("Circle.area"),
    keywords: &["Circle.area", "classdef", "circle area", "radius", "test fixture"],
    slug: Some("circle.area"),
    summary: "Executable class-method fixture.",
    description: "Requires a value object of any class, reads property r only when it is a double scalar (otherwise zero), and returns scalar double pi*r*r.",
    sections: FIXTURE_SECTIONS,
    example_exemption: Some("MethodOnly executable fixture exercised by existing semantic/VM class tests, not a public documentation example."),
    status: Some(BuiltinDocumentationStatus::Partial),
    ..BuiltinDocumentation::EMPTY
};
pub(super) const CTOR_CTOR_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("Ctor.Ctor"),
    keywords: &["Ctor.Ctor", "classdef", "constructor", "property initialization", "test fixture"],
    slug: Some("ctor.ctor"),
    summary: "Executable class-method fixture.",
    description: "Converts its scalar double/native integer x using the legacy f64 conversion, then creates value object Ctor with double property x.",
    sections: FIXTURE_SECTIONS,
    example_exemption: Some("MethodOnly executable fixture exercised by existing semantic/VM class tests, not a public documentation example."),
    status: Some(BuiltinDocumentationStatus::Partial),
    ..BuiltinDocumentation::EMPTY
};
pub(super) const PKGF_FOO_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("PkgF.foo"),
    keywords: &["PkgF.foo", "package function", "import precedence", "test fixture"],
    slug: Some("pkgf.foo"),
    summary: "Executable class-method fixture.",
    description: "Returns scalar double ten without inputs.",
    sections: FIXTURE_SECTIONS,
    example_exemption: Some("MethodOnly executable fixture exercised by existing semantic/VM class tests, not a public documentation example."),
    status: Some(BuiltinDocumentationStatus::Partial),
    ..BuiltinDocumentation::EMPTY
};
pub(super) const PKGG_FOO_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("PkgG.foo"),
    keywords: &["PkgG.foo", "package function", "import precedence", "test fixture"],
    slug: Some("pkgg.foo"),
    summary: "Executable class-method fixture.",
    description: "Returns scalar double twenty without inputs.",
    sections: FIXTURE_SECTIONS,
    example_exemption: Some("MethodOnly executable fixture exercised by existing semantic/VM class tests, not a public documentation example."),
    status: Some(BuiltinDocumentationStatus::Partial),
    ..BuiltinDocumentation::EMPTY
};
