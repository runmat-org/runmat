`core.xsd` and `production.xsd` are the XSD schemas from the 3MF Core Specification 1.3
(Appendix B.1) and the Production Extension 1.1 (Appendix B.1), copyright 3MF Consortium,
BSD 2-Clause License (https://github.com/3MFConsortium/spec_core,
https://github.com/3MFConsortium/spec_production). Two changes: the `xml.xsd` import points at the local `xml.xsd`, so validation needs no
network, and `maxOccurs="2147483647"` reads `unbounded`, because libxml2 refuses that
bound as a schema value. The tests validate
RunMat's 3MF output against them with `xmllint --schema schema.xsd`.
