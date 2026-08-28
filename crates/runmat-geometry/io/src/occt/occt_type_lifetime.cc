// Copyright (c) 1998-1999 Matra Datavision
// Copyright (c) 1999-2014 OPEN CASCADE SAS
//
// This file is part of Open CASCADE Technology software library.
//
// This library is free software; you can redistribute it and/or modify it under
// the terms of the GNU Lesser General Public License version 2.1 as published
// by the Free Software Foundation, with special exception defined in the OCCT
// distribution's OCCT_LGPL_EXCEPTION.txt.

// This focused TKernel override preserves the OCCT 7.8 Standard_Type ABI while
// making its registry process-lifetime storage. OCCT issue #146 documents the
// static-link destruction-order crash after STEP/IGES types are instantiated.

#include <Standard_Type.hxx>

#include <Standard_Assert.hxx>
#include <Standard_CStringHasher.hxx>
#include <Standard_Mutex.hxx>

#include <unordered_map>

IMPLEMENT_STANDARD_RTTIEXT(Standard_Type,Standard_Transient)

namespace {

Standard_CString copy_string(const char* value) {
  const size_t length = strlen(value);
  char* result = static_cast<char*>(Standard::Allocate(length + 1));
  strncpy(result, value, length + 1);
  return result;
}

using registry_type = std::unordered_map<Standard_CString,
                                         Standard_Type*,
                                         Standard_CStringHasher,
                                         Standard_CStringHasher>;

registry_type& type_registry() {
  // Type descriptors are themselves process-lifetime singletons. Keeping their registry alive
  // for the same lifetime prevents cross-library static destructor order from dereferencing a
  // destroyed map. The operating system reclaims this allocation at process exit.
  static registry_type* registry = new registry_type();
  return *registry;
}

Handle(Standard_Type) root_type = STANDARD_TYPE(Standard_Transient);

} // namespace

Standard_Type::Standard_Type(const std::type_info& info,
                             const char* name,
                             Standard_Size size,
                             const Handle(Standard_Type)& parent)
    : myInfo(info),
      myName(copy_string(name)),
      mySize(size),
      myParent(parent) {}

Standard_Boolean Standard_Type::SubType(const Handle(Standard_Type)& other) const {
  return !other.IsNull() &&
         (other == this || (!myParent.IsNull() && myParent->SubType(other)));
}

Standard_Boolean Standard_Type::SubType(const Standard_CString name) const {
  return name != nullptr &&
         (IsEqual(myName, name) || (!myParent.IsNull() && myParent->SubType(name)));
}

void Standard_Type::Print(Standard_OStream& stream) const {
  stream << std::hex << (Standard_Address)this << " : " << std::dec << myName;
}

Standard_Type* Standard_Type::Register(const std::type_info& info,
                                       const char* name,
                                       Standard_Size size,
                                       const Handle(Standard_Type)& parent) {
  static Standard_Mutex mutex;
  Standard_Mutex::Sentry guard(mutex);
  registry_type& registry = type_registry();
  const auto existing = registry.find(name);
  if (existing != registry.end()) {
    return existing->second;
  }
  Standard_Type* descriptor = new Standard_Type(info, name, size, parent);
  registry.emplace(name, descriptor);
  return descriptor;
}

Standard_Type::~Standard_Type() {
  registry_type& registry = type_registry();
  Standard_ASSERT(registry.erase(myName) > 0,
                  "Standard_Type::~Standard_Type() cannot find itself in registry",);
  Standard::Free(myName);
}
