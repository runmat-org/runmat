"""Validate a 3MF model against the local schemas without network access."""

import sys

from lxml import etree


parser = etree.XMLParser(no_network=True, resolve_entities=False)
schema = etree.XMLSchema(etree.parse(sys.argv[1], parser))
schema.assertValid(etree.parse(sys.argv[2], parser))
