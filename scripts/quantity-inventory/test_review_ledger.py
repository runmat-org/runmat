import unittest
from review_ledger import apply, identify, retain_existing


class LedgerTests(unittest.TestCase):
    def test_line_and_numeric_type_changes_preserve_review(self):
        old = dict(path='core/model.rs', kind='field', owner='Material', name='density', line=8, type='f64', disposition='migrated', review={'quantity': 'MassDensity'})
        ledger = {}
        retain_existing([old], ledger)
        new = {**old, 'line': 42, 'type': 'MassDensity', 'disposition': 'unclassified'}
        del new['review']
        apply([new], ledger)
        self.assertEqual(new['disposition'], 'migrated')
        self.assertEqual(new['review']['quantity'], 'MassDensity')

    def test_new_symbols_are_not_classified_by_neighbor(self):
        old = dict(path='core/model.rs', kind='field', owner='Material', name='density', disposition='migrated')
        ledger = {}
        retain_existing([old], ledger)
        new = {**old, 'name': 'viscosity', 'disposition': 'unclassified'}
        apply([new], ledger)
        self.assertEqual(new['disposition'], 'unclassified')

    def test_colliding_names_have_distinct_review_ids(self):
        rows = [dict(path='a.rs', kind='parameter', owner='impl', name='value') for _ in range(2)]
        identify(rows)
        self.assertNotEqual(rows[0]['review_id'], rows[1]['review_id'])

    def test_removed_reviews_are_retained(self):
        ledger = {'old': {'disposition': 'migrated', 'quantity': 'Length'}}
        apply([], ledger)
        self.assertIn('old', ledger)


if __name__ == '__main__':
    unittest.main()
