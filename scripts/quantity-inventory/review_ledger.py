"""Stable review identities independent of source line and numeric storage spelling.

Reviews live separately from regenerated discovery. Repeated same-name records use a
stable source-order ordinal; ambiguous symbol moves must be reviewed rather than guessed.
Removed candidates remain in the ledger for audit and are never counted as new closure.
"""
from collections import Counter
import hashlib
import json


def identify(records, reference=False):
    seen = Counter()
    for record in records:
        if reference:
            key = (record['path'], record['source'], tuple(record['representations']))
        else:
            key = tuple(record.get(k, '') for k in ('path', 'kind', 'owner', 'name'))
        ordinal = seen[key]
        seen[key] += 1
        identity = json.dumps([key, ordinal], ensure_ascii=True, separators=(',', ':'))
        record['review_id'] = hashlib.sha256(identity.encode()).hexdigest()[:24]
    return records


def apply(records, ledger, reference=False):
    identify(records, reference)
    for record in records:
        review = ledger.get(record['review_id'])
        if review is not None:
            record['disposition'] = review['disposition']
            record['review'] = {k: v for k, v in review.items() if k != 'disposition'}
    return records


def retain_existing(records, ledger, reference=False):
    for record in identify(records, reference):
        if record.get('disposition', 'unclassified') != 'unclassified':
            ledger.setdefault(record['review_id'], {
                'disposition': record['disposition'],
                **record.get('review', {}),
            })
