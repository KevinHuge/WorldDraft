"""Question-bank contracts: real mask IDs, prompt provenance, complete categories, balanced polarities."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re

import pytest

from worlddraft_eval.common import paths

requires_data = pytest.mark.skipif(not paths.has_data(), reason='benchmark data not downloaded')
CATALOG = paths.PROTOCOL / 'qa'
FIELDS = 'item_id keywords proposition_id dependency category_broad category_detailed tuple question_natural_language attribute entity_id metric expected_answer applies_to'.split()
CASES = {f'{a}{i:02}' for a in 'ABCDEF' for i in range(1, 9)}


def read(path):
    return json.loads(path.read_text())


def questions(path=CATALOG / 'questions.csv'):
    with path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == FIELDS
        return list(reader)


def test_catalog_coverage():
    rows = questions()
    assert {r['item_id'] for r in rows} == CASES
    counts = read(CATALOG / 'coverage.json')
    assert counts['questions'] == len(rows)
    assert counts['metrics'] == dict(Counter(r['metric'] for r in rows))
    assert counts['expected_answers'] == {
        metric: dict(Counter(r['expected_answer'] for r in rows if r['metric'] == metric))
        for metric in ('RF', 'AP', 'SL')
    }
    for case in CASES:
        scene = [r for r in rows if r['item_id'] == case]
        assert {r['metric'] for r in scene} == {'EntityCount', 'RF', 'AP', 'SL'}
        assert [r['proposition_id'] for r in scene] == [str(i) for i in range(1, len(scene) + 1)]
        assert len({(r['metric'], r['entity_id'], r['question_natural_language']) for r in scene}) == len(scene)


@requires_data
def test_every_region_and_count_group_has_real_visible_gt():
    rows = questions()
    for case in CASES:
        path = paths.PROTOCOL / f'questions/{case}.json'
        config = read(path)
        gt = read(paths.settings_file(path, config['files']['visibility_gt']))
        entities = {e['entity_id'] for e in gt['entities']}
        labels = config['qa']['entity_labels']
        groups = config['qa']['groups'][case]
        bindings = config['qa']['bindings'][case]
        assert set(labels) == entities
        assert config['qa']['crop'] == 'gt_bbox'
        assert config['qa']['global_count_view'] == 'full_frame'
        assert config['qa']['visibility_from_masks'] is True
        assert all({e['entity_id'] for e in f['entities']} == entities for f in gt['frames'])
        for name, members in groups.items():
            assert members and set(members) <= entities
            assert len({labels[e] for e in members}) == 1
            label = labels[members[0]]
            assert set(members) == {e for e in entities if labels[e] == label}
        scene = [r for r in rows if r['item_id'] == case]
        local = set()
        for row in scene:
            eid, pid, metric = row['entity_id'], row['proposition_id'], row['metric']
            assert not eid or eid in entities or eid in groups
            assert row['dependency']=='0' or metric=='AP'
            assert row['question_natural_language'].endswith('?')
            assert 'unknown' not in row['question_natural_language'].lower()
            if metric == 'EntityCount':
                assert not row['attribute'] and not row['expected_answer']
                scope = bindings[pid]['count_scope']
                assert scope in ('global', 'region')
                assert eid in (groups if scope == 'global' else entities)
                if scope == 'region': local.add(eid)
            else:
                assert row['attribute'] in ('color', 'material', 'texture', 'environment', 'lighting')
                assert row['category_detailed'] == row['attribute']
                assert row['expected_answer'] in ({'no'} if metric == 'SL' else {'yes', 'no'})
            if metric == 'SL':
                assert row['attribute'] == 'color' and row['applies_to'] == 'AC'
                assert set(bindings.get(pid, {}).get('cells', [])) <= {'P0_AC', 'P1_AC', 'P2_AC'}
            if metric == 'AP':
                assert eid in entities
                assert ('throughout the sequence?' in row['question_natural_language']) == (row['dependency']!='0')
        original = read(paths.GT / f'{case}/gt.json')
        assert {e['entity_id'] for e in original['entities']} <= local
        if case == 'F03':
            assert len(groups['dominoes']) == 232
            assert len(local) == 16


@requires_data
def test_question_provenance_and_frozen_sources():
    provenance = read(CATALOG / 'provenance.json')
    contrast_source = provenance['factual_contrasts']
    assert hashlib.sha256(paths.resolve(contrast_source['path']).read_bytes()).hexdigest() == contrast_source['sha256']
    rows = questions()
    for case in CASES:
        for source in provenance['sources'][case].values():
            path = paths.resolve(source['path'])
            assert hashlib.sha256(path.read_bytes()).hexdigest() == source['sha256']
        refs = provenance['questions'][case]
        assert set(refs) == {r['proposition_id'] for r in rows if r['item_id'] == case}
        for row in (r for r in rows if r['item_id'] == case):
            ref = refs[row['proposition_id']]
            assert ref['sources']
            if case != 'A01' and row['metric'] == 'SL':
                assert ref['source_color'] != ref['target']
                assert ref['source_color'] in row['question_natural_language']
            elif case != 'A01' and row['metric'] in ('RF', 'AP'):
                assert 'prompt' in ref['sources']


def test_all_original_rows_are_preserved_and_binary_metrics_are_balanced():
    rows = questions()
    contrasts = read(CATALOG / 'factual-contrasts.json')
    provenance = read(CATALOG / 'provenance.json')
    original_count = 0
    for case in CASES:
        scene = [r for r in rows if r['item_id'] == case]
        refs = provenance['questions'][case]
        originals = [dict(r,dependency='0') for r in scene if 'paired_with' not in refs[r['proposition_id']] and refs[r['proposition_id']].get('stage')!='presence']
        frozen = contrasts['original_catalog']['per_scene'][case]
        assert len(originals) == frozen['questions']
        assert hashlib.sha256(json.dumps(originals, sort_keys=True, ensure_ascii=False).encode()).hexdigest() == frozen['rows_sha256']
        assert [dict(r,dependency='0') for r in scene[:len(originals)]] == originals
        original_count += len(originals)
        for metric in ('RF', 'AP'):
            answers = Counter(r['expected_answer'] for r in scene if r['metric'] == metric)
            assert answers['yes'] == answers['no'] > 0
        assert not any('paired_with' in refs[r['proposition_id']] for r in scene if r['metric'] in ('SL', 'EntityCount'))
    assert original_count == contrasts['original_catalog']['questions'] == 4013


def test_factual_contrasts_keep_subject_scope_and_persistence():
    rows = questions()
    contrasts = read(CATALOG / 'factual-contrasts.json')
    provenance = read(CATALOG / 'provenance.json')
    for case in CASES:
        scene = {r['proposition_id']: r for r in rows if r['item_id'] == case}
        refs = provenance['questions'][case]
        paired = []
        for pid, ref in refs.items():
            if 'paired_with' not in ref:
                continue
            original, negative = scene[ref['paired_with']], scene[pid]
            paired.append(ref['paired_with'])
            assert original['metric'] in ('RF', 'AP') and original['expected_answer'] == 'yes'
            assert negative['expected_answer'] == 'no'
            assert int(pid) > int(ref['paired_with'])
            changed={'proposition_id','question_natural_language','tuple','expected_answer','dependency'}
            assert {k:v for k,v in negative.items() if k not in changed}=={k:v for k,v in original.items() if k not in changed}
            assert negative['question_natural_language'].count('not')==1
            assert negative['question_natural_language'].replace('not ','')==original['question_natural_language']
            assert not re.search(r'\b(?:have|keep|remain|stay) \*\*not\*\*',negative['question_natural_language'])
            assert ref['target']==ref['accepted_target']
            if original['dependency']!='0':
                parent=scene[negative['dependency']]
                assert parent['dependency']=='0'
                assert parent['metric']=='AP'
                assert all(parent[k]==negative[k] for k in ('entity_id','attribute','tuple','expected_answer'))
        assert Counter(paired) == Counter(r['proposition_id'] for r in scene.values() if r['metric'] in ('RF', 'AP') and r['expected_answer'] == 'yes')


def test_reviewed_alternatives_assert_incompatible_visible_properties():
    contrasts = read(CATALOG / 'factual-contrasts.json')['contrasts']
    assert contrasts['material']['made of coated glass'] == 'made of coated wood'
    assert contrasts['material']['made of opaque glass'] == 'made of opaque steel'
    assert contrasts['material']['made of corrugated cardboard'] == 'made of corrugated steel'
    assert contrasts['material']['black rubber'] == 'black ceramic'
    assert contrasts['material']['made of black rubber'] == 'made of black ceramic'
    assert contrasts['material']['kraft-brown corrugated cardboard'] == 'kraft-brown corrugated steel'
    assert contrasts['texture']['smooth'] == 'rough and deeply pitted'
    assert contrasts['texture']['braided'] == 'smooth and featureless'
    assert contrasts['texture']['woven cotton'] == 'smooth and featureless'
    assert contrasts['texture']['showing fine oak grain'] == 'smooth and uniform'
    assert contrasts['texture']['matte plaster'] == 'glossy plaster'
    assert contrasts['texture']['sheer linen'] == 'thick opaque velvet'
    assert contrasts['texture']['open-cell porous'] == 'a smooth solid block'
    assert contrasts['environment']['opaque navy'] == 'opaque vivid orange'
    assert contrasts['lighting']['soft'] == 'hard and strongly directional'
    assert contrasts['lighting']['neutral in color'] == 'strongly red-tinted'


def test_ambiguous_prompt_phases_do_not_enforce_hidden_target_colors():
    rows = questions()
    for case, token in [('A05', 'warm ivory or chestnut brown'), ('A07', 'scarlet red, cobalt blue, or golden yellow'), ('B01', 'black or white')]:
        palette = [r for r in rows if r['item_id'] == case and r['metric'] == 'AP' and r['dependency']!='0' and token in r['question_natural_language']]
        assert palette
        assert all('same' in r['question_natural_language'] for r in palette)


def test_actual_source_cells_override_stale_metadata():
    rows = questions()
    b03 = [r for r in rows if r['item_id'] == 'B03' and r['metric'] == 'SL' and 'pin' in r['entity_id']]
    assert len(b03) == 1 and b03[0]['entity_id'] == 'B03 | pin 01'
    assert 'magenta' in b03[0]['question_natural_language']
    config = read(paths.PROTOCOL / 'questions/B03.json')
    assert config['qa']['bindings']['B03'][b03[0]['proposition_id']]['cells'] == ['P2_AC']
    f04 = next(r for r in rows if r['item_id'] == 'F04' and r['metric'] == 'SL' and r['entity_id'] == 'Parked upper container')
    config = read(paths.PROTOCOL / 'questions/F04.json')
    by_cell = config['qa']['bindings']['F04'][f04['proposition_id']]['question_by_cell']
    assert 'oxide red' in by_cell['P0_AC']
    assert 'light gray' in by_cell['P1_AC'] and 'light gray' in by_cell['P2_AC']
    for case in ['A04', 'B06', 'C02', 'F04']:
        config = read(paths.PROTOCOL / f'questions/{case}.json')
        for row in (r for r in rows if r['item_id'] == case and r['metric'] == 'SL'):
            if any(x in row['question_natural_language'] for x in ["boat's sails", "boat's canopy", 'container end door']) or (case == 'B06' and 'car' not in row['entity_id'].lower()):
                assert 'P0_AC' not in config['qa']['bindings'][case][row['proposition_id']]['cells']
    audit = read(CATALOG / 'source-material-audit.json')
    assert len(audit['sources']) == 144
    for source in audit['sources']:
        assert source['sha256'] == source['sha256_at_transfer']


def test_all_1391_targets_have_both_polarities_and_one_anchor():
    path=CATALOG/'questions.csv'
    rows=list(csv.DictReader(path.open()))
    persistence=[r for r in rows if r['metric']=='AP' and r['dependency']!='0']
    assert len(persistence)==2782
    assert sum(r['expected_answer']=='yes' for r in persistence)==1391
    lookup={(r['item_id'],r['proposition_id']):r for r in rows}
    for r in persistence:
        parent=lookup[r['item_id'],r['dependency']]
        assert parent['dependency']=='0' and parent['metric']=='AP'
        assert all(parent[k]==r[k] for k in ('tuple','entity_id','attribute','expected_answer','applies_to'))
        assert 'throughout' not in parent['question_natural_language']
        assert 'throughout' in r['question_natural_language']
