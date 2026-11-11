"""Contract-driven entity selection must not double-count geometry."""
import unittest
from types import SimpleNamespace

from worlddraft_eval.gt.visible import NoUsableEntities, collect_entities, letterbox_transform


class Object:
    def __init__(self, name, kind="MESH", parent=None, **properties):
        self.name, self.type, self.parent = name, kind, parent
        self.properties, self.children = properties, []
        if parent:
            parent.children.append(self)

    def get(self, key, default=None):
        return self.properties.get(key, default)

    @property
    def children_recursive(self):
        return [descendant for child in self.children for descendant in [child, *child.children_recursive]]


def contract(pattern, entities=()):
    return {"case": "A01", "gt_root_pattern": pattern,
            "attributes": [{"entities": list(entities)}] if entities else []}


class ContractEntityTests(unittest.TestCase):
    def test_shared_identifier_children_belong_to_selected_root(self):
        root = Object("Boat 01", "EMPTY", entity_id="boat_01")
        hull = Object("hull", parent=root, entity_id="boat_01")
        sail = Object("sail", parent=root, entity_id="boat_01")
        result = collect_entities(SimpleNamespace(objects=[root, hull, sail]), "A01",
                                  contract("boat_01", ["boat_01"]))
        self.assertEqual(len(result), 1)
        self.assertEqual([o.name for o in result[0]["objects"]], ["hull", "sail"])

    def test_independent_part_can_be_scored_under_its_object_name(self):
        root = Object("vehicle", "EMPTY", entity_id="vehicle_01")
        band = Object("colored band", parent=root, entity_id="vehicle_01")
        body = Object("gray body", parent=root, entity_id="vehicle_01")
        result = collect_entities(SimpleNamespace(objects=[root, band, body]), "A01",
                                  contract("colored band", ["colored band"]))
        self.assertEqual(result[0]["entity_id"], "colored band")
        self.assertEqual(result[0]["objects"], [band])

    def test_missing_contract_entity_cannot_silently_reduce_denominator(self):
        root = Object("object 1")
        with self.assertRaisesRegex(NoUsableEntities, "absent"):
            collect_entities(SimpleNamespace(objects=[root]), "A01",
                             contract("object .*", ["object 1", "object 2"]))

    def test_non_renderable_roots_explicitly_skip(self):
        with self.assertRaisesRegex(NoUsableEntities, "no renderable"):
            collect_entities(SimpleNamespace(objects=[Object("point", "EMPTY")]), "A01",
                             contract("point"))

    def test_portrait_letterbox_uses_actual_target_dimensions(self):
        result = letterbox_transform((360, 640), (480, 832))
        self.assertEqual(result["resized_size"], [468, 832])
        self.assertEqual(result["offset_xy"], [6, 0])


if __name__ == "__main__":
    unittest.main()
