#!/usr/bin/env python3

import yaml
import pprint
import functools
import collections
import math
import itertools


class Counter(collections.Counter):
    def __repr__(self):
        return '<' + ', '.join(f"{f'{c}x ' if (c > 1 or True) else ''}{k}" for k, c in self.items()) + '>'

    def __mul__(self, i):
        for k in self:
            self[k] *= i
        return self

    def __sub__(self, other):
        ''' Subtract count, and throw an error if we try to subtract below zero.'''
        if not isinstance(other, Counter):
            return NotImplemented
        result = Counter()
        for elem, count in self.items():
            newcount = count - other[elem]
            assert newcount >= 0
            if newcount > 0:
                result[elem] = newcount
        for elem, count in other.items():
            if elem not in self:
                assert count < 0
                result[elem] = 0 - count
        return result
    # todo mult


def trycall(obj, fn):
    if isinstance(obj, list):
        return [trycall(o2, fn) for o2 in obj]
    elif hasattr(obj, fn):
        return getattr(obj, fn)()
    else:
        return obj


class HackableFieldAbc():
    fields = []

    def todict(self):
        return {key: trycall(getattr(self, key), 'todict') for key in self.fields}

    def __repr__(self):
        return f"<{type(self).__name__} {self.todict()!r}>"

def lowestCounterDenominator(inventories):
    # print(inventories)
    # all_keys =
    return inventories[0]

class RecipeCalc():
    class Item():
        def __init__(self, name):
            self.name = name

        def __repr__(self):
            props = {"name": self.name}
            return f"<{type(self).__name__} {props!r}>"

    class Recipe(HackableFieldAbc):
        fields = ["consumes", "requires", "produces"]

        @classmethod
        def fromdict(cls, data):
            return cls(**{key: data.get(key, []) for key in cls.fields})

        def __init__(self, consumes, requires, produces):
            # self.consumes = consumes if isinstance(consumes, list) else [consumes]
            # self.requires = requires if isinstance(requires, list) else [requires]
            # self.produces = produces if isinstance(produces, list) else [produces]
            self.consumes = Counter(consumes) if isinstance(consumes, list) else Counter([consumes])
            self.requires = Counter(requires) if isinstance(requires, list) else Counter([requires])
            self.produces = Counter(produces) if isinstance(produces, list) else Counter([produces])

    class CraftingStep(HackableFieldAbc):
        fields = ["prereqs", "consumes", "requires", "produces", "inventory"]

        def __init__(self, recipe, prereqs, start_inventory):
            self.consumes = recipe.consumes
            self.requires = recipe.requires
            self.produces = recipe.produces
            self.start_inventory = start_inventory
            self.prereqs = prereqs
            # for paths in self.prereqs:
            #     for path in paths:
            #         print(path.produces, "<=", self.consumes)
            # assert path.produces == self.consumes

        @property
        def inventory(self):
            # Scale our modifications with __mul__ without scaling other inventory items
            return self.start_inventory - self.consumes + self.produces

        def __eq__(self, other):
            return str(self) == str(other)

        def __mul__(self, i):
            self.consumes *= i
            self.produces *= i
            self.prereqs = [
                [r * i for r in path]
                for path in self.prereqs
            ]
            return self

        def __str__(self):
            return f"<Step Craft {self.consumes} With {self.requires} = {self.produces}>"

        def render(self, partial=False):
            ret = ''
            best_prereq = self.prereqs[0] if self.prereqs else []
            for r in best_prereq:
                # ret += f"{r.render(partial=True)}\n"
                ret += f"{r.render()}\n"
            ret += f"- {str(self)}"
            if partial is False and self.inventory:
                ret += f"\n- Inventory {str(self.inventory)}"
            return ret

    class AxiomaticCraftingStep(CraftingStep):
        fields = ["produces"]

        def __init__(self, produces, start_inventory):
            self.produces = produces
            self.consumes = Counter()
            self.start_inventory = start_inventory

        def __mul__(self, i):
            self.produces *= i
            # print(f"*= {i} = {self.produces}")
            return self

        def render(self, partial=False):
            return f"- {str(self)}"

        def __str__(self):
            return f"<Requires {self.produces}>"

    class InventoryExistingCraftingStep(AxiomaticCraftingStep):
        def __str__(self):
            return f"<Use existing {self.produces}>"

    class NoRecipeError(Exception):
        pass

    def __init__(self, data={}):
        self.load(data)

    def load(self, data):
        self.recipes = [
            self.Recipe.fromdict(d)
            for d in data.get("recipes", {})
        ]

    # @functools.lru_cache()
    def genRecipes(self, target, stack=tuple(), inventory=None):
        if not inventory:
            inventory = Counter()
        stackprint = lambda *a, **k: print(" " * 2*len(stack), *a, **k)
        has_matched = False
        stackprint("genRecipe", target)
        if target in inventory:
            yield self.InventoryExistingCraftingStep(
                produces=Counter([target]),
                start_inventory=inventory
            )
            return
        for recipe in self.recipes:
            try:
                if target in recipe.produces:
                    stackprint("matched", recipe)
                    if any(target in r.consumes for r in stack):
                        # Already trying to fufill this!
                        stackprint("Found recursive recipe set")
                        stackprint(stack, recipe)
                        # yield self.AxiomaticCraftingStep(
                        #     produces=Counter([target]),
                        #     start_inventory=inventory
                        # )
                        continue
                    prereqs = []
                    prereqs_by_consumement = {}
                    for consumement in recipe.consumes:
                        need_count = recipe.consumes[consumement]
                        # Need to flatten this to see our working inventory. Maybe look at "lowest common denominator" of all the different possible prereq cases?
                        prereqs_by_consumement[consumement] = [
                            r * math.ceil(need_count / r.produces[consumement])
                            for r in self.genRecipes(consumement, stack=(*stack, recipe), inventory=inventory)
                        ]
                    # All possible sets of crafting steps that satisify all the prerequisites.
                    # A: x, y
                    # B: z
                    # [[x, z], [y, z]]
                    prereqs = [*itertools.product(*prereqs_by_consumement.values())]
                    has_matched = True
                    # Need to flatten this to see our working inventory. Maybe look at "lowest common denominator" of all the different possible prereq cases?
                    # print(prereqs)
                    step_inventory = lowestCounterDenominator([[recipe.inventory for recipe in path] for path in prereqs])
                    step_inventory = prereqs[0][0].inventory
                    yield self.CraftingStep(
                        recipe=recipe,
                        prereqs=prereqs,
                        start_inventory=step_inventory
                    )
            except AssertionError:
                continue
        if not has_matched:
            stackprint("did not reach", target)
            yield self.AxiomaticCraftingStep(
                produces=Counter([target]),
                start_inventory=inventory
            )

def test_basic_binary():
    rc = RecipeCalc()
    yaml_data = """recipes:
- consumes: "00"
  requires: "inc"
  produces: "01"
- consumes: ["10", "01"]
  requires: "and"
  produces: "11"
- consumes: ["01", "01"]
  requires: "add"
  produces: "10"
"""
    rc.load(yaml.safe_load(yaml_data))

    assert "\n---\n".join(path.render() for path in rc.genRecipes("00")) == "- <Requires <1x 00>>"

    assert "\n---\n".join(path.render() for path in rc.genRecipes("01")) == """
- <Requires <1x 00>>
- <Step Craft <1x 00> With <1x inc> = <1x 01>>
- Inventory Counter({'01': 1})""".strip()

    assert "\n---\n".join(path.render() for path in rc.genRecipes("10")) == """
- <Requires <2x 00>>
- <Step Craft <2x 00> With <1x inc> = <2x 01>>
- Inventory Counter({'01': 2})
- <Step Craft <2x 01> With <1x add> = <1x 10>>
- Inventory Counter({'10': 1})""".strip()

    assert "\n---\n".join(path.render() for path in rc.genRecipes("11")) == """
- <Requires <2x 00>>
- <Step Craft <2x 00> With <1x inc> = <2x 01>>
- Inventory Counter({'01': 2})
- <Step Craft <2x 01> With <1x add> = <1x 10>>
- Inventory Counter({'10': 1})
- <Requires <2x 00>>
- <Step Craft <2x 00> With <1x inc> = <2x 01>>
- Inventory Counter({'01': 2})
- <Step Craft <1x 10, 1x 01> With <1x and> = <1x 11>>
- Inventory Counter({'11': 1})
""".strip()


def test_multi_path():
    rc = RecipeCalc()
    yaml_data = """recipes:
- consumes: "road1"
  requires: "leads"
  produces: "rome"
- consumes: "road2"
  requires: "leads"
  produces: "rome"
"""
    rc.load(yaml.safe_load(yaml_data))

    assert "\n---\n".join(path.render() for path in rc.genRecipes("rome")) == """
- <Requires <1x road1>>
- <Step Craft <1x road1> With <1x leads> = <1x rome>>
- Inventory Counter({'rome': 1})
---
- <Requires <1x road2>>
- <Step Craft <1x road2> With <1x leads> = <1x rome>>
- Inventory Counter({'rome': 1})""".strip()

def test_recursive():
    rc = RecipeCalc()
    yaml_data = """recipes:
- consumes: "A"
  requires: "rec"
  produces: "B"
- consumes: "B"
  requires: "rec"
  produces: "A"
"""
    rc.load(yaml.safe_load(yaml_data))

    generated = list(rc.genRecipes("A"))
    out = "\n---\n".join(path.render() for path in generated)
    pprint.pprint([r.todict() for r in generated])
    print(out)
    assert out == """
- <Requires <1x B>>
- <Step Craft <1x B> With <1x rec> = <1x A>>
- Inventory Counter({'A': 1})
""".strip()

def test_rabbits():
    rc = RecipeCalc()
    yaml_data = """recipes:
- consumes: ["r", "r"]
  requires: "breed"
  produces: ["r", "r", "r"]
- consumes: ["r", "r", "r", "r", "r"]
  requires: "name"
  produces: ["r5"]
"""
    rc.load(yaml.safe_load(yaml_data))

    assert "\n---\n".join(path.render() for path in rc.genRecipes("r5")) == """
- <Requires <4x r>>
- <Step Craft <4x r> With <1x breed> = <6x r>>
- Inventory Counter({'r': 6})
- <Step Craft <5x r> With <1x name> = <1x r5>>
- Inventory Counter({'r': 1, 'r5': 1})
""".strip()


def test_minecraft_recursive():
    rc = RecipeCalc()
    yaml_data = """recipes:
- consumes: ["w"]
  requires: "craft"
  produces: ["p", "p", "p", "p"]
- consumes: ["p", "p", "p", "p", "p", "p"]
  requires: "craft"
  produces: ["door"]
- consumes: ["p", "p", "p", "p", "p", "p", "p", "p"]
  requires: "craft"
  produces: ["chest"]
"""
    rc.load(yaml.safe_load(yaml_data))

    assert "\n---\n".join(path.render() for path in rc.genRecipes("chest")) == """
- <Requires <2x w>>
- <Step Craft <2x w> With <1x craft> = <8x p>>
- Inventory Counter({'p': 8})
- <Step Craft <8x p> With <1x craft> = <1x chest>>
- Inventory Counter({'chest': 1})
""".strip()

    assert "\n---\n".join(path.render() for path in rc.genRecipes("door")) == """
- <Requires <2x w>>
- <Step Craft <2x w> With <1x craft> = <8x p>>
- Inventory Counter({'p': 8})
- <Step Craft <6x p> With <1x craft> = <1x door>>
- Inventory Counter({'p': 2, 'door': 1})
""".strip()

def test_remainders():
    rc = RecipeCalc()
    yaml_data = """recipes:
- consumes: ["X"]
  requires: "split"
  produces: ["x", "z"]
- consumes: ["Y"]
  requires: "split"
  produces: ["y", "z"]
- consumes: ["x", "y", "z", "z"]
  requires: "name"
  produces: ["remainder"]
"""
    rc.load(yaml.safe_load(yaml_data))

    assert "\n---\n".join(path.render() for path in rc.genRecipes("remainder")) == """
- <Requires <1x X>>
- <Step Craft <2x X> With <1x split> = <2x x, 2x z>>
- Inventory Counter({'x': 2, 'z': 2})
- <Requires <1x Y>>
- <Step Craft <2x Y> With <1x split> = <2x y, 2x z>>
- Inventory Counter({'x': 2, 'y': 2, 'z': 4})
- <Step Craft <1x x, 1x y, 2x z> With <1x name> = <1x remainder>>
- Inventory Counter({'x': 1, 'remainder': 1})
""".strip()


if __name__ == "__main__":
    test_basic_binary()
    test_recursive()
    test_rabbits()
    test_minecraft_recursive()
    test_remainders()