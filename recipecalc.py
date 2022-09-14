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
        ret = self.copy()
        for k in ret:
            ret[k] *= i
        return ret

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

    def __hash__(self):
        return hash(str(self))


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
    print(inventories)
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

        def __repr__(self):
            return f"<Recipe {self.consumes} w/ {self.requires} = {self.produces}>"

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
            return self.start_inventory + self.produces - self.consumes

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

    class RecursiveRecipeError(Exception):
        pass

    def __init__(self, data={}):
        self.load(data)

    def load(self, data):
        self.recipes = [
            self.Recipe.fromdict(d)
            for d in data.get("recipes", {})
        ]

    def pickBestPath(self, paths):
        return paths[0]

    # @functools.lru_cache()
    def _genRecipes(self, target, target_count=1, stack=tuple(), inventory=None):
        """Generate a CraftingStep for each way to craft the target.

        Args:
            target (Item): The target item ID
            target_count (int, optional): The number of targets to craft, default 1
            inventory (optional): An existing inventory to draw items from if applicable.
            stack (TYPE, optional): Internal, loop prevention

        Yields:
            CraftingStep
        """
        def stackprint(*a, **k):
            print(" " * 2 * len(stack), *a, **k)

        if not inventory:
            inventory = Counter()

        target_counter = Counter([target] * target_count)

        has_matched = False
        stackprint("Generating crafting step for", target_counter, "inventory", inventory)

        if target in inventory:
            existing_count = inventory[target]
            remaining_count = target_count - existing_count
            yield self.InventoryExistingCraftingStep(
                produces=Counter([target] * existing_count),
                start_inventory=inventory
            )
            if remaining_count == 0:
                return  # Early exit
            else:
                target_count = remaining_count
        for recipe in filter(lambda r: target in r.produces, self.recipes):
            # How many times do we need to craft this recipe to get sufficient output?
            recipe_iterations = math.ceil(target_count / recipe.produces[target])
            stackprint("matched", recipe, "will need", recipe_iterations)

            # Loop detection
            if any(target in r2.produces and r2.produces[target] >= target_count for r2 in stack):
                # Already trying to fufill this!
                stackprint("Found recursive recipe set:")
                stackprint("Recipe", recipe, "produces an existing target item")
                stackprint("Target", target_counter, "in", [r2.produces for r2 in stack])
                stackprint(stack, recipe)
                raise self.RecursiveRecipeError((*stack, recipe))

            # Compute cost to produce all the requirements of the recipe
            prereqs = []
            prereqs_by_consumement = {}
            try:
                for consumement in recipe.consumes:
                    need_count = recipe.consumes[consumement]
                    # Need to flatten this to see our working inventory.
                    # Maybe look at "lowest common denominator" of all the different possible prereq cases?
                    prereqs_by_consumement[consumement] = self.genRecipes(
                        consumement,
                        target_count=need_count,
                        stack=(*stack, recipe),
                        inventory=inventory
                    )
                    # inventory_input = self.pickBestPath(prereqs_by_consumement[consumement]).inventory
                    # stackprint("Setting inventory to resolved prereq", inventory_input)
                    # inventory = inventory_input
            except AssertionError:
                print(prereqs_by_consumement)
                raise
            except self.RecursiveRecipeError:
                stackprint("Prerequisites are recursive")
                stackprint("skipping recipe", recipe, "in favor of axiom")
                yield self.AxiomaticCraftingStep(
                    produces=target_counter,
                    start_inventory=inventory
                )
                continue

            # Collapse prereqs into paths
            prereqs = [*itertools.product(*prereqs_by_consumement.values())]

            # TODO: Update our inventory with the newly produced items in the prereqs.

            # Yield the CraftingStep for this recipe
            has_matched = True
            yield self.CraftingStep(
                recipe=recipe,
                prereqs=prereqs,
                start_inventory=inventory
            ) * recipe_iterations  # Multiply step by needed count
        if not has_matched:
            # There's no way to craft this item, ergo it's an input requirement.
            stackprint("did not reach", target, "yielding axiom")
            yield self.AxiomaticCraftingStep(
                produces=target_counter,
                start_inventory=inventory
            )

    def genRecipes(self, *args, **kwargs):
        return list(self._genRecipes(*args, **kwargs))

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

def test_long_binary():
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
    assert "\n---\n".join(path.render() for path in rc.genRecipes("11")) == """
- <Requires <2x 00>>
- <Step Craft <2x 00> With <1x inc> = <2x 01>>
- Inventory Counter({'01': 2})
- <Step Craft <2x 01> With <1x add> = <1x 10>>
- Inventory Counter({'10': 1})
- <Requires <1x 00>>
- <Step Craft <1x 00> With <1x inc> = <1x 01>>
- Inventory Counter({'10': 1, '01': 1})
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

def test_multi_path_minecraft():
    rc = RecipeCalc()
    yaml_data = """recipes:
- consumes: ["w"]
  requires: "craft"
  produces: ["p", "p", "p", "p"]
- consumes: ["p", "p", "p", "p", "p", "p", "p", "p"]
  requires: "craft"
  produces: ["chest"]
- consumes: ["w", "w", "w", "w", "w", "w", "w", "w"]
  requires: "craft"
  produces: ["chest", "chest", "chest", "chest"]
"""
    rc.load(yaml.safe_load(yaml_data))

    assert "\n---\n".join(path.render() for path in rc.genRecipes("chest", target_count=4)) == """
- <Requires <8x w>>
- <Step Craft <8x w> With <1x craft> = <32x p>>
- Inventory Counter({'p': 32})
- <Step Craft <32x p> With <1x craft> = <4x chest>>
- Inventory Counter({'chest': 4})
---
- <Requires <8x w>>
- <Step Craft <8x w> With <1x craft> = <4x chest>>
- Inventory Counter({'chest': 4})
""".strip()

def test_recursive():
    rc = RecipeCalc()
    yaml_data = """recipes:
- consumes: "A"
  requires: "rec"
  produces: "B"
- consumes: "B"
  requires: "rec"
  produces: "C"
- consumes: "B"
  requires: "rec"
  produces: "A"
"""
    rc.load(yaml.safe_load(yaml_data))

    # Creating B for C
    #   Creating A for B
    #     Found recipe to make A with B
    #     Skipping that recipe because B is a target in the recipe stack

    generated = rc.genRecipes("C")
    pprint.pprint([r.todict() for r in generated])
    out = "\n---\n".join(path.render() for path in generated)
    print(out)
    assert out == """
- <Requires <1x A>>
- <Step Craft <1x A> With <1x rec> = <1x B>>
- Inventory Counter({'B': 1})
- <Step Craft <1x B> With <1x rec> = <1x C>>
- Inventory Counter({'C': 1})
""".strip()

def test_rabbits():
    rc = RecipeCalc()
    yaml_data = """recipes:
- consumes: ["r", "r"]
  requires: "breed"
  produces: ["r", "r", "r"]
"""
    rc.load(yaml.safe_load(yaml_data))

    generated = rc.genRecipes("r", target_count=5)
    pprint.pprint([r.todict() for r in generated])
    out = "\n---\n".join(path.render() for path in generated)
    print(out)
    assert out == """
- <Requires <2x r>>
- <Step Craft <2x r> With <1x breed> = <3x r>>
- Inventory Counter({'r': 3})
- <Step Craft <2x r> With <1x breed> = <3x r>>
- Inventory Counter({'r': 4})
- <Step Craft <2x r> With <1x breed> = <3x r>>
- Inventory Counter({'r': 5})
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

    generated = rc.genRecipes("remainder")
    pprint.pprint([r.todict() for r in generated])
    out = "\n---\n".join(path.render() for path in generated)
    print(out)
    assert out == """
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