#!/usr/bin/env python3

import yaml
import pprint
import functools
import collections
import math
import itertools

def return_subtype(*method_names):
    def decorator(cls):
        for name in method_names:
            method = getattr(cls, name)
            def wrap(*args):
                return cls(method(*args))
            setattr(cls, name, wrap)
        return cls
    return decorator


@return_subtype('__add__')
class Counter(collections.Counter):
    def __repr__(self):
        return '[' + ', '.join(f"{f'{c}x ' if (c > 1 or True) else ''}{k}" for k, c in self.items()) + ']'

    def __mul__(self, i):
        ret = self.copy()
        for k in ret:
            ret[k] *= i
        return ret

    def __sub__(self, other):
        ''' Subtract count, and throw an error if we try to subtract below zero.'''
        if not isinstance(other, self.__class__):
            return NotImplemented
        result = self.__class__()
        for elem, count in self.items():
            newcount = count - other[elem]
            assert newcount >= 0, f"{other} not a subset of {self}"
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
            return f"<Recipe {self.produces} from {self.consumes} with {self.requires}>"

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
        fields = ["prereqs", "consumes", "requires", "produces", "start_inventory"]

        def __init__(self, consumes, requires, produces, prereqs, start_inventory=Counter()):
            self.consumes = consumes
            self.requires = requires
            self.produces = produces
            self.start_inventory = start_inventory
            self.prereqs = prereqs
            # for paths in self.prereqs:
            #     for path in paths:
            #         print(path.produces, "<=", self.consumes)
            # assert path.produces == self.consumes

        @classmethod
        def fromRecipe(cls, recipe, **kwargs):
            return cls(consumes=recipe.consumes, requires=recipe.requires, produces=recipe.produces, **kwargs)

        @property
        def inventory(self):
            # Scale our modifications with __mul__ without scaling other inventory items
            prereq_inventories = [prereq.inventory for prereq in self.prereqs]
            ret = sum(prereq_inventories, self.start_inventory) + self.produces - self.consumes
            # print(
            #     self.start_inventory,
            #     "+", [prereq.inventory for prereq in self.prereqs],
            #     "+", self.produces,
            #     "-", self.consumes,
            #     "=", ret)
            return ret

        def __eq__(self, other):
            return str(self) == str(other)

        def __mul__(self, i):
            ret = self.__class__(
                consumes=self.consumes,
                requires=self.requires,
                produces=self.produces,
                prereqs=self.prereqs,
                start_inventory=self.start_inventory
            )
            ret.consumes *= i
            ret.produces *= i
            ret.prereqs = [
                req * i for req in self.prereqs
            ]
            # if i > 1:
            # print(self, "*", i, "=", ret)
            return ret

        def __str__(self):
            return (f"<CraftingStep {self.produces} from {self.consumes} with {self.requires} "
                    f"| {sum([prereq.inventory for prereq in self.prereqs], self.start_inventory)} -> {self.inventory}>")

        def render(self, partial=False):
            ret = ''
            for r in self.prereqs:
                # ret += f"{r.render(partial=True)}\n"
                ret += f"{r.render()}\n"
            ret += f"- {str(self)}"
            # if partial is False and self.inventory:
            #     ret += f"\n- Inventory {str(self.inventory)}"
            return ret

    class AxiomaticCraftingStep(CraftingStep):
        fields = ["produces"]

        def __init__(self, produces, start_inventory):
            self.produces = produces
            self.consumes = Counter()
            self.start_inventory = start_inventory
            self.prereqs = []

        def __mul__(self, i):
            self.produces *= i
            # print(f"*= {i} = {self.produces}")
            return self

        def render(self, partial=False):
            return f"- {str(self)}"

        def __str__(self):
            return f"<Requires {self.produces} | {self.start_inventory} -> {self.inventory}>"

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
        stackprint("Generating crafting step for", target_counter, "w/ starting inventory", inventory)

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

            try:
                for i, consumement in enumerate(recipe.consumes):
                    stackprint("Consumement #", i, consumement, "x", recipe.consumes[consumement], "*", recipe_iterations)
                    next_prereq = self.pickBestPath(self.genRecipes(
                        consumement,
                        target_count=recipe.consumes[consumement],
                        stack=(*stack, recipe),
                        inventory=inventory # sum([prereq.inventory for prereq in prereqs], inventory)
                    ))
                    stackprint("New peer inventory", next_prereq.inventory)
                    # inventory += next_prereq.inventory
                    prereqs.append(next_prereq)
            except self.RecursiveRecipeError:
                stackprint("Prerequisites are recursive")
                stackprint("skipping recipe", recipe, "in favor of axiom")
                yield self.AxiomaticCraftingStep(
                    produces=target_counter,
                    start_inventory=inventory
                )
                continue

            # Yield the CraftingStep for this recipe
            has_matched = True
            stackprint("Starting inventory", inventory)
            stackprint("Total working inventory", [prereq.inventory for prereq in prereqs], sum([prereq.inventory for prereq in prereqs], inventory))
            # stackprint("Prereq inventories", [prereq.inventory for prereq in prereqs])
            # stackprint("Prereqs", prereqs)
            step = self.CraftingStep.fromRecipe(
                recipe=recipe,
                prereqs=prereqs,
                # start_inventory=Counter(),  # Our starting inventory is already in the prereqs
            ) * recipe_iterations  # Multiply step by needed count
            # pprint.pprint(step.todict())
            stackprint("Yielding step", step)
            yield step
        if not has_matched:
            # There's no way to craft this item, ergo it's an input requirement.
            axiom = self.AxiomaticCraftingStep(
                produces=target_counter,
                start_inventory=inventory
            )
            stackprint("did not reach", target, "yielding axiom", axiom)
            yield axiom

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

    assert "\n---\n".join(path.render() for path in rc.genRecipes("00")) == "- <Requires [1x 00] | [] -> [1x 00]>"

    assert "\n---\n".join(path.render() for path in rc.genRecipes("01")) == """
- <Requires [1x 00] | [] -> [1x 00]>
- <CraftingStep [1x 01] from [1x 00] with [1x inc] | [1x 00] -> [1x 01]>""".strip()

    assert "\n---\n".join(path.render() for path in rc.genRecipes("10")) == """
- <Requires [2x 00] | [] -> [2x 00]>
- <CraftingStep [2x 01] from [2x 00] with [1x inc] | [2x 00] -> [2x 01]>
- <CraftingStep [1x 10] from [2x 01] with [1x add] | [2x 01] -> [1x 10]>""".strip()

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
- <Requires [2x 00] | [] -> [2x 00]>
- <CraftingStep [2x 01] from [2x 00] with [1x inc] | [2x 00] -> [2x 01]>
- <CraftingStep [1x 10] from [2x 01] with [1x add] | [2x 01] -> [1x 10]>
- <Requires [1x 00] | [] -> [1x 00]>
- <CraftingStep [1x 01] from [1x 00] with [1x inc] | [1x 00] -> [1x 01]>
- <CraftingStep [1x 11] from [1x 10, 1x 01] with [1x and] | [1x 10, 1x 01] -> [1x 11]>
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
- <Requires [1x road1] | [] -> [1x road1]>
- <CraftingStep [1x rome] from [1x road1] with [1x leads] | [1x road1] -> [1x rome]>
---
- <Requires [1x road2] | [] -> [1x road2]>
- <CraftingStep [1x rome] from [1x road2] with [1x leads] | [1x road2] -> [1x rome]>""".strip()

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
- <Requires [8x w] | [] -> [8x w]>
- <CraftingStep [32x p] from [8x w] with [1x craft] | [8x w] -> [32x p]>
- <CraftingStep [4x chest] from [32x p] with [1x craft] | [32x p] -> [4x chest]>
---
- <Requires [8x w] | [] -> [8x w]>
- <CraftingStep [4x chest] from [8x w] with [1x craft] | [8x w] -> [4x chest]>
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
- <Requires [1x A] | [] -> [1x A]>
- <CraftingStep [1x B] from [1x A] with [1x rec] | [1x A] -> [1x B]>
- <CraftingStep [1x C] from [1x B] with [1x rec] | [1x B] -> [1x C]>
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
- <Requires [2x r] | [] -> [2x r]>
- <CraftingStep [3x r] from [2x r] with [1x breed] | [3x r]>
- <CraftingStep [3x r] from [2x r] with [1x breed] | [4x r]>
- <CraftingStep [3x r] from [2x r] with [1x breed] | [5x r]>
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
- <Requires [2x w] | [] -> [2x w]>
- <CraftingStep [8x p] from [2x w] with [1x craft] | [2x w] -> [8x p]>
- <CraftingStep [1x chest] from [8x p] with [1x craft] | [8x p] -> [1x chest]>
""".strip()

    assert "\n---\n".join(path.render() for path in rc.genRecipes("door")) == """
- <Requires [2x w] | [] -> [2x w]>
- <CraftingStep [8x p] from [2x w] with [1x craft] | [2x w] -> [8x p]>
- <CraftingStep [1x door] from [6x p] with [1x craft] | [8x p] -> [2x p, 1x door]>
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
- <Requires [1x X] | [] -> [1x X]>
- <CraftingStep [2x x, 2x z] from [2x X] with [1x split] | [2x X] -> [2x x, 2x z]>
- <Requires [1x Y] | [] -> [1x Y]>
- <CraftingStep [2x y, 2x z] from [2x Y] with [1x split] | [2x Y] -> [2x y, 2x z]>
- <CraftingStep [1x remainder] from [1x x, 1x y, 2x z] with [1x name] | [3x x, 4x z, 1x y] -> [2x x, 2x z, 1x remainder]>
""".strip()


if __name__ == "__main__":
    test_basic_binary()
    test_recursive()
    test_rabbits()
    test_minecraft_recursive()
    test_remainders()