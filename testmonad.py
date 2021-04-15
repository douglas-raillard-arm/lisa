#! /usr/bin/env python3

import uuid
import builtins
import sys
from operator import attrgetter
from collections.abc import MutableMapping

class MonadBind:
    def __init__(self, ns, name, val):
        self.ns = ns
        self.name = name
        self.val = val

    def __lshift__(self, x):
        self.val = x
        return self


class MonadMeta(type):
    @classmethod
    def __prepare__(metacls, cls_name, bases, **kwargs):

        # Get hold onto the parent namespace, so that the class can access
        # names in enclosing scopes
        parent_frame = sys._getframe(1)
        enclosing_ns = {
            **vars(parent_frame.f_globals['__builtins__']),
            **parent_frame.f_locals,
            **parent_frame.f_globals,
        }

        # Decorate each method when it is bound to its name in the class'
        # namespace, so that other methods can use e.g. undecided_filter
        # If we do that from __new__, the decoration will happen after all
        # methods are defined, just before the class object is created.
        class NS(MutableMapping, dict):
            def __init__(self):
                self._ns = {}

            def __getitem__(self, key):
                try:
                    val = self._ns[key]
                except KeyError:
                    try:
                        val = enclosing_ns[key]
                    except KeyError:
                        x = MonadBind(self, key, None)
                        self._ns[key] = x
                        return x

                return val

            def __setitem__(self, key, val):
                try:
                    existing = self._ns[key]
                except KeyError:
                    self._ns[key] = val
                else:
                    if isinstance(existing, MonadBind):
                        existing.val = val
                    else:
                        self._ns[key] = val

            def __delitem__(self, key):
                del self._ns[key]

            def __iter__(self):
                return iter(self._ns)

            def __len__(self):
                return len(self._ns)

        return NS()


class Monad(metaclass=MonadMeta):
    pass

# class X(Monad):
#     print('hello')
#     print(a)
#     a = 'world'
#     print(a.val)


import ast
stmt = """
def foo(arg1, arg2):
    x = 1
    y = 2
    if y:
        del y
        if x:
            return x + arg1 + arg2
    else:
        with open('bar') as f:
            return 55

foo(1, 2) + 3

"""
tree = ast.parse(stmt)
# print(ast.dump(tree))

def parse(string):
    return ast.parse(string).body[0].value

class RewriteStmt(ast.NodeTransformer):
    def __init__(self):
        self._cnt = 0
        self._ctx = 'ctx'
        self._stack = []

    def _new_name(self, prefix=''):
        self._cnt += 1
        return f'{prefix}_{self._cnt}'
        return name

    def visit(self, node):
        if isinstance(node, ast.FunctionDef):
            stack = self._stack.copy()
            dump_stack = True
        else:
            dump_stack = False
        node = super().visit(node)
        orig_node=node
        if dump_stack:
            self._stack = stack

        self._modified = True
        if isinstance(node, ast.Name):
            node = parse(f'{self._ctx}["{node.id}"]')
        elif isinstance(node, ast.Return):
            node = ast.Expr(ast.Call(
                func=ast.Name(id='return_f', ctx=ast.Load()),
                args=[node.value],
                keywords=[],
            ))
        elif isinstance(node, ast.stmt):
            name = self._new_name('stmt')
            lazy_stmt = ast.FunctionDef(
                name=name,
                body=[node],
                args=ast.arguments(
                    posonlyargs=[],
                    kwonlyargs=[],
                    args=[],
                    defaults=[],
                ),
                decorator_list=[],
            )

            stmt_call = ast.Call(
                func=ast.Name(
                    id=name,
                    ctx=ast.Load()
                ),
                args=[],
                keywords=[],
                starargs=[],
                kwargs=[]
            )

            # node = [lazy_stmt, ast.Expr(stmt_call)]
            node=lazy_stmt
            self._stack.append(lazy_stmt)

        if dump_stack:
            node = [node, ast.Expr(ast.Constant(list(map(attrgetter('name'), stack))))]

        return node

    # def lazyfy(self, node):
    #     node = self.visit(node)
    #     return ast.Module(body=node, type_ignores=[])

new_tree = ast.fix_missing_locations(RewriteStmt().visit(tree))
# print(ast.dump(new_tree))
print(ast.unparse(new_tree))
