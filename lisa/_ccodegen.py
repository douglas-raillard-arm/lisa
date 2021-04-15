#! /usr/bin/env python3

import abc
import uuid
import functools
import textwrap
from functools import partial
from operator import attrgetter, itemgetter
from itertools import chain, accumulate
from pathlib import Path

from lisa.utils import SimpleHash, deduplicate
from lisa._assets import ASSETS_PATH

def accumulate_right(xs, f, reverse=False):
    res = accumulate(
        reversed(xs),
        lambda x, y: f(y, x),
    )
    # Since the original iterable has been reversed already for the right-fold,
    # nothing to do
    if reverse:
        return res
    else:
        return reversed(list(res))

class Node:
    def traverse(self, cls=None):
        cls = cls or Node
        check = lambda x: isinstance(x, cls)

        def traverse(obj, visited):
            if obj in visited:
                return
            else:
                visited.add(obj)
                yield obj
                yield from chain.from_iterable(
                    traverse(x, visited)
                    for x in obj.__dict__.values()
                    if check(x)
                )

        if check(self):
            return traverse(self, set())
        else:
            return (_ for _ in ())

    @property
    def used_typs(self):
        typs = set(
            chain.from_iterable(
                x._used_typs
                for x in self.traverse()
                if isinstance(x, Typed)
            )
        )
        children = set(
            chain.from_iterable(
                typ.typ_deps
                for typ in typs
            )
        )
        roots = typs - children

        def sort(typs):
            return sorted(typs, key=attrgetter('name'))

        visited = set()
        def dfs(typ):
            if typ in visited:
                return
            else:
                visited.add(typ)
                yield from chain.from_iterable(
                    map(dfs, sort(typ.typ_deps))
                )
                yield typ

        return chain.from_iterable(
            map(dfs, sort(roots))
        )


    @property
    def used_stmts(self):
        return set(self.traverse(Stmt))


class Typed:

    @property
    def _used_typs(self):
        return {self.typ}

class Var(SimpleHash, Node, Typed):
    def __init__(self, name, typ, *, prog):
        self.name = name
        self.typ = typ

    @property
    def local_ref(self):
        return f'(ctx->{self.name})'

    def get_c_decl(self):
        return f'{self.typ.name} {self.name};'


class UserVar(Var):
    def __init__(self, name, **kwargs):
        super().__init__(
            name=f'__user_{name}',
            **kwargs
        )


class FuncUserParam(Var):
    def __init__(self, name, **kwargs):
        super().__init__(
            name=f'__param_user_{name}',
            **kwargs
        )


class TmpVar(Var):
    def __init__(self, *, prog, **kwargs):
        super().__init__(
            name=f'__tmp_{prog.make_unique_id()}',
            prog=prog,
            **kwargs
        )


class BlackHoleVar(Var):
    def __init__(self, prog):
        super().__init__(
            prog=prog,
            name='blackhole',
            typ=prog.builtin_typs['unit']
        )


class TypBase(SimpleHash, Node, Typed, abc.ABC):
    def __init__(self, name, *, prog, deps=None):
        self.name = name
        self.prog = prog
        self.typ_deps = deps or set()

    def get_c(self):
        # TODO: is this indirection useful ?
        return self._get_c()

    @property
    def _used_typs(self):
        return {self}

    def __str__(self):
        return f'{self.__class__.__qualname__}({self.name})'


class Typ(TypBase, Typed):
    @property
    def ctrl_monad(self):
        return self.prog.ControlMonadOf(self)

    @property
    def _used_typs(self):
        return {
            *super()._used_typs,
            *self.ctrl_monad._used_typs,
        }


class BuiltinTyp(Typ):
    def __init__(self, name, *, prog):
        super().__init__(
            name=f'lisa_{name}_t',
            prog=prog,
        )
        self.iso_name = name

    def _get_c(self):
        return f'typedef {self.iso_name} {self.name};'

    def __eq__(self, other):
        if self.__class__ != other.__class__:
            return False
        else:
            return self.name == other.name

    def __hash__(self):
        return hash(self.name)


class UnitTyp(BuiltinTyp):
    def __init__(self, *, prog):
        super().__init__(
            name='unit',
            prog=prog,
        )
        self.c_val = f'(({self.name}){{}})'

    def _get_c(self):
        return f'typedef struct {{}} {self.name};'


class ControlMonadOf(TypBase):
    def __init__(self, typ, *, prog):
        super().__init__(
            name=f'CTRL_MONAD({typ.name})',
            prog=prog,
            deps=[typ],
        )
        self._typ_param = typ

    def get_c(self):
        typ = self._typ_param
        return f'MAKE_CTRL_MONAD({typ.name}, {typ.name});'


class SuspendedGeneratorOf(Typ):
    def __init__(self, typ, *, prog):
        super().__init__(
            name=f'SUSPENDED_GENERATOR({typ.name})',
            prog=prog,
            deps=[typ],
        )
        self._typ_param = typ

    @property
    def wrapped_typ(self):
        return self._typ_param

    def get_c(self):
        typ = self._typ_param
        return f'MAKE_SUSPENDED_GENERATOR({typ.name}, {typ.name});'


class UserTyp(Typ):
    def __init__(self, name, decl, *, prog):
        self.decl = decl
        super().__init__(
            name=f'__user_{name}',
            prog=prog,
        )

    def _get_c(self):
        return self.decl


class CtxTyp(TypBase):
    def __init__(self, name, variables, *, prog):
        self.variables = variables
        super().__init__(
            name=f'struct {name}',
            prog=prog,
            deps=self._vars_used_typs,
        )

    @property
    def _vars_used_typs(self):
        return set(chain.from_iterable(
            var._used_typs
            for var in self.variables
        )) - {self}

    @property
    def _used_typs(self):
        return {
            *super()._used_typs,
            *self._vars_used_typs,
        }

    def get_c(self):
        return (
            f'MAKE_CTX_BEGIN({self.name})' +
            '\n    ' +
            '\n    '.join(
                var.get_c_decl()
                for var in sorted(self.variables, key=attrgetter('name'))
            ) +
            '\n' +
            f'MAKE_CTX_END({self.name})'
        )


class Expr(SimpleHash, Node, Typed):
    def __init__(self, typ, *, prog):
        if not isinstance(typ, TypBase):
            typ = prog.builtin_typs[typ]

        self.typ = typ
        self.prog = prog

    @abc.abstractmethod
    def get_c(self, stmt):
        pass

    def __add__(self, expr):
        return self.prog.CompoundExpr(self, expr)


class ControlExpr(Expr):
    def __init__(self, name, typ, val=None, *, prog):
        self.name = name
        self.val = val
        super().__init__(
            typ=typ,
            prog=prog
        )

    def get_c(self, stmt):
        val = self.val
        val = '' if val is None else f', {val}'
        return f'{self.name}({stmt.name}{val})'


class YieldExpr(Expr):
    def __init__(self, typ, val, *, prog):
        self.val = val
        super().__init__(
            typ=SuspendedGeneratorOf(typ, prog=prog),
            prog=prog
        )

    def get_c(self, stmt):
        val = self.val
        if val is None:
            return f'GeneratorFinish({stmt.name})'
        else:
            return f'GeneratorYield({stmt.name}, {self.val})'


class SimpleExpr(ControlExpr):
    def __init__(self, val, typ=None, *, prog):
        if val is None:
            name = 'Return'
            typ = prog.builtin_typs['unit']
            val = typ.c_val
        else:
            name = 'Return'
            if typ is None:
                raise ValueError(f'typ=None is not allowed for val != None ({val})')

        super().__init__(
            name=name,
            typ=typ,
            val=val,
            prog=prog,
        )


class CompoundExpr(Expr):
    def __init__(self, expr1, expr2, prog):
        self.expr1 = expr1
        self.expr2 = expr2
        super().__init__(
            typ=expr2.typ,
            prog=prog,
        )

    def get_c(self, stmt):
        return '({' + '; '.join(map(
            lambda expr: '({ ' + expr.get_c(stmt=stmt) + ' ;})',
            (self.expr1, self.expr2)
        )) + ';})'


class RawExpr(Expr):
    def __init__(self, code, typ, *, prog):
        self.code = code
        super().__init__(
            typ=typ,
            prog=prog,
        )

    def get_c(self, stmt):
        return self.code


class Stmt(SimpleHash, Node, Typed):
    def get_c_decl(self, func):
        stmts = {
            # Uniqify on the name (to avoid generating duplicated prototypes).
            # This is safe since every statement has a unique name anyway
            stmt.name: stmt
            for stmt in self.used_stmts
        }
        return '\n'.join(
            f'MAKE_STMT_PROTOTYPE({stmt.name}, {func.ctx_typ.name}, {stmt.typ.ctrl_monad.name})'
            for stmt in sorted(
                stmts.values(),
                key=attrgetter('name')
            )
        )


class ExprStmt(Stmt):
    def __init__(self, expr, name=None, *, prog):
        self.expr = expr
        self.variables = set()
        self.name = name or f'expr_{prog.make_unique_id()}'

    @property
    def typ(self):
        return self.expr.typ

    def get_c(self, func):
        return f'MAKE_STMT({self.name}, {func.ctx_typ.name}, {self.expr.typ.ctrl_monad.name}) {{ {self.expr.get_c(stmt=self)}; }}'


class RawStmt(ExprStmt):
    def __init__(self, typ, code, variables, name=None, *, prog):
        super().__init__(
            expr=RawExpr(
                code=code,
                typ=typ,
                prog=prog,
            ),
            name=name,
            prog=prog,
        )
        self.variables = set(variables)


class IfStmt(Stmt):
    def __init__(self, cond, true, false, *, prog):
        self.prog = prog

        if true.typ != false.typ:
            raise ValueError(f'Branches of an "if" expression must have the same type: {true.typ.name} != {false.typ.name}')

        cond = prog.ExprStmt(cond)

        self.cond = cond
        self.true = true
        self.false = false
        self.tmp_cond = prog.TmpVar(typ=self.cond.typ)

        self.typ = true.typ
        self.variables = (
            self.cond.variables |
            self.true.variables |
            self.false.variables |
            {self.tmp_cond}
        )
        assign = AssignStmt(self.cond, var=self.tmp_cond, prog=prog)

        if_expr = prog.RawExpr(
            code=f'({{ if({self.tmp_cond.local_ref}) TAIL_CALL({self.true.name}(ctx)); else TAIL_CALL({self.false.name}(ctx)); }})',
            typ=self.typ,
            prog=prog,
        )
        if_stmt = ExprStmt(expr=if_expr, prog=prog)
        bound = BoundStmt(assign, if_stmt, prog=prog)

        self.bound = bound
        self.assign = assign
        self.if_stmt = if_stmt

    @property
    def name(self):
        return self.bound.name

    def get_c(self, func):
        stmts = [
            self.true,
            self.false,
            self.assign,
            self.if_stmt,
            self.bound,
        ]
        return '\n'.join(
            stmt.get_c(func=func)
            for stmt in stmts
        )


class AssignStmt(Stmt):
    def __init__(self, expr, var=None, *, prog):
        if isinstance(expr, Stmt):
            stmt = expr
        else:
            stmt = prog.ExprStmt(expr)

        if isinstance(var, Var):
            pass
        elif var is None:
            var = prog.TmpVar(typ=stmt.typ)
        else:
            var = prog.UserVar(name=var, typ=stmt.typ)

        if not ((var.typ == stmt.typ) or var.typ == prog.builtin_typs['unit']):
            raise ValueError(f'variable type {var.typ} mismatching expression type {stmt.typ}')

        self.var = var
        self.stmt = stmt

    @property
    def variables(self):
        return {self.var}

    @property
    def name(self):
        return self.stmt.name

    @property
    def typ(self):
        return self.stmt.typ

    def get_c(self, func):
        return self.stmt.get_c(func=func)


class BoundStmt(Stmt):
    def __init__(self, assign, stmt, *, prog):
        self.stmt = stmt
        self.assign = assign
        self.name = f'bound_stmt_{prog.make_unique_id()}'
        self.variables = assign.variables | stmt.variables
        self.prog = prog

    @property
    def var(self):
        assign = self.assign
        if isinstance(assign, AssignStmt):
            return assign.var
        else:
            return self.prog.BlackHoleVar()

    @property
    def typ(self):
        return self.stmt.typ

    def get_c(self, func):
        return f'BIND_STMT({self.name}, {func.ctx_typ.name}, {self.var.name}, {self.assign.name}, {self.stmt.name})'


class SequenceStmt(Stmt):
    def __init__(self, stmts, *, prog):
        stmts = list(stmts)
        _, *bound_stmts = accumulate_right(
            stmts,
            prog.BoundStmt,
            reverse=True,
        )

        self.variables = {
            var
            for stmt in stmts
            for var in stmt.variables
        }
        self.prog = prog
        self.stmts = stmts
        self._bound_stmts = bound_stmts

        if bound_stmts:
            self.stmt = bound_stmts[-1]
        else:
            self.stmt = stmts[0]

    @property
    def used_stmts(self):
        return self.stmt.used_stmts

    def get_c(self, func):
        dedup_stmts = deduplicate(self.stmts)
        return '\n'.join(
            stmt.get_c(func=func)
            for stmt in chain(
                dedup_stmts,
                self._bound_stmts,
            )
            if stmt is not self
        )

    @property
    def name(self):
        return self.stmt.name

    @property
    def typ(self):
        return self.stmt.typ


class _LoopBodyStmt(SequenceStmt):
    def __init__(self, body, *, prog):
        body = list(body)
        assert body

        self.prog = prog
        self._last_stmt = body[-1]
        # Needed so that self.variables does not fail in super().__init__
        self.variables = set()
        super().__init__(
            stmts=body + [self],
            prog=prog,
        )

    @property
    def typ(self):
        return self._last_stmt.typ


class LoopStmt(SequenceStmt):
    def __init__(self, body, init=None, *, prog):
        body = prog._LoopBodyStmt(body=body)
        super().__init__(
            stmts = ([] if init is None else [init]) + [body],
            prog=prog,
        )


class ConsumeGeneratorStmt(LoopStmt):
    def __init__(self, generator_stmt, make_consumer_stmt, *, prog):
        # This needs to be a fully private variable, since it will be
        # overwritten with junk on the last iteration when the generator breaks
        # without actually providing a value.
        assign_var = prog.TmpVar(typ=generator_stmt.typ)
        stmt = make_consumer_stmt(assign_var)

        self.stmt = stmt
        self.generator_stmt = generator_stmt
        self.prog = prog

        gen_state = prog.TmpVar(typ=generator_stmt.typ.ctrl_monad)
        stmt_ret = prog.TmpVar(typ=stmt.typ)
        resume = prog.TmpVar(typ=prog.builtin_typs['bool'])

        consume_name = f'__consume_{prog.make_unique_id()}'
        code = textwrap.dedent(f'''
            if ({resume.local_ref}) {{
                {gen_state.local_ref} = {gen_state.local_ref}.value.value.thunk.k(ctx);
            }} else {{
                {gen_state.local_ref} = {generator_stmt.name}(ctx);
                {resume.local_ref} = 1;
            }}
            if ({gen_state.local_ref}.tag != CTRL_GENERATOR_YIELD)
                {resume.local_ref} = 0;
            return {gen_state.local_ref}.value;
        ''')
        consume_stmt = prog.RawStmt(
            typ=generator_stmt.typ,
            code=code,
            variables={
                gen_state,
                stmt_ret,
                resume,
            },
            name=consume_name,
        )

        init_stmt = prog.ExprStmt(
            expr=prog.ControlExpr('Return', val=0, typ=resume.typ),
        )

        super().__init__(
            init=init_stmt,
            body=[
                prog.AssignStmt(
                    var=assign_var,
                    expr=consume_stmt,
                ),
                prog.AssignStmt(
                    var=stmt_ret,
                    expr=stmt,
                ),
            ],
            prog=prog,
        )

        self.variables.update((gen_state, resume))

    def get_c(self, func):
        return (
            self.generator_stmt.get_c(func=func) +
            '\n' +
            super().get_c(func=func)
        )


class Func(SimpleHash, Node):
    def __init__(self, name, stmts, prog, params=None, public=False):
        params = {
            name: prog.FuncUserParam(name, typ=typ)
            for name, typ in (params or {}).items()
        }

        stmts = list(stmts)
        assert stmts
        variables = {
            var
            for stmt in stmts
            for var in stmt.variables
        } | set(params.values())
        self.stmt = prog.SequenceStmt(stmts=stmts)

        self.name = name
        self.prog = prog
        self.public = public

        self.params = params
        self.variables = variables
        self.ctx_typ = CtxTyp(
            prog=prog,
            name=f'ctx_{self.name}',
            variables=variables
        )
        prog.funcs.append(self)

    @property
    def return_typ(self):
        return self.stmt.typ

    def get_c(self):
        macro = 'PUBLIC_FUNC' if self.public else 'FUNC'
        return (
            self.stmt.get_c_decl(func=self) +
            '\n\n' +
            self.stmt.get_c(func=self) +
            '\n\n' +
            f'{macro}({self.name}, {self.ctx_typ.name}, {self.stmt.name})'
        )


class UserFunc(Func):
    def __init__(self, name, **kwargs):
        super().__init__(
            name=f'__user_{name}',
            **kwargs,
        )


class Program:
    def __init__(self):
        self._unique_id_cnt = -1
        self.funcs = []

        builtin_typs = {
            self.UnitTyp(),
            self.BuiltinTyp('char'),
            self.BuiltinTyp('int'),
            self.BuiltinTyp('bool'),
        }
        self.builtin_typs = {
            typ.iso_name: typ
            for typ in builtin_typs
        }

    def __getattr__(self, attr):
        cls = self.__init__.__globals__[attr]
        if isinstance(cls, type):
            return partial(cls, prog=self)
        else:
            raise AttributeError(f'{attr} cannot be looked up on {self.__class__.__qualname__} as it is not a type')

    def make_unique_id(self):
        self._unique_id_cnt += 1
        return str(self._unique_id_cnt)

    @property
    def used_typs(self):
        return [
            typ
            for func in self.funcs
            for typ in func.used_typs
        ]


def main():

    prog = Program()
    var_x=prog.UserVar('x', typ=prog.builtin_typs['int'])

    main_f = prog.UserFunc(
        name='f',
        params={
            'param1': prog.builtin_typs['int'],
        },
        stmts=[
            (stmt_x:=prog.AssignStmt(
                var='x',
                expr=(
                    prog.RawExpr(
                        code=rf'printf("user param1=%d\n", ctx->__param_user_param1)',
                        typ='unit',
                    ) +
                    prog.SimpleExpr(val=3, typ=prog.builtin_typs['int'])
                ),
            )),
            prog.AssignStmt(
                # var=var_x,
                var=prog.BlackHoleVar(),
                expr=prog.SimpleExpr(val=42, typ=prog.builtin_typs['int']),
            ),
            # (stmt_y:=prog.AssignStmt(
            #     var='y',
            #     expr=prog.Expr(
            #         code=prog.ControlExpr('Return', stmt_x.var.local_ref),
            #         typ=prog.builtin_typs['int'],
            #     ),
            # )),
            (stmt_y:=prog.AssignStmt(
                var='y',
                expr=prog.LoopStmt(
                    body=[
                        prog.ExprStmt(
                            expr=(
                                prog.RawExpr(r'printf("useless loop\n")', typ=prog.builtin_typs['unit']) +
                                prog.SimpleExpr(val=None)
                            ),
                        ),
                        prog.ExprStmt(
                            expr=(
                                prog.RawExpr(r'printf("useless loop\n")', typ=prog.builtin_typs['unit']) +
                                prog.ControlExpr('Break', val=stmt_x.var.local_ref, typ=prog.builtin_typs['int'])
                            ),
                        )
                    ],
                ),
            )),
            (stmt_z:=prog.AssignStmt(
                # var='z',
                var=None,
                expr=(
                    prog.RawExpr(
                        rf'printf("y=%d\n", {stmt_y.var.local_ref})',
                        typ=prog.builtin_typs['unit'],
                    ) +
                    prog.SimpleExpr(val=rf'{stmt_y.var.local_ref} + 1', typ=stmt_y.var.typ)
                ),
            )),
            prog.ExprStmt(
                expr=(
                    prog.RawExpr(
                        rf'printf("snd z=%d\n", {stmt_z.var.local_ref})',
                        typ=prog.builtin_typs['unit'],
                    ) +
                    prog.SimpleExpr(val=None)
                ),
            ),
            prog.ConsumeGeneratorStmt(
                generator_stmt=prog.SequenceStmt(
                    # TODO: GeneratorYield currently depends on being bound to
                    # something else, otherwise it will not go through ___BIND
                    # and will not set the thunk. The interaction of ___BIND
                    # and the consumer code in ConsumeGeneratorStmt needs to be
                    # rethaught to avoid the duplication
                    #
                    # TODO: GeneratorSuspend is broken, as it is just a "YieldNone".
                    stmts=[
                        prog.ExprStmt(
                            prog.RawExpr(rf'printf("hello1\n");', typ=prog.builtin_typs['unit']) +
                            # prog.ControlExpr('GeneratorYield', val=3, typ=prog.builtin_typs['int'])
                            prog.YieldExpr(val=3, typ=prog.builtin_typs['int']),
                        ),
                        prog.ExprStmt(
                            prog.RawExpr(rf'printf("hello2\n");', typ=prog.builtin_typs['unit']) +
                            #prog.ControlExpr('GeneratorYield', val=4, typ=prog.builtin_typs['int'])
                            # prog.ControlExpr('GeneratorYield', val=4, typ=prog.builtin_typs['int'])
                            prog.YieldExpr(val=4, typ=prog.builtin_typs['int']),
                        ),
                        prog.ExprStmt(
                            prog.RawExpr(rf'printf("hello3\n");', typ=prog.builtin_typs['unit']) +
                            # prog.ControlExpr('GeneratorFinish', typ=prog.builtin_typs['int'])
                            prog.YieldExpr(val=None, typ=prog.builtin_typs['int']),
                        ),
                    ]
                ),
                make_consumer_stmt=lambda var: prog.ExprStmt(
                    expr=(
                        prog.RawExpr(
                            rf'printf("yielded value=%d\n", {var.local_ref})',
                            typ=prog.builtin_typs['unit'],
                        ) +
                        prog.SimpleExpr(val=None)
                    ),
                ),
            ),
            prog.LoopStmt(
                init=prog.AssignStmt(
                    var=stmt_z.var,
                    # var=prog.BlackHoleVar(),
                    expr=prog.SimpleExpr(val=55, typ=stmt_z.var.typ),
                ),
                body=[
                    prog.AssignStmt(
                        var=stmt_z.var,
                        expr=(
                            prog.RawExpr(
                                rf'printf("in loop   z=%d\n", {stmt_z.var.local_ref})',
                                typ=prog.builtin_typs['unit'],
                            ) +
                            prog.SimpleExpr(val=rf'{stmt_z.var.local_ref} + 1', typ=stmt_z.var.typ)
                        ),
                    ),
                    prog.AssignStmt(
                        var=stmt_z.var,
                        expr=(
                            prog.RawExpr(
                                rf'printf("in loop 2 z=%d\n", {stmt_z.var.local_ref})',
                                typ=prog.builtin_typs['unit'],
                            ) +
                            prog.SimpleExpr(val=rf'{stmt_z.var.local_ref}', typ=stmt_z.var.typ)
                        ),
                    ),
                    prog.IfStmt(
                        cond=prog.SimpleExpr(val=f'{stmt_z.var.local_ref} % 2 == 0', typ=prog.builtin_typs['int']),
                        true=prog.ExprStmt(
                            expr=(
                                prog.RawExpr(
                                    rf'printf("in loop 3 is odd\n")',
                                    typ=prog.builtin_typs['unit'],
                                ) +
                                prog.SimpleExpr(val=None)
                            )
                        ),
                        false=prog.ExprStmt(
                            expr=(
                                prog.RawExpr(
                                    rf'printf("in loop 3 is even\n")',
                                    typ=prog.builtin_typs['unit'],
                                ) +
                                prog.SimpleExpr(val=None)
                            )
                        ),
                    ),
                    prog.LoopStmt(
                        body=[
                            prog.ExprStmt(
                                expr=(
                                    prog.RawExpr(
                                        r'printf("inner loop\n")',
                                        typ=prog.builtin_typs['unit'],
                                    ) +
                                    prog.SimpleExpr(val=None)
                                ),
                            ),
                            prog.LoopStmt(
                                body=[
                                    prog.ExprStmt(
                                        expr=(
                                            prog.RawExpr(
                                                r'printf("inner loop 2\n")',
                                                typ=prog.builtin_typs['unit'],
                                            ) +
                                            prog.SimpleExpr(val=None)
                                        ),
                                    ),
                                    prog.ExprStmt(
                                        expr=prog.ControlExpr('Break', val=stmt_x.var.local_ref, typ=prog.builtin_typs['int'])
                                    )
                                ],
                            ),
                            prog.ExprStmt(
                                expr=prog.ControlExpr('Break', val=stmt_x.var.local_ref, typ=prog.builtin_typs['int']),
                            ),
                        ],
                    ),
                ],
            ),
        ]
    )

    typs = '\n'.join(
        code
        for code in (
            typ.get_c()
            for typ in prog.used_typs
        )
        if code
    )

    import tempfile
    import subprocess

    assets = Path(ASSETS_PATH)
    monad_h = assets / 'ccodegen' / 'monad.h'

    with open(monad_h, 'r') as f:
        test_monad_h = f.read()

    src = '\n'.join((
        typs,
        '\n',
        main_f.get_c(),
        r"""
    int main() {
        USER_FUNC_PARAM_TYPE(f, param1) param1 = 22;
        USER_FUNC_RET_TYPE(f) x = USER_FUNC_CALL(f, USER_FUNC_PARAM(param1, param1));
        printf("tag=%d\n", x.tag);
        return 0;
    }
        """,
    ))

    import sys
    print(src, file=sys.stderr)
    # exit(0)

    exe_name = './testgen.exe'
    with tempfile.NamedTemporaryFile(suffix='.c') as f:
        f.write(test_monad_h.encode('utf-8'))
        f.write(src.encode('utf-8'))
        f.write(b'\n')
        f.flush()
        import os
        # os.system(f'cpp -P {f.name} | less')
        # os.system(f'cat {f.name} | less')
        # breakpoint()
        cpped_name = 'x.c'
        cpped = subprocess.check_output(['cpp', '-P', f.name])
        with open(cpped_name, 'wb') as f_cpp:
            f_cpp.write(cpped)
        subprocess.check_call(['clang-format', '-i', '--style={IndentWidth: 4}', cpped_name])

        cmd = ['gcc', '-Wall', '-Wextra', '-Wno-unneeded-internal-declaration', '-Wno-unused-parameter', '-Wno-unused-function', '-Wno-unused-variable', '-Wno-unused-label', '-std=gnu11', '-O3', f.name, '-o', exe_name]
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as e:
            exit(1)


    print('='*40)
    cmd = ['valgrind', '--leak-check=full', exe_name]
    subprocess.check_call(cmd)

main()
