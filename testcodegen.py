#! /usr/bin/env python3

import abc
import uuid
import functools
import textwrap
from functools import partial
from operator import attrgetter, itemgetter
from itertools import chain, accumulate


from lisa.utils import SimpleHash, deduplicate

def accumulate_right(xs, f):
    return reversed(list(
        accumulate(
            reversed(xs),
            lambda x, y: f(y, x),
        )
    ))


class Var(SimpleHash):
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
            typ=prog.BuiltinTyp('void_val')
        )


class Typ(SimpleHash, abc.ABC):
    def __init__(self, name, *, prog):
        self.name = name
        self.prog = prog

    def get_c(self):
        decl = self._get_c()
        return (
            rf'MAKE_CTRL_MONAD({self.name}, {self.name});' +
            (('\n' + decl) if decl else '')
        )

    @property
    def ctrl_monad(self):
        return self.prog.ControlMonadOf(self)


class BuiltinTyp(Typ):
    def _get_c(self):
        return ''


class ControlMonadOf(BuiltinTyp):
    def __init__(self, typ, *, prog):
        super().__init__(
            name=f'CTRL_MONAD({typ.name})',
            prog=prog,
        )


class UserTyp(Typ):
    def __init__(self, name, decl, *, prog):
        self.decl = decl
        super().__init__(
            name=f'__user_{name}',
            prog=prog,
        )

    def _get_c(self):
        return self.decl


class CtxTyp(Typ):
    def __init__(self, name, variables, *, prog):
        self.variables = variables
        super().__init__(
            name=f'struct {name}',
            prog=prog,
        )

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


class Expr(SimpleHash):
    def __init__(self, typ, *, prog):
        self.typ = typ
        self.prog = prog

    @abc.abstractmethod
    def get_c(self, stmt):
        pass

    def __add__(self, expr):
        return self.prog.CompoundExpr(self, expr)


class Control(Expr):
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


class Stmt(SimpleHash):
    def get_c_decl(self, func):
        return '\n'.join(
            f'MAKE_STMT_PROTOTYPE({stmt.name}, {func.ctx_typ.name}, {stmt.typ.ctrl_monad.name})'
            for stmt in sorted(
                self.used_stmts,
                key=attrgetter('name')
            )
        )

    @property
    def used_typs(self):
        return {self.typ}

    @property
    def used_stmts(self):
        return {self}


class ExprStmt(Stmt):
    def __init__(self, expr, *, prog):
        self.expr = expr
        self.variables = set()
        self.name = f'expr_{prog.make_unique_id()}'

    @property
    def typ(self):
        return self.expr.typ

    def get_c(self, func):
        return f'MAKE_STMT({self.name}, {func.ctx_typ.name}, {self.expr.typ.ctrl_monad.name}) {{ {self.expr.get_c(stmt=self)}; }}'


class RawStmt(ExprStmt):
    def __init__(self, code, typ, *, prog):
        super().__init__(
            expr=RawExpr(
                code=code,
                typ=typ,
                prog=prog,
            ),
            prog=prog,
        )


class IfStmt(Stmt):
    def __init__(self, cond, true, false, *, prog):
        self.prog = prog

        if true.typ != false.typ:
            raise ValueError(f'Branches of an "if" expression must have the same type: {true.typ.name} != {false.typ.name}')

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

    @property
    def used_typs(self):
        return (
            self.cond.used_typs |
            self.true.used_typs |
            self.false.used_typs
        )

    @property
    def used_stmts(self):
        return (
            self.cond.used_stmts |
            self.true.used_stmts |
            self.false.used_stmts |
            self.assign.used_stmts |
            self.if_stmt.used_stmts |
            self.bound.used_stmts
        )

    def get_c(self, func):
        return '\n'.join((
            self.true.get_c(func=func),
            self.false.get_c(func=func),
            self.assign.get_c(func=func),
            self.if_stmt.get_c(func=func),
            self.bound.get_c(func=func),
        ))


class AssignStmt(Stmt):
    def __init__(self, expr, var, *, prog):
        if isinstance(expr, Stmt):
            stmt = expr
        else:
            stmt = prog.ExprStmt(expr)

        if isinstance(var, Var):
            if not ((var.typ == stmt.typ) or var == prog.BlackHoleVar()):
                raise ValueError('variable type {var.typ} mismatching expression type {stmt.typ}')
        elif var is None:
            var = prog.TmpVar(typ=stmt.typ)
        else:
            var = prog.UserVar(name=var, typ=stmt.typ)

        self.var = var
        self.variables = {var}
        self.stmt = stmt

    @property
    def name(self):
        return self.stmt.name

    @property
    def used_typs(self):
        return self.stmt.used_typs

    @property
    def used_stmts(self):
        return self.stmt.used_stmts

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
    def used_typs(self):
        return self.stmt.used_typs | self.assign.used_typs

    @property
    def used_stmts(self):
        return (
            self.stmt.used_stmts |
            self.assign.used_stmts |
            {self}
        )

    @property
    def typ(self):
        return self.stmt.typ

    def get_c(self, func):
        return f'BIND_STMT({self.name}, {func.ctx_typ.name}, {self.var.name}, {self.assign.name}, {self.stmt.name})'


class SequenceStmt(Stmt):
    def __init__(self, stmts, *, prog):
        stmts = list(stmts)
        bound_stmts = list(
            reversed(
                list(
                    accumulate_right(
                        stmts,
                        prog.BoundStmt,
                    )
                )
            )
        )[1:]

        self.variables = {
            var
            for stmt in stmts
            for var in stmt.variables
        }
        self.prog = prog
        self.stmts = stmts
        self._bound_stmts = bound_stmts

        try:
            self.stmt = bound_stmts[-1]
        except IndexError:
            self.stmt = stmts[0]

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

    @property
    def used_typs(self):
        stmts = set(self.stmts) - {self}
        return {
            typ
            for stmt in stmts
            for typ in stmt.used_typs
        } | {self.typ}

    @property
    def used_stmts(self):
        stmts = set(self.stmts) - {self}
        return {
            _stmt
            for stmt in stmts
            for _stmt in stmt.used_stmts
        } | {self}


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


class ConsumeGeneratorStmt(Stmt):
    def __init__(self, *, prog):
        self.prog = prog


class Func(SimpleHash):
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

        self.name = name
        self.stmts = stmts
        self.stmt = prog.SequenceStmt(stmts=stmts)
        self.prog = prog
        self.public = public
        prog.funcs.append(self)

        self.params = params
        self.variables = variables
        self.ctx_typ = CtxTyp(
            prog=prog,
            name=f'ctx_{self.name}',
            variables=variables
        )

    @property
    def used_typs(self):
        return self.stmt.used_typs

    @property
    def typ(self):
        return self.stmt.typ

    def get_c(self):
        macro = 'PUBLIC_FUNC' if self.public else 'FUNC'
        return (
            self.ctx_typ.get_c() +
            '\n\n' +
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
        return {
            typ
            for func in self.funcs
            for typ in func.used_typs
        }


prog = Program()
typs = {
    'int': prog.BuiltinTyp('int'),
    'char': prog.BuiltinTyp('char'),
    'void_val': prog.BuiltinTyp('void_val'),
}
var_x=prog.UserVar('x', typ=typs['int'])

f = prog.UserFunc(
    name='f',
    params={
        'param1': typs['int'],
    },
    stmts=[
        (stmt_x:=prog.AssignStmt(
            var='x',
            expr=(
                prog.RawExpr(
                    code=rf'printf("user param1=%d\n", ctx->__param_user_param1)',
                    typ=typs['void_val'],
                ) +
                prog.Control('Return', val=3, typ=typs['int'])
            ),
        )),
        prog.AssignStmt(
            # var=var_x,
            var=prog.BlackHoleVar(),
            expr=prog.Control('Return', val=42, typ=typs['int']),
        ),
        # (stmt_y:=prog.AssignStmt(
        #     var='y',
        #     expr=prog.Expr(
        #         code=prog.Control('Return', stmt_x.var.local_ref),
        #         typ=typs['int'],
        #     ),
        # )),
        (stmt_y:=prog.AssignStmt(
            var='y',
            expr=prog.LoopStmt(
                body=[
                    prog.ExprStmt(
                        expr=(
                            prog.RawExpr(r'printf("useless loop\n")', typ=typs['void_val']) +
                            prog.Control('ReturnNone', typ=typs['int'])
                        ),
                    ),
                    prog.ExprStmt(
                        expr=(
                            prog.RawExpr(r'printf("useless loop\n")', typ=typs['void_val']) +
                            prog.Control('Break', val=stmt_x.var.local_ref, typ=typs['int'])
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
                    typ=typs['void_val'],
                ) +
                prog.Control('Return', val=rf'{stmt_y.var.local_ref} + 1', typ=typs['char'])
            ),
        )),
        prog.ExprStmt(
            expr=(
                prog.RawExpr(
                    rf'printf("snd z=%d\n", {stmt_z.var.local_ref})',
                    typ=typs['void_val'],
                ) +
                prog.Control('ReturnNone', typ=typs['int'])
            ),
        ),
        prog.LoopStmt(
            init=prog.AssignStmt(
                var=stmt_z.var,
                # var=prog.BlackHoleVar(),
                expr=prog.Control('Return', val=55, typ=typs['char']),
            ),
            body=[
                prog.AssignStmt(
                    var=stmt_z.var,
                    expr=(
                        prog.RawExpr(
                            rf'printf("in loop   z=%d\n", {stmt_z.var.local_ref})',
                            typ=typs['void_val'],
                        ) +
                        prog.Control('Return', val=rf'{stmt_z.var.local_ref} + 1', typ=typs['char'])
                    ),
                ),
                prog.AssignStmt(
                    var=stmt_z.var,
                    expr=(
                        prog.RawExpr(
                            rf'printf("in loop 2 z=%d\n", {stmt_z.var.local_ref})',
                            typ=typs['void_val'],
                        ) +
                        prog.Control('Return', val=rf'{stmt_z.var.local_ref}', typ=typs['char'])
                    ),
                ),
                prog.IfStmt(
                    cond=prog.ExprStmt(
                        expr=prog.Control('Return', val=f'{stmt_z.var.local_ref} % 2 == 0', typ=typs['int'])
                    ),
                    true=prog.ExprStmt(
                        expr=(
                            prog.RawExpr(
                                rf'printf("in loop 3 is odd\n")',
                                typ=typs['void_val'],
                            ) +
                            prog.Control('ReturnNone', typ=typs['void_val'])
                        )
                    ),
                    false=prog.ExprStmt(
                        expr=(
                            prog.RawExpr(
                                rf'printf("in loop 3 is even\n")',
                                typ=typs['void_val'],
                            ) +
                            prog.Control('ReturnNone', typ=typs['void_val'])
                        )
                    ),
                ),
                prog.LoopStmt(
                    body=[
                        prog.ExprStmt(
                            expr=(
                                prog.RawExpr(
                                    r'printf("inner loop\n")',
                                    typ=typs['void_val'],
                                ) +
                                prog.Control('ReturnNone', typ=typs['int'])
                            ),
                        ),
                        prog.LoopStmt(
                            body=[
                                prog.ExprStmt(
                                    expr=(
                                        prog.RawExpr(
                                            r'printf("inner loop 2\n")',
                                            typ=typs['void_val'],
                                        ) +
                                        prog.Control('ReturnNone', typ=typs['int'])
                                    ),
                                ),
                                prog.ExprStmt(
                                    expr=prog.Control('Break', val=stmt_x.var.local_ref, typ=typs['int'])
                                )
                            ],
                        ),
                        prog.ExprStmt(
                            expr=prog.Control('Break', val=stmt_x.var.local_ref, typ=typs['int']),
                        ),
                    ],
                ),
                # prog.ExprStmt(
                #     expr=prog.Expr(
                #         code=prog.RawCode(rf'if({stmt_z.var.local_ref} > 100) {{ BreakNone(expr_9); }} else {{ ReturnNone(expr_9); }}'),
                #         typ=typs['void_val'],
                #     ),
                # ),
            ],
        ),
    ]
)

typs = '\n'.join(
    typ.get_c()
    for typ in sorted(prog.used_typs, key=attrgetter('name'))
)

import tempfile
import subprocess

with open('testmonad.h') as fh:
    test_monad_h = fh.read()

src = '\n'.join((
    '#include <stdio.h>',
    typs,
    f.get_c(),
    r"""
int main() {
    // typeof(__user_f()) x = __user_f();
    USER_FUNC_PARAM_TYPE(f, param1) param1 = 22;
    USER_FUNC_RET_TYPE(f) x = USER_FUNC_CALL(f, USER_FUNC_PARAM(param1, param1));
    //FUNC_RET_TYPE(__user_f) x = CALL_FUNC(__user_f);
    printf("tag=%d\n", x.tag);
    return 0;
}
    """,
))

import sys
print(src, file=sys.stderr)

exe_name = './testgen.exe'
with tempfile.NamedTemporaryFile(suffix='.c') as f:
    f.write(test_monad_h.encode('utf-8'))
    f.write(src.encode('utf-8'))
    f.write(b'\n')
    f.flush()
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

    # breakpoint()

print('='*40)
cmd = ['valgrind', '--leak-check=full', exe_name]
subprocess.check_call(cmd)
