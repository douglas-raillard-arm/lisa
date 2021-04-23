// usr/bin/clang -Wall -Wextra -Wno-unused-parameter -O3 -std=gnu11 "$0" &&

// ./a.out; exit

#define INLINE __attribute__((always_inline))
/* #define INLINE inline */

#include <stdbool.h>

#ifdef BPF
#include <linux/slab.h>
#include <linux/stddef.h>
#include <linux/string.h>
#define myprint(s, ...)                                                        \
    ({                                                                         \
        char _fmt[] = s;                                                       \
        bpf_trace_printk_(_fmt, sizeof(_fmt), __VA_ARGS__);                    \
    })
#define NULL ((void *)0)
#define size_t unsigned long
#define malloc(x) kmalloc(x, GFP_USER)
#define free(x) kfree(x)
#else
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define myprint printf
#endif

enum ctrl_tag {
    CTRL_YIELD,
    CTRL_RETURN,
    CTRL_SKIP,
    CTRL_BREAK,
    CTRL_RETURN_NONE,
    CTRL_BREAK_NONE,
};

#define THUNK(name) struct thunk_##name
#define MAKE_THUNK(name)                                                       \
    THUNK(name) {                                                              \
        CTRL_MONAD(name) (*k)(void *);                                         \
        void *ctx;                                                             \
    }

#define CTRL_MONAD(name) struct ctrl_monad_##name
#define MAKE_CTRL_MONAD(name, type)                                            \
    CTRL_MONAD(name);                                                          \
    MAKE_THUNK(name);                                                          \
    CTRL_MONAD(name) {                                                         \
        enum ctrl_tag tag;                                                     \
        THUNK(name) thunk;                                                     \
        type value;                                                            \
    }

MAKE_THUNK(None);
CTRL_MONAD(None) {
    enum ctrl_tag tag;
    THUNK(None) thunk;
    char value[0];
};

MAKE_CTRL_MONAD(int, int);
MAKE_CTRL_MONAD(ull, unsigned long long);

#define RETURN_X(stmt, _tag, val)                                              \
    return (typeof(stmt(NULL))) { .tag = _tag, .value = (val), }

#define RETURN_EMPTY(stmt, _tag)                                               \
    return (typeof(stmt(NULL))) { .tag = _tag }

#define Yield(stmt, x) RETURN_X(stmt, CTRL_YIELD, x)
#define Return(stmt, x) RETURN_X(stmt, CTRL_RETURN, x)
#define Break(stmt, x) RETURN_X(stmt, CTRL_BREAK, x)
#define BreakNone(stmt) RETURN_EMPTY(stmt, CTRL_BREAK_NONE)
#define ReturnNone(stmt) RETURN_EMPTY(stmt, CTRL_RETURN_NONE)
#define Skip(stmt) RETURN_EMPTY(stmt, CTRL_SKIP)

#define SET_MONAD_VALUE(m, x)                                                  \
    memcpy((char *)&(m) + offsetof(typeof(m), value), &(x), sizeof(m.value))

#define GET_MONAD_VALUE(addr, m)                                               \
    memcpy((char *)(addr), &(m).value, sizeof((m).value))

#define ___BIND(ctx, name, ma, a_mb, a_mb_addr)                                \
    do {                                                                       \
        typeof(ma) __bind_ma;                                                  \
        typeof((a_mb_addr(NULL))) __bind_mb;                                   \
    __bind_recurse:                                                            \
        __bind_ma = (ma);                                                      \
        if (__bind_ma.tag == CTRL_RETURN) {                                    \
            GET_MONAD_VALUE(&ctx->name, __bind_ma);                            \
            return (a_mb);                           \
        } else if (__bind_ma.tag == CTRL_BREAK) {                              \
            GET_MONAD_VALUE(&ctx->name, __bind_ma);                            \
            SET_MONAD_VALUE(__bind_mb, __bind_ma.value);                       \
            __bind_mb.tag = CTRL_RETURN;                                       \
        } else if (__bind_ma.tag == CTRL_RETURN_NONE) {                        \
            return (a_mb);                           \
        } else if (__bind_ma.tag == CTRL_BREAK_NONE) {                         \
            __bind_mb.tag = CTRL_RETURN_NONE;                                  \
        } else if (__bind_ma.tag == CTRL_YIELD) {                              \
            __bind_mb.tag = CTRL_YIELD;                                        \
            __bind_mb.thunk.ctx = malloc(sizeof(*ctx));                        \
            __bind_mb.thunk.k = (typeof(__bind_mb.thunk.k))(a_mb_addr);        \
            SET_MONAD_VALUE(__bind_mb, __bind_ma.value);                       \
            GET_MONAD_VALUE(&ctx->name, __bind_ma);                            \
            memcpy(__bind_mb.thunk.ctx, ctx, sizeof(*ctx));                    \
        } else {                                                               \
            __bind_mb.tag = __bind_ma.tag;                                     \
        }                                                                      \
        return __bind_mb;                                                      \
    } while (0)

#define __BIND(ctx, name, ma, a_mb) ___BIND(ctx, name, ma, a_mb(ctx), a_mb)

#define BIND_STMT(bound_name, ctx_type, name, ma, a_mb)                        \
    static inline typeof((a_mb)(NULL)) bound_name(ctx_type *ctx) {             \
        __BIND(ctx, name, (ma)(ctx), (*(typeof(bound_name)*)(&a_mb)));                                    \
    }

/* Specialized implementation for tight loops. We could use tail recursion, but
 * since clang is not always able to recognize it we just use a goto with an
 * associated lavel in __BIND
 */
#define BIND_REC_STMT(bound_name, ctx_type, name, a_mb)                        \
    static typeof(a_mb(NULL)) bound_name(ctx_type *ctx) {                      \
        ___BIND(ctx, name, (a_mb)(ctx), bound_name(ctx), a_mb); \
    }

#define BIND_EXPR(bound_name, ctx_type, name, expr, a_mb)                      \
    static INLINE typeof((a_mb)(NULL)) bound_name(ctx_type *ctx) {             \
        __BIND(ctx, name, expr, a_mb);                                         \
    }

#define __FUNC(name, ctx_type, expr)                                           \
    typeof(expr(NULL)) __attribute__((flatten)) name(void) {                   \
        ctx_type ctx;                                                          \
        return expr(&ctx);                                                     \
    }
#define FUNC(name, ctx_type, expr) static inline __FUNC(name, ctx_type, expr)
#define PUBLIC_FUNC(name, ctx_type, expr) static __FUNC(name, ctx_type, expr)
#define EVAL(ctx_type, m)                                                      \
    ({                                                                         \
        ctx_type ctx;                                                          \
        m(&ctx).value;                                                         \
    })

#define CONSUME_GENERATOR(bound_name, ctx_type, name, gen_expr, a_mb)          \
    static INLINE typeof(((ctx_type *)NULL)->__scratch_gen_##bound_name)       \
        __consume_gen_##bound_name(ctx_type *ctx) {                            \
        if (ctx->__resume_gen_##bound_name) {                                  \
            void *__thunk_ctx;                                                 \
            __thunk_ctx = ctx->__scratch_gen_##bound_name.thunk.ctx;           \
            ctx->__scratch_gen_##bound_name =                                  \
                ctx->__scratch_gen_##bound_name.thunk.k(__thunk_ctx);          \
            free(__thunk_ctx);                                                 \
        } else {                                                               \
            ctx->__scratch_gen_##bound_name = (gen_expr);                      \
            ctx->__resume_gen_##bound_name = 1;                                \
        }                                                                      \
        if (ctx->__scratch_gen_##bound_name.tag == CTRL_YIELD)                 \
            ctx->__scratch_gen_##bound_name.tag = CTRL_RETURN;                 \
        return ctx->__scratch_gen_##bound_name;                                \
    }                                                                          \
    static typeof((a_mb)(NULL))                                         \
        __consume_gen_loop2_##bound_name(ctx_type *ctx);                       \
    BIND_STMT(__consume_gen_loop1_##bound_name, ctx_type, __sink, a_mb,        \
              __consume_gen_loop2_##bound_name);                               \
    BIND_STMT(__consume_gen_loop2_##bound_name, ctx_type, name,                \
              __consume_gen_##bound_name, __consume_gen_loop1_##bound_name);   \
    static INLINE typeof(__consume_gen_loop2_##bound_name(NULL))               \
        bound_name(ctx_type *ctx) {                                            \
        ctx->__resume_gen_##bound_name = 0;                                    \
        return __consume_gen_loop2_##bound_name(     \
            ctx);                                                              \
    }


struct ctx {
    int y;
    int x;
};

static INLINE CTRL_MONAD(int) stmt1(struct ctx *ctx) {
    /* Skip(); */
    Return(stmt1, 33);
}

static INLINE CTRL_MONAD(int) stmt2(struct ctx *ctx) {
    myprint("got val=%d\n", ctx->y);
    /* ReturnNone(); */
    Return(stmt2, ctx->y * 3);
}

static INLINE CTRL_MONAD(int) stmt3(struct ctx *ctx) { Return(stmt3, ctx->x); }
static INLINE CTRL_MONAD(int) stmt4(struct ctx *ctx) {
    /* /\* printf("loop=%d\n", ctx->x); *\/ */
    if (ctx->x < 104) {
        Return(stmt4, (ctx->x + 1));
        /* ReturnNone(stmt4); */
    } else {
        Break(stmt4, (ctx->x));
        /* BreakNone(stmt4); */
    }
}

static INLINE CTRL_MONAD(int) stmt5(struct ctx *ctx) {
    printf("loop=%d\n", ctx->x);
    Return(stmt5, 1);
}

BIND_REC_STMT(bind2, struct ctx, x, stmt4);
BIND_STMT(bind3, struct ctx, x, stmt3, bind2);
BIND_STMT(bind4, struct ctx, x, stmt2, bind3);
BIND_STMT(bind5, struct ctx, y, stmt1, bind4);
FUNC(loop_f, struct ctx, bind5);

/* CONSUME_GENERATOR(bind1, struct ctx, x, bind2(ctx), stmt5); */

unsigned long strlen(const char *s) {
    unsigned long i = 0;
    while (*s != 0) {
        s++;
        i++;
    }
    return i + 1;
}

/* static INLINE void print(const char *s) { myprint(s, strlen(s)); } */
/* static INLINE void custom_print(const char *s, size_t len) { myprint(s, len);
 * } */

// Trick to be able to act on arbitrary string
static INLINE void __str(int i, void (*f)(const char *, size_t)) {
    switch (i) {
    case 0: {
        const char s[] = "ebpf\n";
        f(s, sizeof(s));
    } break;
    case 1: {
        const char s[] = "python\n";
        f(s, sizeof(s));
    } break;
    }
}

static INLINE CTRL_MONAD(int) __use_loop_stmt1(struct ctx *ctx) {
    /* __str(0, custom_print); */
    /* __str(1, custom_print); */
    myprint("finished loop=%d\n", ctx->x);
    /* myprint(global_var); */
    Return(__use_loop_stmt1, ctx->x);
}

BIND_EXPR(use_loop_stmt1, struct ctx, x, loop_f(), __use_loop_stmt1);
/* BIND_STMT(use_loop_stmt1, struct ctx, x, stmt_loop, __use_loop_stmt1); */

static INLINE CTRL_MONAD(int) __use_loop_body(struct ctx *ctx) {
    myprint("loop last value=%d\n", ctx->y);
    ReturnNone(__use_loop_body);
}
BIND_STMT(use_loop_body, struct ctx, y, use_loop_stmt1, __use_loop_body)
/* BIND_STMT(use_loop_body, struct ctx, y, loop1, __use_loop_body) */
PUBLIC_FUNC(use_loop, struct ctx, use_loop_body)

void use_loop1(void) {
    int x = 0;
    int y;
    /* ReturnNone(); */
    while (x < 100004) {
        /* myprint("loop=%d\n", x); */
        x = (x + 1);
    }

    myprint("finished loop=%d\n", x);
    y = x;
    myprint("loop last value=%d\n", y);
}

static INLINE void _use_loop2(struct ctx *ctx) {
    int ret;

    /* ReturnNone(); */
    if (ctx->x < 100004)
        ret = 1;
    else
        ret = 0;

    if (ret) {
        ctx->x = ctx->x + 1;
        return _use_loop2(ctx);
    }
}

void use_loop2(void) {
    struct ctx ctx;
    struct ctx *ctxp = &ctx;
    ctxp->x = 0;
    _use_loop2(&ctx);
    myprint("finished loop=%d\n", ctxp->x);
    ctxp->y = ctxp->x;
    myprint("loop last value=%d\n", ctxp->y);
}

#define COR_YIELD(ctx, x, next)                                                \
    do {                                                                       \
        ctx->yield = x;                                                        \
        ctx->next_state = next;                                                \
        return;                                                                \
    } while (0)

#define COR_JMP(name, ctx, state)                                              \
    do {                                                                       \
        ctx->next_state = state;                                               \
        name(ctx);                                                             \
    } while (0)

enum cor_state {
    COR_DONE,
    COR_STATE1,
    COR_STATE2,
    COR_STATE3,
};

struct cor_ctx {
    enum cor_state next_state;
    int yield;
    int x;
    unsigned long long y;
};

void cor_body(struct cor_ctx *ctx) {
    switch (ctx->next_state) {
    case COR_STATE1:
        printf("stage1\n");
        ctx->x = 1;
        COR_YIELD(ctx, 3, COR_STATE2);
    case COR_STATE2:
        printf("stage2=%d\n", ctx->x);
        COR_YIELD(ctx, 1, COR_DONE);
    case COR_STATE3:
        printf("stage2=%d\n", ctx->x);
        COR_JMP(cor_body, ctx, COR_STATE2);
    case COR_DONE:
        return;
    }
}

void cor_init(struct cor_ctx *ctx) { ctx->next_state = COR_STATE1; }

struct ctx2 {
    CTRL_MONAD(int) __scratch_gen_gen_bind2;
    bool __resume_gen_gen_bind2;

    CTRL_MONAD(int) __scratch_gen_gen_bind3;
    bool __resume_gen_gen_bind3;

    CTRL_MONAD(int) __scratch_gen_loop1;
    bool __resume_gen_loop1;

    char __sink[0];

    int y;
    int x;
    int z;
};

static INLINE CTRL_MONAD(int) init1(struct ctx2 *ctx) { Return(init1, 20); }
static INLINE CTRL_MONAD(int) gen1(struct ctx2 *ctx) {
    int _x = ctx->x + 1;
    /* if (_x % 2) */
    /*     Yield(gen1, _x); */
    /* else */
    /*     Return(gen1, _x); */
    if (ctx->x < 100)
        Yield(gen1, ctx->x + 1);
    else
        BreakNone(gen1);
}

static INLINE CTRL_MONAD(int) __1gen1(struct ctx2 *ctx) {
    printf("from 1=%d\n", ctx->x);
    Yield(__1gen1, ctx->x);
}
static INLINE CTRL_MONAD(int) __2gen1(struct ctx2 *ctx) {
    printf("from 2=%d\n", ctx->x);
    Yield(__2gen1, ctx->x + 1);
}
static INLINE CTRL_MONAD(int) loop1(struct ctx2 *ctx);

BIND_STMT(__2_loopgen1, struct ctx2, x, __2gen1, loop1)
BIND_STMT(loop1, struct ctx2, x, __1gen1, __2_loopgen1)

static INLINE CTRL_MONAD(int) consume1(struct ctx2 *ctx) {
    printf("y=%d\n", ctx->y);
    Yield(consume1, ctx->y);
}

static INLINE CTRL_MONAD(None) consume2(struct ctx2 *ctx) {
    printf("z=%d\n", ctx->z);
    /* if (ctx->z > 100) */
    /*     BreakNone(consume2); */
    ReturnNone(consume2);
    /* Yield(consume2, ctx->z); */
}

/* BIND_REC_STMT(loop1, struct ctx2, x, gen1) */

BIND_STMT(gen_bind1, struct ctx2, x, init1, loop1)
CONSUME_GENERATOR(gen_bind2, struct ctx2, y, gen_bind1(ctx), consume1)
CONSUME_GENERATOR(gen_bind3, struct ctx2, z, gen_bind2(ctx), consume2)
PUBLIC_FUNC(use_gen, struct ctx2, gen_bind3)

int main() {
    // print() displays the string inside quotation
    myprint("Hello, World %lu %s!\n", sizeof(struct ctx2), "");
    /* union value x = EVAL(struct ctx, stmt3); */
    /* use_loop(); */
    use_gen();

    /* myprint("result=%d\n", *(int*)&x); */
    return 0;
}
