// Clang
#if defined(__clang__)
#define INLINE __attribute__((always_inline))
#if __clang_major__ >= 13
#define TAIL_CALL(x) __attribute__((musttail)) return (x)
#else
#warning
"Clang >= 13.0.0 is required to properly apply tail call optimization, which can otherwise lead to stack overflow for loops"
#define TAIL_CALL(x) return (x)
#endif
// GCC
#elif defined(__GNUC__) || defined(__GNUG__)
#define TAIL_CALL(x) return (x)
#define INLINE inline
// Ensure we get sibling call elimination, which is critical for loops.
#pragma GCC optimize "-foptimize-sibling-calls"
#endif

#ifdef BPF
#include <linux/slab.h>
#include <linux/stddef.h>
#include <linux/string.h>
#define myprint(s, ...)                                     \
    ({                                                      \
        char _fmt[] = s;                                    \
        bpf_trace_printk_(_fmt, sizeof(_fmt), __VA_ARGS__); \
    })
#define NULL ((void *)0)
#define size_t unsigned long
#define malloc(x) kmalloc(x, GFP_USER)
#define free(x) kfree(x)
#else
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define myprint printf
#endif

enum ctrl_tag {
    CTRL_RETURN,
    CTRL_BREAK,
    CTRL_GENERATOR_YIELD,
};

#define THUNK(name) __THUNK(name)
#define __THUNK(name) thunk_##name
#define MAKE_THUNK(name) \
    typedef struct {     \
        CTRL_MONAD(name) \
        (*k)(void *);    \
    } THUNK(name)

#define SUSPENDED_GENERATOR(name) __SUSPENDED_GENERATOR(name)
#define __SUSPENDED_GENERATOR(name) suspended_gen_##name
#define MAKE_SUSPENDED_GENERATOR(name, type) \
    MAKE_THUNK(name);                        \
    typedef struct {                         \
        CTRL_MONAD(name)                     \
        value;                               \
        THUNK(name)                          \
        thunk;                               \
    } SUSPENDED_GENERATOR(name)

#define CTRL_MONAD(name) __CTRL_MONAD(name)
#define __CTRL_MONAD(name) ctrl_monad_##name
#define MAKE_CTRL_MONAD(name, type) \
    typedef struct {                \
        enum ctrl_tag tag;          \
        type value;                 \
    } CTRL_MONAD(name)

#define RETURN_X(stmt, _tag, val) \
    return (typeof(stmt(NULL))) { .tag = _tag, .value = (val), }

#define RETURN_EMPTY(stmt, _tag) \
    return (typeof(stmt(NULL))) { .tag = _tag }

#define GeneratorYield(stmt, x) RETURN_X(stmt, CTRL_GENERATOR_YIELD, x)
#define GeneratorFinish(stmt) RETURN_EMPTY(stmt, CTRL_BREAK)
#define Return(stmt, x) RETURN_X(stmt, CTRL_RETURN, x)
#define Break(stmt, x) RETURN_X(stmt, CTRL_BREAK, x)

#define MIN_SIZE(x, y) (sizeof(x) < sizeof(y) ? sizeof(x) : sizeof(y))

#define SET_MONAD_VALUE(m, x)                               \
    memcpy((char *)&(m) + offsetof(typeof(m), value), &(x), \
           MIN_SIZE(m.value, x))

#define GET_MONAD_VALUE(addr, m) \
    memcpy((char *)(addr), &(m).value, MIN_SIZE(*(addr), (m).value))

#define MAKE_CTX_BEGIN(name) name {
#define MAKE_CTX_END(name) \
    }                      \
    ;

#define ___BIND(ctx, name, ma, a_mb, a_mb_addr)                                     \
    do {                                                                            \
        typeof(ma) __bind_ma;                                                       \
        typeof((a_mb_addr(NULL))) __bind_mb;                                        \
        __bind_ma = (ma);                                                           \
        if (__bind_ma.tag == CTRL_RETURN) {                                         \
            GET_MONAD_VALUE(&ctx->name, __bind_ma);                                 \
            TAIL_CALL(a_mb);                                                        \
        } else if (__bind_ma.tag == CTRL_BREAK) {                                   \
            GET_MONAD_VALUE(&ctx->name, __bind_ma);                                 \
            SET_MONAD_VALUE(__bind_mb, __bind_ma.value);                            \
            __bind_mb.tag = CTRL_RETURN;                                            \
        } else if (__bind_ma.tag == CTRL_GENERATOR_YIELD) {                         \
            GET_MONAD_VALUE(&ctx->name, __bind_ma);                                 \
            SET_MONAD_VALUE(__bind_mb, __bind_ma.value);                            \
            __bind_mb.tag = __bind_ma.tag;                                          \
            __bind_mb.value.thunk.k = (typeof(__bind_mb.value.thunk.k))(a_mb_addr); \
        } else {                                                                    \
            __builtin_unreachable();                                                \
        }                                                                           \
        return __bind_mb;                                                           \
    } while (0)

#define __BIND(ctx, name, ma, a_mb) ___BIND(ctx, name, ma, a_mb(ctx), a_mb)

#define BIND_STMT(bound_name, ctx_type, name, ma, a_mb)            \
    static INLINE typeof((a_mb)(NULL)) bound_name(ctx_type *ctx) { \
        __BIND(ctx, name, (ma)(ctx), a_mb);                        \
    }

/* #define BIND_REC_STMT(bound_name, ctx_type, name, a_mb) \ */
/*     static inline typeof(a_mb(NULL)) bound_name(ctx_type *ctx) { \ */
/*         ___BIND(ctx, name, (a_mb)(ctx), bound_name(ctx), a_mb); \ */
/*     } */

#define BIND_EXPR(bound_name, ctx_type, name, expr, a_mb)          \
    static INLINE typeof((a_mb)(NULL)) bound_name(ctx_type *ctx) { \
        __BIND(ctx, name, expr, a_mb);                             \
    }

#define __FUNC(name, ctx_type, expr)                                  \
    typeof(expr(NULL)) __attribute__((flatten)) name(ctx_type *ctx) { \
        return expr(ctx);                                             \
    }                                                                 \
    typedef ctx_type __ctx_type_##name;
#define FUNC(name, ctx_type, expr) static __FUNC(name, ctx_type, expr)
#define PUBLIC_FUNC(name, ctx_type, expr) __FUNC(name, ctx_type, expr)
#define FUNC_RET_TYPE(func) typeof((func)(NULL))
#define FUNC_PARAM(name, val) .name = (val)
#define CALL_FUNC(func, ...) (func)(&(__ctx_type_##func){__VA_ARGS__})
#define USER_FUNC_RET_TYPE(func) typeof((__user_##func)(NULL))
#define USER_FUNC_PARAM(name, val) FUNC_PARAM(__param_user_##name, val)
#define USER_FUNC_PARAM_TYPE(func, name) \
    typeof(((__ctx_type___user_##func *)NULL)->__param_user_##name)
#define USER_FUNC_CALL(func, ...) CALL_FUNC(__user_##func, __VA_ARGS__)
/* #define EVAL(ctx_type, m)                                                      \ */
/*     ({                                                                         \ */
/*         ctx_type ctx;                                                          \ */
/*         m(&ctx).value;                                                         \ */
/*     }) */

/* #define GENERATOR_CONSUMER_STATE(name, producer_type, consumer_type)           \ */
/*     CTRL_MONAD(producer_type) __state_gen_##name;                              \ */
/*     CTRL_MONAD(consumer_type) __scratch_gen_##name;                      \ */
/*     bool __resume_gen_##name; */

/* #define CONSUME_GENERATOR(bound_name, ctx_type, name, gen_expr, a_mb)          \ */
/*     static INLINE typeof(((ctx_type *)NULL)->__state_gen_##bound_name)         \ */
/*         __consume_gen_##bound_name(ctx_type *ctx) {                            \ */
/*         if (ctx->__resume_gen_##bound_name) {                                  \ */
/*             if (ctx->__state_gen_##bound_name.thunk.k) {                       \ */
/*                 void *__thunk_ctx;                                             \ */
/*                 __thunk_ctx = ctx->__state_gen_##bound_name.thunk.ctx;         \ */
/*                 ctx->__state_gen_##bound_name =                                \ */
/*                     ctx->__state_gen_##bound_name.thunk.k(__thunk_ctx);        \ */
/*                 free(__thunk_ctx);                                             \ */
/*             } else {                                                           \ */
/*                 return ctx->__scratch_gen_##bound_name;                        \ */
/*             }                                                                  \ */
/*         } else {                                                               \ */
/*             ctx->__state_gen_##bound_name = (gen_expr);                        \ */
/*             ctx->__resume_gen_##bound_name = 1;                                \ */
/*         }                                                                      \ */
/*         if (ctx->__state_gen_##bound_name.tag == CTRL_GENERATOR_YIELD)                   \ */
/*             ctx->__state_gen_##bound_name.tag = CTRL_RETURN;                   \ */
/*         /\* Set the continuation to NULL to indicate that the generator will    \ */
/*          * not yield anymore. We cannot just check                             \ */
/*          * "ctx->__state_gen_##bound_name.tag == CTRL_GENERATOR_YIELD" as it has been    \ */
/*          * re-written to CTRL_RETURN. *\/                                       \ */
/*         else                                                                   \ */
/*             ctx->__state_gen_##bound_name.thunk.k = NULL;                      \ */
/*         return ctx->__state_gen_##bound_name;                                  \ */
/*     }                                                                          \ */
/*     static INLINE typeof((a_mb)(NULL)) __consume_gen_loop1_##bound_name(       \ */
/*         ctx_type *ctx);                                                        \ */
/*     static INLINE typeof(__consume_gen_loop1_##bound_name(NULL)) bound_name( \ */
/*             ctx_type *ctx) {                                                \ */
/*             ctx->__resume_gen_##bound_name = 0;                                \ */
/*             TAIL_CALL(__consume_gen_loop1_##bound_name(ctx));                \ */
/*     } \ */
/*     BIND_STMT(__consume_gen_loop1_##bound_name, ctx_type, name,                \ */
/*               __consume_gen_##bound_name, __consume_gen_loop2_##bound_name);   \ */
/*     BIND_STMT(__consume_gen_loop2_##bound_name, ctx_type,                \ */
/*               __scratch_gen_##bound_name, a_mb,                                \ */
/*               __consume_gen_loop1_##bound_name);                        \ */

#define MAKE_STMT(name, ctx_type, type) \
    static INLINE type name(ctx_type *ctx)

#define MAKE_STMT_PROTOTYPE(name, ctx_type, type) \
    MAKE_STMT(name, ctx_type, type);
