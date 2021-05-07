// Clang
#if defined(__clang__)
#if __clang_major__ >= 13
#define TAIL_CALL(x) __attribute__((musttail)) return (x)
#else
#warning                                                                       \
    "Clang >= 13.0.0 is required to properly apply tail call optimization, which can otherwise lead to stack overflow for loops"
#define TAIL_CALL(x) return (x)
#endif
#define INLINE __attribute__((always_inline))
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

#include <stdbool.h>
typedef char void_val[0];

enum ctrl_tag {
    CTRL_RETURN,
    CTRL_RETURN_NONE,
    CTRL_BREAK,
    CTRL_BREAK_NONE,
    CTRL_YIELD,
    CTRL_SUSPEND,
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

/* MAKE_CTRL_MONAD(void, void_val); */
/* MAKE_CTRL_MONAD(void_val, void_val); */
/* MAKE_CTRL_MONAD(char, char); */
/* MAKE_CTRL_MONAD(int, int); */
/* MAKE_CTRL_MONAD(ull, unsigned long long); */

#define RETURN_X(stmt, _tag, val)                                              \
    return (typeof(stmt(NULL))) { .tag = _tag, .value = (val), }

#define RETURN_EMPTY(stmt, _tag)                                               \
    return (typeof(stmt(NULL))) { .tag = _tag }

#define Yield(stmt, x) RETURN_X(stmt, CTRL_YIELD, x)
#define Return(stmt, x) RETURN_X(stmt, CTRL_RETURN, x)
#define Break(stmt, x) RETURN_X(stmt, CTRL_BREAK, x)
#define BreakNone(stmt) RETURN_EMPTY(stmt, CTRL_BREAK_NONE)
#define ReturnNone(stmt) RETURN_EMPTY(stmt, CTRL_RETURN_NONE)
#define Suspend(stmt) RETURN_EMPTY(stmt, CTRL_SUSPEND)

#define MIN_SIZE(x, y) (sizeof(x) < sizeof(y) ? sizeof(x) : sizeof(y))

#define SET_MONAD_VALUE(m, x)                                                  \
    memcpy((char *)&(m) + offsetof(typeof(m), value), &(x),                    \
           MIN_SIZE(m.value, x))

#define GET_MONAD_VALUE(addr, m)                                               \
    memcpy((char *)(addr), &(m).value, MIN_SIZE(*(addr), (m).value))

#define MAKE_CTX_BEGIN(name) name {
#define MAKE_CTX_END(name)                                                     \
    }                                                                          \
    ;

#define ___BIND(ctx, name, ma, a_mb, a_mb_addr)                                \
    do {                                                                       \
        typeof(ma) __bind_ma;                                                  \
        typeof((a_mb_addr(NULL))) __bind_mb;                                   \
        __bind_ma = (ma);                                                      \
        if (__bind_ma.tag == CTRL_RETURN) {                                    \
            GET_MONAD_VALUE(&ctx->name, __bind_ma);                            \
            TAIL_CALL(a_mb);                                                   \
        } else if (__bind_ma.tag == CTRL_BREAK) {                              \
            GET_MONAD_VALUE(&ctx->name, __bind_ma);                            \
            SET_MONAD_VALUE(__bind_mb, __bind_ma.value);                       \
            __bind_mb.tag = CTRL_RETURN;                                       \
        } else if (__bind_ma.tag == CTRL_RETURN_NONE) {                        \
            TAIL_CALL(a_mb);                                                   \
        } else if (__bind_ma.tag == CTRL_BREAK_NONE) {                         \
            __bind_mb.tag = CTRL_RETURN_NONE;                                  \
        } else if (__bind_ma.tag == CTRL_YIELD ||                              \
                   __bind_ma.tag == CTRL_SUSPEND) {                            \
            if (__bind_ma.tag == CTRL_YIELD) {                                 \
                SET_MONAD_VALUE(__bind_mb, __bind_ma.value);                   \
                GET_MONAD_VALUE(&ctx->name, __bind_ma);                        \
            }                                                                  \
            __bind_mb.tag = __bind_ma.tag;                                     \
            __bind_mb.thunk.ctx = malloc(sizeof(*ctx));                        \
            __bind_mb.thunk.k = (typeof(__bind_mb.thunk.k))(a_mb_addr);        \
            memcpy(__bind_mb.thunk.ctx, ctx, sizeof(*ctx));                    \
        } else {                                                               \
            __bind_mb.tag = __bind_ma.tag;                                     \
        }                                                                      \
        return __bind_mb;                                                      \
    } while (0)

#define __BIND(ctx, name, ma, a_mb) ___BIND(ctx, name, ma, a_mb(ctx), a_mb)

#define BIND_STMT(bound_name, ctx_type, name, ma, a_mb)                        \
    static INLINE typeof((a_mb)(NULL)) bound_name(ctx_type *ctx) {             \
        __BIND(ctx, name, (ma)(ctx), a_mb);                                    \
    }

/* #define BIND_REC_STMT(bound_name, ctx_type, name, a_mb) \ */
/*     static inline typeof(a_mb(NULL)) bound_name(ctx_type *ctx) { \ */
/*         ___BIND(ctx, name, (a_mb)(ctx), bound_name(ctx), a_mb); \ */
/*     } */

#define BIND_EXPR(bound_name, ctx_type, name, expr, a_mb)                      \
    static INLINE typeof((a_mb)(NULL)) bound_name(ctx_type *ctx) {             \
        __BIND(ctx, name, expr, a_mb);                                         \
    }

#define __FUNC(name, ctx_type, expr)                                           \
    typeof(expr(NULL)) __attribute__((flatten)) name(ctx_type *ctx) {          \
        return expr(ctx);                                                      \
    }                                                                          \
    typedef ctx_type __ctx_type_##name;
#define FUNC(name, ctx_type, expr) static __FUNC(name, ctx_type, expr)
#define PUBLIC_FUNC(name, ctx_type, expr) __FUNC(name, ctx_type, expr)
#define FUNC_RET_TYPE(func) typeof((func)(NULL))
#define FUNC_PARAM(name, val) .name = (val)
#define CALL_FUNC(func, ...) (func)(&(__ctx_type_##func){__VA_ARGS__})
#define USER_FUNC_RET_TYPE(func) typeof((__user_##func)(NULL))
#define USER_FUNC_PARAM(name, val) FUNC_PARAM(__param_user_##name, val)
#define USER_FUNC_PARAM_TYPE(func, name)                                       \
    typeof(((__ctx_type___user_##func *)NULL)->__param_user_##name)
#define USER_FUNC_CALL(func, ...) CALL_FUNC(__user_##func, __VA_ARGS__)
/* #define EVAL(ctx_type, m)                                                      \ */
/*     ({                                                                         \ */
/*         ctx_type ctx;                                                          \ */
/*         m(&ctx).value;                                                         \ */
/*     }) */

#define GENERATOR_CONSUMER_STATE(name, producer_type, consumer_type)           \
    CTRL_MONAD(producer_type) __state_gen_##name;                              \
    CTRL_MONAD(consumer_type) __scratch_gen_##name;                      \
    bool __resume_gen_##name;

#define CONSUME_GENERATOR(bound_name, ctx_type, name, gen_expr, a_mb)          \
    static INLINE typeof(((ctx_type *)NULL)->__state_gen_##bound_name)         \
        __consume_gen_##bound_name(ctx_type *ctx) {                            \
        if (ctx->__resume_gen_##bound_name) {                                  \
            if (ctx->__state_gen_##bound_name.thunk.k) {                       \
                void *__thunk_ctx;                                             \
                __thunk_ctx = ctx->__state_gen_##bound_name.thunk.ctx;         \
                ctx->__state_gen_##bound_name =                                \
                    ctx->__state_gen_##bound_name.thunk.k(__thunk_ctx);        \
                free(__thunk_ctx);                                             \
            } else {                                                           \
                return ctx->__scratch_gen_##bound_name;                        \
            }                                                                  \
        } else {                                                               \
            ctx->__state_gen_##bound_name = (gen_expr);                        \
            ctx->__resume_gen_##bound_name = 1;                                \
        }                                                                      \
        if (ctx->__state_gen_##bound_name.tag == CTRL_YIELD)                   \
            ctx->__state_gen_##bound_name.tag = CTRL_RETURN;                   \
        /* Set the continuation to NULL to indicate that the generator will    \
         * not yield anymore. We cannot just check                             \
         * "ctx->__state_gen_##bound_name.tag == CTRL_YIELD" as it has been    \
         * re-written to CTRL_RETURN. */                                       \
        else                                                                   \
            ctx->__state_gen_##bound_name.thunk.k = NULL;                      \
        return ctx->__state_gen_##bound_name;                                  \
    }                                                                          \
    static INLINE typeof((a_mb)(NULL)) __consume_gen_loop2_##bound_name(       \
        ctx_type *ctx);                                                        \
    BIND_STMT(__consume_gen_loop1_##bound_name, ctx_type,                      \
              __scratch_gen_##bound_name, a_mb,                                \
              __consume_gen_loop2_##bound_name);                               \
    BIND_STMT(__consume_gen_loop2_##bound_name, ctx_type, name,                \
              __consume_gen_##bound_name, __consume_gen_loop1_##bound_name);   \
    static INLINE typeof(__consume_gen_loop2_##bound_name(NULL)) bound_name(   \
        ctx_type *ctx) {                                                       \
        ctx->__resume_gen_##bound_name = 0;                                    \
        TAIL_CALL(__consume_gen_loop2_##bound_name(ctx));                      \
    }

#define MAKE_STMT(name, ctx_type, type)                                        \
    static INLINE CTRL_MONAD(type) name(ctx_type *ctx)

#define MAKE_STMT_PROTOTYPE(name, ctx_type, type)                              \
    MAKE_STMT(name, ctx_type, type);
