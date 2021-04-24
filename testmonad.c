// usr/bin/clang -Wall -Wextra -Wno-unused-parameter -O3 -std=gnu11 "$0" &&

// ./a.out; exit
//

#include "testmonad.h"

MAKE_STMT(stmt1, struct ctx, int) {
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

MAKE_CTX_BEGIN(ctx2)
GENERATOR_CONSUMER_STATE(gen_bind2, int);
GENERATOR_CONSUMER_STATE(gen_bind3, int);
GENERATOR_CONSUMER_STATE(loop1, int);
int y;
int x;
int z;
MAKE_CTX_END(ctx2)

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
    if (ctx->x < 100)
        Yield(__2gen1, ctx->x + 1);
    else
        Suspend(__2gen1);
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
    typeof(use_gen()) x = use_gen();
    printf("tag=%d\n", x.tag);

    /* myprint("result=%d\n", *(int*)&x); */
    return 0;
}
