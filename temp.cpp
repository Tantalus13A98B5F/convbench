template <int START, int END, int STEP=1, bool VALID=(START < END)>
struct RangeCore {
    template <typename Func>
    static void exec(Func func) {
        func(START, END);
        RangeCore<START+STEP, END, STEP>::exec(func);
    }
};

template <int START, int END, int STEP>
struct RangeCore<START, END, STEP, false> {
    template <typename Func>
    static void exec(Func func) {}
};

template <int START, int END, int STEP=1>
struct Range {
    template <typename Func>
    static void exec(Func func) {
        RangeCore<START, END, STEP>::exec(func);
    }
};

template <int END>
struct Range<0, END> {
    template <typename Func>
    static void exec(Func func) {
        RangeCore<0, END>::exec(func);
    }
};

#define STATIC_FOR(s, e) Range<(s), (e)>::exec
#include <iostream>

int main() {
    int a = 0;
    STATIC_FOR (0, 4) ([&](int idx, int lim) {
        a += idx;
    });
    std::cout << a << std::endl;
    return 0;
}
