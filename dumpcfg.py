#!/usr/bin/env python3
from collections import OrderedDict
from contextlib import redirect_stdout
import subprocess as subp
import weakref
import sys
import re

rsect = re.compile(r'Disassembly of section (.+):')
rfunc = re.compile(r'(\w+) \<(.+)\>:')
rinst = re.compile(r'\s*(\w+):\t([^\t]+)(?:\t(\w+)\s*([^#]*?)\s*(#.+)?)?')
rjdst = re.compile(r'(\w+) \<([^+]+)(\+\w+)?\>')
uncond_terminal = ['ret', 'jmp']


class Inst:
    @classmethod
    def trymatch(cls, line):
        m = rinst.fullmatch(line)
        if m and m.group(3):
            return cls(m)

    def __init__(self, mobj):
        self.loc, self.hex_, self.op, self.arg, self.comment = mobj.groups()
        m2 = rjdst.fullmatch(self.arg)
        self.jloc, self.jfunc, self.joff = m2.groups() if m2 else ('', '', '')

    def __repr__(self):
        return '<Inst "{} {}">'.format(self.op, self.arg)


def proclines(fobj):
    ret, cursec, curfunc = OrderedDict(), None, None
    for ln in fobj:
        ln = ln.rstrip()
        if not ln: continue
        # if section
        m = rsect.fullmatch(ln)
        if m:
            (secname,) = m.groups()
            ret[secname] = cursec = OrderedDict()
            #print('into section:', secname)
            continue
        # if function
        m = rfunc.fullmatch(ln)
        if m:
            loc, funcname = m.groups()
            cursec[funcname] = curfunc = OrderedDict()
            #print('into function:', funcname)
            continue
        # if instruction
        m = Inst.trymatch(ln)
        if m:
            curfunc[m.loc] = m
    return ret


class BasicBlock:
    def __init__(self, loc, prev):
        self.loc = loc
        self.insts = []
        self.prev = weakref.ref(prev) if prev else None
        if prev and prev.next_ is None:
            prev.next_ = self
        self.next_ = self.jump = None

    def __repr__(self):
        fmt = lambda x: x.loc if x else None
        return '<BB@{} next={} jump={}>'.format(
            self.loc, fmt(self.next_), fmt(self.jump))


def buildcfg(func):
    anystartswith = lambda x, s: any(x.startswith(i) for i in s)
    # find BB heads
    head, needhead = set(), True
    for loc, inst in func.items():
        if needhead:
            needhead = False
            head.add(loc)
        if inst.jloc in func:
            head.add(inst.jloc)
            needhead = True
        if anystartswith(inst.op, uncond_terminal):
            needhead = True
    # create BBs
    bbs, curbb = OrderedDict(), None
    for loc, inst in func.items():
        if loc in head:
            bbs[loc] = curbb = BasicBlock(loc, curbb)
        curbb.insts.append(inst)
        if inst.jloc in func:
            curbb.jump = inst.jloc
        if anystartswith(inst.op, uncond_terminal):
            curbb.next_ = False
    # fix jump
    for v in bbs.values():
        if v.jump:
            v.jump = bbs[v.jump]
    return bbs


def drawcfg(bbs, f=None):
    if f:
        with redirect_stdout(f):
            drawcfg(bbs)
        return
    print('digraph G {')
    for bb in bbs.values():
        label = bb.loc + ':\\n' + \
                '\\n'.join('%s %s' % (i.op, i.arg) for i in bb.insts)
        bbloc = 'B' + bb.loc
        print(bbloc, '[shape=box,label="%s"];' % label)
        if bb.next_:
            print(bbloc, '->', 'B' + bb.next_.loc, ';')
        if bb.jump:
            print(bbloc, '->', 'B' + bb.jump.loc, '[style=dotted];')
    print('}')


if __name__ == '__main__':
    binname, funcname, outfn = sys.argv[1:]
    txt = subp.getoutput('objdump -Cd %s' % binname).splitlines()
    fobj = proclines(txt)
    funclist = [v for k, v in fobj['.text'].items() if funcname in k]
    assert len(funclist) == 1
    bbs = buildcfg(funclist[0])
    with open(outfn, 'w') as f:
        drawcfg(bbs, f)

