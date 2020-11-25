#!/usr/bin/env python3
from collections import namedtuple, OrderedDict
import subprocess as subp
import weakref
import json
import sys
import re


FuncLine = namedtuple('FuncLine',
    'Children Self Command SharedObject Type Symbol')
InstLine = namedtuple('InstLine',
    'percent loc op args dst comment')
BasicBlock = namedtuple('BasicBlock',
    'percent loc insts next_ jump')
REPORTCMD = 'perf report --stdio -q -g none -t \; --percent-limit 1'
ANNOTACMD = 'perf annotate --stdio --no-source "%s"'
INSTREGEX = re.compile(r'([^<#]+)(<[^#]+>\s*)?(#.+)?')


def get_funcs():
    lines = subp.getoutput(REPORTCMD).splitlines()
    for ln in lines:
        ln = ln.strip()
        if not ln: continue
        words = [i.strip() for i in ln.split(';')]
        child, self, cmd, obj, sym = words
        child, self = float(child[:-1]), float(self[:-1])
        typ, sym = sym.split(maxsplit=1)
        yield FuncLine(child, self, cmd, obj, typ, sym)


def get_insts(sym):
    print(ANNOTACMD % sym)
    lines = subp.getoutput(ANNOTACMD % sym).splitlines()
    for ln in lines:
        try:
            ln = ln.strip()
            if '|' in ln or ':' not in ln: continue
            segs = [i.strip() for i in ln.split(':', maxsplit=2)]
            if not segs or not segs[0]: continue
            percent, loc, inst = segs
            percent = float(percent)
            inst, dst, comment = INSTREGEX.fullmatch(inst).groups()
            dst, comment = (dst or '').strip(), (comment or '').strip()
            op, args, *_ = inst.strip().rsplit(maxsplit=1) + ['']
            yield InstLine(percent, loc, op, args, dst, comment)
        except:
            print(ln)
            #raise
            return


def is_terminal_inst(inst):
    return inst.op.startswith('jmp') \
        or inst.op.startswith('ret') \
        or '__cxa_call_unexpected' in inst.dst \
        or '_Unwind_Resume' in inst.dst


def get_bbs(sym):
    # collect insts
    insts, locs = [], set()
    for item in get_insts(sym):
        insts.append(item)
        locs.add(item.loc)
    # collect BB heads
    headset, ishead = set(), True
    for item in insts:
        if ishead:
            headset.add(item.loc)
            ishead = False
        if item.dst and item.args in locs:
            headset.add(item.args)
            ishead = True
        elif is_terminal_inst(item):
            ishead = True
    # enumerate BBs
    def foo():
        newsym, _ = re.subn(r'\W', '_', sym)
        cur = BasicBlock(0, 'START_' + newsym, [], '', '')
        for item in insts:
            if item.loc in headset:
                if cur:
                    yield cur._replace(next_=item.loc)
                cur = BasicBlock(0, item.loc, [], '', '')
            cur.insts.append(item)
            if item.dst and item.args in headset:
                cur = cur._replace(jump=item.args)
            if is_terminal_inst(item):
                yield cur
                cur = None
    # calc percentage
    for bb in foo():
        yield bb._replace(percent=sum(i.percent for i in bb.insts))


def to_json(data):
    if isinstance(data, list):
        return [to_json(i) for i in data]
    elif hasattr(data, '_asdict'):
        return {k: to_json(v) for k, v in data._asdict().items()}
    else:
        return data


if __name__ == '__main__':
    assert len(sys.argv) == 2
    fn = sys.argv[1]
    temp = []
    problematics = ['[unknown', 'libc-', 'ld-']
    for func in get_funcs():
        if all(m not in func.SharedObject for m in problematics):
            temp.extend(get_bbs(func.Symbol))
    with open(fn, 'w') as f:
        json.dump(to_json(temp), f)

