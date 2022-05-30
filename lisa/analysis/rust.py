# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2022, ARM Limited and contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

""" Rust Analysis Module """

import heapq
import operator
import multiprocessing
import threading
import subprocess
import functools
import contextlib
import itertools
import os
import time
import json
from pathlib import Path
import uuid
import shlex
import copy
from collections.abc import Mapping
import inspect

import ujson
import pandas as pd

from lisa.analysis.base import TraceAnalysisBase
from lisa.trace import TxtTraceParser, TraceView, TraceEventChecker, AndTraceEventChecker, OrTraceEventChecker, OptionalTraceEventChecker, DynamicTraceEventChecker, _CacheDataDesc
from lisa.analysis._rust_fast_import import _json_line, _json_record
from lisa.utils import mp_spawn_pool, Loggable, nullcontext, measure_time, FrozenDict, get_nested_key, group_by_value
from lisa._assets import HOST_BINARIES
from lisa.version import VERSION_TOKEN
from lisa.datautils import series_update_duplicates


_RUST_ANALYSIS_PATH = HOST_BINARIES['lisa-rust-analysis']


class _PipeSafePopen:
    def __init__(self, popen):
        self.popen = popen

    def __enter__(self):
        return self.popen.__enter__()

    def __exit__(self, *args, **kwargs):
        try:
            return self.popen.__exit__(*args, **kwargs)
        except BrokenPipeError:
            pass


def _map_best_effort(f, xs):
    exceps = []
    for x in xs:
        try:
            f(x)
        except Exception as e:
            exceps.append(e)
        else:
            exceps.append(None)

    if all(exceps):
        raise exceps[0]


class _TeeFile:
    def __init__(self, *fs):
        self.fs = fs

    def write(self, data) :
        _map_best_effort(lambda f: f.write(data), self.fs)

    def flush(self) :
        _map_best_effort(lambda f: f.flush(), self.fs)

    def close(self) :
        _map_best_effort(lambda f: f.close(), self.fs)

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _event_checker_from_json(mapping):
    k, v = mapping.popitem()
    if k == 'single':
        return TraceEventChecker(v)
    else:
        classes = {
            'and': AndTraceEventChecker,
            'or': OrTraceEventChecker,
            'optional': OptionalTraceEventChecker,
            'dynamic': DynamicTraceEventChecker,
        }
        try:
            cls = classes[k]
        except KeyError:
            raise ValueError(f'Unknown trace event checker type: {k}')
        else:
            return cls(
                list(map(_event_checker_from_json, v))
            )

def _post_process_data(data, normalize_time):
    schema = data['schema']
    value = data['value']
    title = schema['title']

    if title.startswith('Table_'):
        table = value
        columns = table['columns']
        row_schema = schema['properties']['data']['items']
        assert len(columns) > 0

        def _deref(_schema):
            # Expand the $ref. We do not expect recursive types.
            if tuple(_schema) == ('$ref',):
                ref = _schema['$ref']
                path = [x for x in ref.split('/') if x != '#']
                return (path[-1], get_nested_key(schema, path))
            else:
                return (None, _schema)

        def deref(_schema):
            new = (None, _schema)
            old = (None, None)
            while old[1] != new[1]:
                old = new
                new = _deref(new[1])

            # The fix point has name == None, so we backtrack one step to get
            # the actual name
            return (old[0], new[1])

        class Type:
            def __init__(self, name):
                self.name = name

            def __repr__(self):
                return str(self)

        class NewType(Type):
            def __init__(self, name, typ):
                super().__init__(name)
                self.typ = typ

            def __str__(self):
                name = self.name
                typ = self.typ
                if name:
                    return f'{name}({typ})'
                else:
                    return typ

        class BasicType(Type):
            def __init__(self, typ):
                super().__init__(typ)
                self.typ = typ

            def __str__(self):
                return self.typ


        class UnitType(Type):
            def __init__(self):
                super().__init__('()')

            def __str__(self):
                return self.name


        class SumType(Type):
            def __init__(self, name, ctors):
                name = name or '<anonymous>'
                super().__init__(name)

                self.ctors = dict(ctors)

            def __str__(self):
                ctors = ', '.join(
                    (
                        f'{name}({param})'
                        if not isinstance(param, UnitType) else
                        name
                    )
                    for name, param in self.ctors.items()
                )
                return f'{self.name}{{{ctors}}}'


        class ProductType(Type):
            def __init__(self, name, items, names=None):
                super().__init__(name)
                self.items = list(items)
                self.names = names or list(range(len(items)))
                assert len(self.items) == len(self.names)

            @property
            def named_items(self):
                return zip(self.names, self.items)

            def __str__(self):
                name = self.name or ''
                params = ', '.join(map(str, self.items))
                return f'{name}({params})'


        def infer_adt(name, schema):
            _name, schema = deref(schema)
            name = name or _name

            if 'items' in schema:
                items = [
                    infer_adt(None, _schema)
                    for _schema in schema['items']
                ]
                return ProductType(name, items)
            elif 'properties' in schema:
                items = [
                    (_name, infer_adt(None, _schema))
                    for _name, _schema in schema['properties'].items()
                ]
                if items:
                    names, items = zip(*items)
                else:
                    names = []
                    items = []
                return ProductType(name, items, names)
            elif 'oneOf' in schema:
                variants = schema['oneOf']
                ctors = {}
                for variant in variants:
                    if 'enum' in variant:
                        _ctors = variant['enum']
                        ctors.update(dict.fromkeys(_ctors, UnitType()))
                    elif variant['type'] == 'object':
                        ctor, = variant['required']
                        ctors[ctor] = infer_adt(None, variant['properties'][ctor])
                return SumType(name, ctors)
            else:
                try:
                    typ = schema['format']
                except KeyError:
                    typ = schema['type']

                if name is None:
                    return BasicType(typ)
                else:
                    return NewType(name, BasicType(typ))

        def expand_adt(adt):
            if isinstance(adt, BasicType):
                return ([(None, adt.typ, adt)], lambda x: [x])
            elif isinstance(adt, NewType):
                [(typ, col, _)], expand = expand_adt(adt.typ)
                return ([(typ, col, adt)], expand)
            elif isinstance(adt, UnitType):
                return ([], None)
            elif isinstance(adt, ProductType):
                if adt.items:
                    cols, expands = zip(*map(expand_adt, adt.items))
                    def expand(x):
                        return [
                            __x
                            for expand, _x in zip(expands, x)
                            for __x in expand(_x)
                        ]

                    cols = [
                        (
                            f'{i}.{col}' if col else str(i),
                            typ,
                            adt
                        )
                        for i, _cols in zip(adt.names, cols)
                        for col, typ, adt in _cols
                    ]
                    return (cols, expand)
                else:
                    raise ValueError(f'Product type with no constructor: {adt}')
            elif isinstance(adt, SumType):
                if adt.ctors:
                    variants = {
                        ctor: (cols, expand)
                        for (ctor, (cols, expand)) in (
                            (ctor, expand_adt(variant_adt))
                            for ctor, variant_adt in adt.ctors.items()
                        )
                        if cols
                    }
                    variants = sorted(variants.items())

                    cols = [
                        (f'{name}.{col}' if col else name, typ, adt)
                        for name, (cols, _) in variants
                        for col, typ, adt in cols
                    ]
                    nr_cols = len(cols)

                    variant_names, _ = zip(*variants)
                    variant_cols = [cols for _, (cols, _) in variants]
                    variant_index = list(itertools.accumulate([0] + list(map(len, variant_cols))))
                    variant_index = {
                        name: (index, expand)
                        for name, index, (_, (_, expand)) in zip(variant_names, variant_index, variants)
                    }

                    def expand(x):
                        if isinstance(x, Mapping):
                            k, = x
                            v = x[k]
                            index, expand = variant_index[k]
                            res = [None] * (nr_cols + 1)
                            res[0] = k
                            v = expand(v)
                            res[index + 1: index + 1 + len(v)] = v
                            return res
                        else:
                            res = (nr_cols + 1) * [None]
                            res[0] = x
                            return res

                    cols = [(None, 'category', adt)] + cols
                    return (cols, expand)
                else:
                    raise ValueError(f'Sum type with no constructor: {adt}')
            else:
                raise ValueError(f'Unknown type: {adt}')


        # Each row is a tuple of values, so it will map to a
        # ProductType
        row_adt = infer_adt(None, row_schema)
        assert len(row_adt.items) == len(columns)
        row_adt.names = columns
        typed_cols, expand = expand_adt(row_adt)
        columns, _, _ = zip(*typed_cols)


        rows = pd.json_normalize(table['data'])
        df = pd.DataFrame.from_records(
            map(expand, table['data']),
            columns=columns,
        )

        # Use nullable types
        json_pd_dtypes = {
            'uint8': 'UInt8',
            'uint16': 'UInt16',
            'uint32': 'UInt32',
            'uint64': 'UInt64',
            'int8': 'Int8',
            'int16': 'Int16',
            'int32': 'Int32',
            'int64': 'Int64',
            'string': 'string',
            'bool': 'bool',
            'category': 'category',
        }
        dtypes = {
            col: json_pd_dtypes[typ]
            for col, typ, adt in typed_cols
            if typ in json_pd_dtypes
        }

        df = df.astype(dtypes)

        def _fixup_ts(series):
            series /= 1e9
            if normalize_time is not None:
                series -= normalize_time

            # Deduplicate timestamps, since pandas cannot deal properly with
            # duplicated indices
            return series_update_duplicates(series)

        fixups = {
            'Timestamp': _fixup_ts,
        }

        for col, _, adt in typed_cols:
            try:
                f = fixups[adt.name]
            except KeyError:
                pass
            else:
                df[col] = f(df[col])

        df.set_index(columns[0], inplace=True)
        return (df, 'parquet')
    else:
        return (value, 'json')


class RustAnalysis(TraceAnalysisBase, Loggable):
    """
    Support for Rust analysis.
    """

    name = '_rust'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ana_map = ujson.loads(self._call_binary('list'))
        self._metadata = {
            name: dict(
                event_checker=_event_checker_from_json(spec['eventreq'])
            )
            for name, spec in ana_map.items()
        }

    @classmethod
    def _call_binary(cls, subcommand, *cli_args):
        cmd = [_RUST_ANALYSIS_PATH, subcommand, *cli_args]
        pretty_cmd = ' '.join(map(lambda x: shlex.quote(str(x)), cmd))
        try:
            completed = subprocess.run(
                cmd,
                universal_newlines=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as e:
            cls.get_logger().debug(f"Command {cmd} failed: {e.stderr}")

        cls.get_logger().debug(f'Rust analysis stderr for {pretty_cmd}:\n{completed.stderr}')
        return completed.stdout

    def call_anas(self, analyses):
        trace = self.trace
        cache = trace._cache
        window = self._trace_window

        def resolve(spec):
            name = spec['name']
            try:
                metadata = self._metadata[name]
            except KeyError:
                raise ValueError(f'Unknown analysis "{name}", available analysis are: {sorted(self._metadata.keys())}')
            else:
                checker = metadata['event_checker']
                return (checker, spec)

        def make_cache_desc(spec, fmt):
            name = spec['name']
            args = spec['args']

            cache_spec = dict(
                func=RustAnalysis.call_anas.__qualname__,
                module=RustAnalysis.call_anas.__module__,

                ana_name=name,
                ana_args=copy.deepcopy(args),
                trace_state=trace.trace_state,
            )
            return _CacheDataDesc(spec=cache_spec, fmt=fmt)

        def try_cache(spec):
            for fmt in ('parquet', 'json'):
                cache_desc = make_cache_desc(spec, fmt)
                try:
                    data = cache.fetch(cache_desc)
                except KeyError:
                    continue
                else:
                    return data

            raise KeyError('Could not find data in cache')

        analyses = list(analyses)
        if analyses:

            analyses = list(map(FrozenDict, analyses))
            exceps = {}
            results = {}
            not_in_cache = list()
            for spec in analyses:
                try:
                    data = try_cache(spec)
                except KeyError:
                    not_in_cache.append(spec)
                else:
                    results[spec] = data

            if not_in_cache:
                checkers, cli_specs = zip(*map(resolve, not_in_cache))
                checker = AndTraceEventChecker(checkers)
                cli_spec = json.dumps(list(map(dict, cli_specs)))
                if window is None:
                    window = '"none"'
                else:
                    window = f'{{"time": [{window[0]}, {window[1]}]}}'

                with measure_time() as measure:
                    stdout = self._do_call_ana(event_checker=checker, cli_args=[cli_spec, window])

                # Consider that running multiple analysis takes the same time as
                # running just one, as the time is dominated by iterating over the
                # events.
                compute_cost = measure.delta
                computed = ujson.loads(stdout)

                for spec, data in zip(not_in_cache, computed):
                    try:
                        data = data['ok']
                    except KeyError:
                        err = data['err']

                        if isinstance(err, str) and err.startswith('invalid type'):
                            excep = TypeError(err)
                        else:
                            excep = ValueError(err)

                        exceps[spec] = excep
                    else:
                        data, fmt = _post_process_data(
                            data=data,
                            normalize_time=(self.trace.basetime if trace.normalize_time else None),
                        )
                        cache_desc = make_cache_desc(spec=spec, fmt=fmt)
                        cache.insert(cache_desc, data, compute_cost=compute_cost, write_swap=True)
                        results[spec] = data

            return [
                (results.get(spec), exceps.get(spec))
                for spec in analyses
            ]
        else:
            return []

    def _do_call_ana(self, event_checker, cli_args):
        events = set(event_checker.get_all_events())
        trace = self.trace

        def run_ana(populated, json_path):
            if populated and json_path:
                # Checking the available events on this path should be cheap
                # since the available events have been updated on the other
                # path.
                event_checker.check_events(trace.available_events)
                stdout = self._call_binary('run', json_path, *cli_args)
            else:
                stdout, available_events = self._create_json_and_call_ana(
                    events=events,
                    cli_args=cli_args,
                    json_path=json_path,
                )
                if available_events is not None:
                    # Record the available events on the Trace object so that the
                    # other path can get them cheaply, as well as any other code in
                    # LISA
                    trace._update_parseable_events({
                        event: True
                        for event in available_events
                    })
                    # Check the events after the fact. The analysis is "pure", i.e.
                    # we only care about the JSON it returns so it's ok if the
                    # result is non-sensical since we will discard it.
                    event_checker.check_events(available_events)

            return stdout

        return self._with_json_trace(events=events, f=run_ana)

    @TraceAnalysisBase.cache(fmt='disk-only', ignored_params=['f'])
    def _with_json_trace(self, json_path, events, f):
        if json_path is None:
            return f(False, None)
        else:
            return f(os.path.exists(json_path), json_path)

    def _create_json_and_call_ana(self, events, json_path, cli_args):
        logger = self.logger
        cmd = [_RUST_ANALYSIS_PATH, 'run', '-', *cli_args]
        pretty_cmd = ' '.join(map(lambda x: shlex.quote(str(x)), cmd))

        def reader_f(f, into):
            try:
                while True:
                    time.sleep(0.05)
                    x = f.readlines()
                    if x:
                        into.extend(x)
            # Exit when encountering:
            # ValueError: I/O operation on closed file.
            except ValueError:
                pass

        logger.debug(f'Running rust analysis: {pretty_cmd}')
        stdout = []
        stderr = []
        popen = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        popen = _PipeSafePopen(popen)

        if json_path is None:
            @contextlib.contextmanager
            def cm():
                with popen as p:
                    yield p.stdin
        else:
            @contextlib.contextmanager
            def cm():
                with popen as p, open(json_path, 'w') as j:
                    with _TeeFile(p.stdin, j) as tf:
                        yield tf

        with popen as p, cm() as f:

            retcode = 0
            stdout_thread = threading.Thread(target=reader_f, args=(p.stdout, stdout), daemon=True)
            stderr_thread = threading.Thread(target=reader_f, args=(p.stderr, stderr), daemon=True)
            try:
                stdout_thread.start()
                stderr_thread.start()

                available_events = self._dump_json_lines(events=events, f=f)

            # We get BrokenPipeError for stdin in case the binary stops before
            # having consuming all the input we try to feed it.
            except BrokenPipeError:
                available_events = None
            finally:
                try:
                    f.flush()
                    f.close()
                except BrokenPipeError:
                    pass
                retcode = p.wait()

                p.stdout.close()
                p.stderr.close()

                def is_alive(thread):
                    return False if thread is None else thread.is_alive()

                while not all(map(is_alive, (stdout_thread, stderr_thread))):
                    time.sleep(0.01)
                stdout = ''.join(stdout)
                stderr = ''.join(stderr)

        logger.debug(f'Rust analysis stderr:\n{stderr}')
        if retcode:
            raise subprocess.CalledProcessError(retcode, pretty_cmd, output=stdout, stderr=stderr)
        else:
            return (stdout, available_events)

    @property
    def _full_trace(self):
        def full(trace):
            if isinstance(trace, TraceView):
                return full(trace.base_trace)
            else:
                return trace
        return full(self.trace)

    @property
    def _trace_window(self):
        trace = self.trace
        if isinstance(trace, TraceView):
            window = trace.window
            if trace.normalize_time:
                window = map(lambda x: x + trace.basetime, window)
            window = map(lambda x: int(x * 1e9), window)
            return tuple(window)
        else:
            return None

    def _dump_json_lines(self, events, f):
        if self._full_trace.trace_path.endswith('.dat'):
            return self._dat_dump_json(events, f)
        else:
            return self._generic_dump_json(events, f)

    def _generic_dump_json(self, events, f):
        def filternan(x):
            return {
                k: v
                for k, v in x.items()
                if not pd.isna(v)
            }

        def make_ts(seconds):
            return int(seconds * 1e9)

        def df_to_records(event, df):
            df = df.copy(deep=False)
            df['__type'] = event
            df['__ts'] = (df.index * 1e9).astype('uint64')
            # Remove any NA since it cannot be serialized to JSON,
            # and means that the data is missing anyway.
            return map(filternan, df.to_dict(orient='records'))

        def event_records(trace, events):
            # Merge the stream of events coming
            # from all dataframes based on the timestamp.
            return heapq.merge(
                *(
                    df_to_records(event, trace.df_event(event))
                    for event in events
                ),
                key=operator.itemgetter('__ts'),
            )

        trace = self._full_trace
        data = event_records(trace, events)
        start = {
            '__type': '__lisa_event_stream_start',
            '__ts': make_ts(trace.start),
        }
        end = {
            '__type': '__lisa_event_stream_end',
            '__ts': make_ts(trace.end),
        }
        data = itertools.chain([start], data, [end])

        for item in data:
            ujson.dump(item, f, reject_bytes=False, ensure_ascii=False, indent=0)
            f.write('\n')

        # If we made it this far, we have all the events that we need
        return set(events)

    def _dat_dump_json(self, events, f):
        """
        Specialized trace.dat handling that is much faster than
        :meth:`_generic_dump_json` as it does not require parsing the
        dataframes for each event prior to creating the JSON.

        .. note:: This might yield a different result since these events will
            not undergo sanitization as they do when parsed by
            :class:`lisa.trace.Trace` into dataframes. Problematic cases are
            usually unsupported, as it is the result of using things like
            bitmask in events which is too painful to support across the
            variety of data formats. If confronted with those, register your
            own event on the tracepoint and "unroll" the event (e.g. one event
            emitted per CPU in the cpumask).
        """
        trace = self._full_trace

        cmd = TxtTraceParser._tracecmd_report(
            path=trace.trace_path,
            events=events,
        )

        bufsize = 10 * 1024 * 1024
        popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=bufsize)

        dump_record = lambda record: f.write(ujson.dumps(record, reject_bytes=False, ensure_ascii=False, indent=0) + '\n')

        with popen as p, mp_spawn_pool() as pool:
            lines = p.stdout

            # Dump the event stream start event
            for first_line in lines:
                try:
                    record = _json_record(first_line)
                except ValueError:
                    continue
                else:
                    dump_record({
                        '__type': '__lisa_event_stream_start',
                        '__ts': record['__ts'],
                    })
                    dump_record(record)
                    break


            # Dump the events by streaming the output of trace-cmd report into
            # a set of workers
            available_events = set()
            record = None
            for record, txt in pool.imap(_json_line, lines, chunksize=512):
                available_events.add(record['__type'])
                f.write(txt + '\n')

            # Dump the event stream end event
            if record is not None:
                dump_record({
                    '__type': '__lisa_event_stream_end',
                    '__ts': record['__ts'],
                })

            return {
                event.decode()
                for event in available_events
            }

    def _run_coros(self, coros):
        """
        Run coroutines in lockstep, and execute the analyses request for each
        step.
        """
        coros_list = list(coros)
        coros = set(coros_list)

        final = {}
        to_send = dict.fromkeys(coros, None)

        while True:
            requests = {}
            for coro, x in list(to_send.items()):
                try:
                    request = coro.send(x)
                except StopIteration as e:
                    final[coro] = e.value
                    del to_send[coro]
                except Exception as e:
                    for _coro in to_send.keys() - {coro}:
                        try:
                            _coro.throw(CanceledAnalysis)
                        except CanceledAnalysis:
                            pass

                    raise
                else:
                    requests[coro] = request

            if to_send:
                descs = {
                    FrozenDict(desc): coro
                    for coro, request in requests.items()
                    for desc in request
                }
                results = dict(zip(
                    descs.keys(),
                    self.call_anas(descs.keys())
                ))

                to_send = {}
                for desc, res in results.items():
                    to_send.setdefault(descs[desc], []).append(res)
            else:
                return tuple(
                    final[coro]
                    for coro in coros_list
                )


class CanceledAnalysis(BaseException):
    pass


async def join(*coros):
    coros_list = list(coros)
    coros = set(coros_list)
    results = {}
    xs = [None] * len(coros)
    while True:
        _coros = list(coros)
        requests = []
        for coro, x in zip(_coros, xs):
            try:
                descs = coro.send(x)
            except StopIteration as e:
                results[coro] = e.value
                coros.remove(coro)
            else:
                requests.append(_RunMany(
                    Run(**desc)
                    for desc in descs
                ))

        if requests:
            xs = await _RunMany(requests)
        else:
            return tuple(
                results[coro]
                for coro in coros_list
            )


async def concurrently(*awaitables, raise_=True):
    coros = []
    requests = []

    for x in awaitables:
        (
            coros
            if inspect.iscoroutine(x) else
            requests
        ).append(x)

    async def proxy():
        cls = _RunManyRaise if raise_ else _RunMany
        return await cls(requests)

    if not raise_:
        def wrap_coro(coro):
            async def wrapper():
                try:
                    x = await coro
                except Exception as e:
                    excep = e
                    x = None
                else:
                    excep = None

                return (x, excep)
            return wrapper()

        coros = list(map(wrap_coro, coros))

    requests_res, *coros_res = await join(proxy(), *coros)
    coros_res = list(coros_res)
    requests_res = list(requests_res)

    return [
        (
            coros_res.pop()
            if inspect.iscoroutine(x) else
            requests_res.pop()
        )
        for x in awaitables
    ]



class _RunMany:
    def __init__(self, requests):
        self.requests = list(requests)

    def __await__(self):
        def expand_request(request):
            if isinstance(request, Run):
                return (request.desc,)
            elif isinstance(request, _RunMany):
                return tuple(
                    desc
                    for sub in request.requests
                    for desc in expand_request(sub)
                )
            else:
                raise TypeError('Unknown request type')

        descs = expand_request(self)

        xs = yield descs

        def rebuild_response(request, xs):
            if isinstance(request, Run):
                x, *xs = xs
                return (x, xs)
            elif isinstance(request, _RunMany):
                res = []
                for sub in request.requests:
                    x, xs = rebuild_response(sub, xs)
                    res.append(x)
                return (res, xs)
            else:
                raise TypeError('Unknown request')

        xs, remaining = rebuild_response(self, xs)
        assert not remaining
        return xs


class _RunManyRaise(_RunMany):
    def __await__(self):
        results = yield from super().__await__()

        if results:
            xs, exceps = zip(*results)
            exceps = [
                excep
                for excep in exceps
                if excep is not None
            ]
            if exceps:
                # Choose one arbitrarily, might benefit from PEP 654 exception
                # groups
                raise exceps[0]
            else:
                return tuple(xs)
        else:
            return []


class Run:
    def __init__(self, **desc):
        self.desc = desc

    def __await__(self):
        x, excep = (yield [self.desc])[0]
        if excep is None:
            return x
        else:
            raise excep



def rust_analysis(f):
    sig = inspect.signature(f)

    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
        coro = f(self, *args, **kwargs)
        return self._run_coros([coro])[0]

    wrapper.asyn = f
    return wrapper

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
