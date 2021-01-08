# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2018, Arm Limited and contributors.
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


"""
Miscellaneous utilities that don't fit anywhere else.

Also used as a home for everything that would create cyclic dependency issues
between modules if they were hosted in their "logical" module. This is mostly
done for secondary utilities that are not used often.
"""

import hashlib
import zlib
import time
import re
import abc
import copy
import collections
from collections.abc import Mapping
from collections import OrderedDict
import contextlib
import inspect
import io
import logging
import logging.config
import functools
import pickle
import sys
import os
import importlib
import pkgutil
import operator
import threading
import itertools
import weakref
from weakref import WeakKeyDictionary
import urllib.parse
import warnings
import textwrap
import webbrowser
import mimetypes

import ruamel.yaml
from ruamel.yaml import YAML

# These modules may not be installed as they are only used for notebook usage
try:
    import sphobjinv
    from IPython.display import IFrame
# ModuleNotFoundError does not exist in Python < 3.6
except ImportError:
    pass

import lisa
import lisa.assets
from lisa.version import parse_version, format_version


# Do not infer the value using __file__, since it will break later on when
# lisa package is installed in the site-package locations using pip, which
# are typically not writable.
LISA_HOME = os.getenv('LISA_HOME')
"""
The detected location of your LISA installation
"""

RESULT_DIR = 'results'
LATEST_LINK = 'results_latest'

TASK_COMM_MAX_LEN = 16 - 1
"""
Value of ``TASK_COMM_LEN - 1`` macro in the kernel, to account for ``\0``
terminator.
"""


class Loggable:
    """
    A simple class for uniformly named loggers
    """

    @classmethod
    def get_logger(cls, suffix=None):
        cls_name = cls.__name__
        module = inspect.getmodule(cls)
        if module:
            name = module.__name__ + '.' + cls_name
        else:
            name = cls_name
        if suffix:
            name += '.' + suffix
        return logging.getLogger(name)

    @classmethod
    def log_locals(cls, var_names=None, level='debug'):
        """
        Debugging aid: log the local variables of the calling function

        :param var_names: List of variable names to display, or all of them if
            left to default.
        :type var_names: list(str)

        :param level: log level to use.
        :type level: str
        """
        level = getattr(logging, level.upper())
        call_frame = sys._getframe(1)

        for name, val in call_frame.f_locals.items():
            if var_names and name not in var_names:
                continue
            cls.get_logger().log(level, f'Local variable: {name}: {val}')


def get_subclasses(cls, only_leaves=False, cls_set=None):
    """Get all indirect subclasses of the class."""
    if cls_set is None:
        cls_set = set()

    for subcls in cls.__subclasses__():
        if subcls not in cls_set:
            to_be_added = set(get_subclasses(subcls, only_leaves, cls_set))
            to_be_added.add(subcls)
            if only_leaves:
                to_be_added = {
                    cls for cls in to_be_added
                    if not cls.__subclasses__()
                }
            cls_set.update(to_be_added)

    return cls_set


def get_cls_name(cls, style=None, fully_qualified=True):
    """
    Get a prettily-formated name for the class given as parameter

    :param cls: class to get the name from
    :type cls: type

    :param style: When "rst", a RestructuredText snippet is returned
    :param style: str

    """
    if cls is None:
        return 'None'

    if fully_qualified or style == 'rst':
        mod_name = inspect.getmodule(cls).__name__
        mod_name = mod_name + '.' if mod_name not in ('builtins', '__main__') else ''
    else:
        mod_name = ''

    name = mod_name + cls.__qualname__
    if style == 'rst':
        name = f':class:`~{name}`'
    return name


class HideExekallID:
    """Hide the subclasses in the simplified ID format of exekall.

    That is mainly used for uninteresting classes that do not add any useful
    information to the ID. This should not be used on domain-specific classes
    since alternatives may be used by the user while debugging for example.
    Hiding too many classes may lead to ambiguity, which is exactly what the ID
    is fighting against.
    """
    pass


def memoized(f):
    """
    Decorator to memoize the result of a callable, based on
    :func:`functools.lru_cache`

    .. note:: The first parameter of the callable is cached with a weak
        reference. This suits well the method use-case, since we don't want the
        memoization of methods to prevent garbage collection of the instances
        they are bound to.
    """
    return lru_memoized()(f)

def lru_memoized(first_param_maxsize=None, other_params_maxsize=1024):
    """
    Decorator to memoize the result of a callable, based on
    :func:`functools.lru_cache`

    :param first_param_maxsize: Maximum number of cached values for the first
        parameter.
    :type first_param_maxsize: int or None

    :param other_params_maxsize: Maximum number of cached combinations of all
        parameters except the first one.
    :type other_params_maxsize: int or None

    .. note:: The first parameter of the callable is cached with a weak
        reference. This suits well the method use-case, since we don't want the
        memoization of methods to prevent garbage collection of the instances
        they are bound to.
    """

    def decorator(f):
        def apply_lru(f):
            # maxsize should be a power of two for better speed, see:
            # https://docs.python.org/3/library/functools.html#functools.lru_cache
            return functools.lru_cache(maxsize=other_params_maxsize, typed=True)(f)

        # We need at least one positional parameter for the WeakKeyDictionary
        if inspect.signature(f).parameters:
            cache_map = WeakKeyDictionary()
            insertion_counter = 0
            insertion_order = WeakKeyDictionary()

            @functools.wraps(f)
            def wrapper(first, *args, **kwargs):
                nonlocal insertion_counter
                try:
                    partial = cache_map[first]
                except KeyError:
                    # Only keep a weak reference here for the "partial" closure
                    ref = weakref.ref(first)

                    # This partial function does not take "first" as parameter, so
                    # that the lru_cache will not keep a reference on it
                    @apply_lru
                    def partial(*args, **kwargs):
                        return f(ref(), *args, **kwargs)

                    cache_map[first] = partial
                    insertion_order[first] = insertion_counter
                    insertion_counter += 1

                    # Delete the caches for objects that are too old
                    if first_param_maxsize is not None:
                        # Make sure the content of insertion_order will not
                        # change while iterating over it
                        to_remove = [
                            val
                            for val, counter in insertion_order.items()
                            if insertion_counter - counter > first_param_maxsize
                        ]

                        for val in to_remove:
                            del cache_map[val]
                            del insertion_order[val]

                return partial(*args, **kwargs)

            return wrapper
        else:
            return apply_lru(f)

    return decorator


def resolve_dotted_name(name):
    """Only resolve names where __qualname__ == __name__, i.e the callable is a
    module-level name."""
    mod_name, callable_name = name.rsplit('.', 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, callable_name)


def import_all_submodules(pkg, best_effort=False):
    """
    Import all submodules of a given package.

    :param pkg: Package to import.
    :type pkg: types.ModuleType

    :param best_effort: If ``True``, modules in the hierarchy that cannot be
        imported will be silently skipped.
    :type best_effort: bool
    """
    return _import_all_submodules(pkg.__name__, pkg.__path__, best_effort)


def _import_all_submodules(pkg_name, pkg_path, best_effort=False):
    modules = []
    for _, module_name, _ in (
        pkgutil.walk_packages(pkg_path, prefix=pkg_name + '.')
    ):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            if best_effort:
                pass
            else:
                raise
        else:
            modules.append(module)

    return modules


class UnknownTagPlaceholder:
    def __init__(self, tag, data, location=None):
        self.tag = tag
        self.data = data
        self.location = location

    def __str__(self):
        return f'<UnknownTagPlaceholder of {self.tag}>'


class Serializable(Loggable):
    """
    A helper class for YAML serialization/deserialization

    The following YAML tags are supported on top of what YAML provides out of
    the box:

        * ``!call``: call a Python callable with a mapping of arguments:

            .. code-block:: yaml

                # will execute:
                # package.module.Class(arg1='foo', arg2='bar', arg3=42)
                # NB: there is no space after "call:"
                !call:package.module.Class
                    arg1: foo
                    arg2: bar
                    arg3: 42

        * ``!include``: include the content of another YAML file. Environment
          variables are expanded in the given path:

            .. code-block:: yaml

                !include /foo/$ENV_VAR/bar.yml

          Relative paths are treated as relative to the file in which the
          ``!include`` tag appears.

        * ``!env``: take the value of an environment variable, and convert
          it to a Python type:

            .. code-block:: yaml

                !env:int MY_ENV_VAR

            If `interpolate` is used as type, the value will be interpolated
            using :func:`os.path.expandvars` and the resulting string
            returned:

                .. code-block:: yaml

                    !env:interpolate /foo/$MY_ENV_VAR/bar

        * ``!var``: reference a module-level variable:

            .. code-block:: yaml

                !var package.module.var

    .. note:: Not to be used on its own - instead, your class should inherit
        from this class to gain serialization superpowers.
    """
    serialized_whitelist = []
    serialized_blacklist = []
    serialized_placeholders = dict()

    YAML_ENCODING = 'utf-8'
    "Encoding used for YAML files"

    DEFAULT_SERIALIZATION_FMT = 'yaml'
    "Default format used when serializing objects"

    @classmethod
    def _get_yaml(cls, typ):
        yaml = YAML(typ=typ)

        # If allow_unicode=True, true unicode characters will be written to the
        # file instead of being replaced by escape sequence.
        yaml.allow_unicode = ('utf' in cls.YAML_ENCODING)
        yaml.default_flow_style = False
        yaml.indent = 4
        yaml.constructor.add_constructor('!include', functools.partial(cls._yaml_include_constructor, typ))
        yaml.constructor.add_constructor('!var', cls._yaml_var_constructor)
        yaml.constructor.add_multi_constructor('!env:', cls._yaml_env_var_constructor)
        yaml.constructor.add_multi_constructor('!call:', cls._yaml_call_constructor)

        # Replace unknown tags by a placeholder object containing the data.
        # This happens when the class was not imported at the time the object
        # was deserialized
        yaml.constructor.add_constructor(None, cls._yaml_unknown_tag_constructor)

        return yaml

    @classmethod
    def _yaml_unknown_tag_constructor(cls, loader, node):
        # Get the basic data types that can be expressed using the YAML syntax,
        # without using any tag-specific constructor
        data = None
        for constructor in (
            loader.construct_scalar,
            loader.construct_sequence,
            loader.construct_mapping
        ):
            try:
                data = constructor(node)
            except ruamel.yaml.constructor.ConstructorError:
                continue
            else:
                break

        tag = node.tag
        cls.get_logger().debug(f'Could not find constructor for YAML tag "{tag}" ({str(node.start_mark).strip()}), using a placeholder')

        return UnknownTagPlaceholder(tag, data, location=node.start_mark)

    @classmethod
    def _yaml_call_constructor(cls, loader, suffix, node):
        # Restrict to keyword arguments to have improve stability of
        # configuration files.
        kwargs = loader.construct_mapping(node, deep=True)
        return loader.make_python_instance(suffix, node, kwds=kwargs, newobj=False)

    # Allow !include to use relative paths from the current file. Since we
    # introduce a global state, we use thread-local storage.
    _included_path = threading.local()
    _included_path.val = None
    @staticmethod
    @contextlib.contextmanager
    def _set_relative_include_root(path):
        old = Serializable._included_path.val
        Serializable._included_path.val = path
        try:
            yield
        finally:
            Serializable._included_path.val = old

    @classmethod
    def _yaml_include_constructor(cls, typ, loader, node):
        path = loader.construct_scalar(node)
        assert isinstance(path, str)
        path = os.path.expandvars(path)

        # Paths are relative to the file that is being included
        if not os.path.isabs(path):
            path = os.path.join(Serializable._included_path.val, path)

        # Since the parser is not re-entrant, create a fresh one
        yaml = cls._get_yaml(typ)

        with cls._set_relative_include_root(path):
            with open(path, encoding=cls.YAML_ENCODING) as f:
                return yaml.load(f)

    @classmethod
    def _yaml_env_var_constructor(cls, loader, suffix, node):
        string = loader.construct_scalar(node)
        assert isinstance(string, str)

        type_ = suffix
        if type_ == 'interpolate':
            return os.path.expandvars(string)
        else:
            varname = string

            type_ = loader.find_python_name(type_, node.start_mark)
            assert callable(type_)
            try:
                value = os.environ[varname]
            except KeyError:
                cls._warn_missing_env(varname)
                return None
            else:
                return type_(value)

    @classmethod
    # memoize to avoid displaying the same message twice
    @memoized
    def _warn_missing_env(cls, varname):
        cls.get_logger().warning(f'Environment variable "{varname}" not defined, using None value')

    @classmethod
    def _yaml_var_constructor(cls, loader, node):
        varname = loader.construct_scalar(node)
        assert isinstance(varname, str)
        return loader.find_python_name(varname, node.start_mark)

    def to_path(self, filepath, fmt=None):
        """
        Serialize the object to a file

        :param filepath: The path of the file or file-like object in which the
            object will be dumped.
        :type filepath: str or io.IOBase

        :param fmt: Serialization format.
        :type fmt: str
        """

        data = self
        return self._to_path(data, filepath, fmt)

    @classmethod
    def _to_path(cls, instance, filepath, fmt):
        if fmt is None:
            fmt = cls.DEFAULT_SERIALIZATION_FMT

        yaml_kwargs = dict(mode='w', encoding=cls.YAML_ENCODING)
        if fmt == 'yaml':
            kwargs = yaml_kwargs
            dumper = cls._get_yaml('unsafe').dump
        elif fmt == 'yaml-roundtrip':
            kwargs = yaml_kwargs
            dumper = cls._get_yaml('rt').dump
        elif fmt == 'pickle':
            kwargs = dict(mode='wb')
            dumper = pickle.dump
        else:
            raise ValueError(f'Unknown format "{fmt}"')

        if isinstance(filepath, io.IOBase):
            cm = nullcontext(filepath)
        else:
            cm = open(str(filepath), **kwargs)

        with cm as fh:
            dumper(instance, fh)

    @classmethod
    def _to_yaml(cls, data):
        yaml = cls._get_yaml('unsafe')
        buff = io.StringIO()
        yaml.dump(data, buff)
        return buff.getvalue()

    def to_yaml(self):
        """
        Return a YAML string with the serialized object.
        """
        return self._to_yaml(self)

    @classmethod
    def from_path(cls, filepath, fmt=None):
        """
        Deserialize an object from a file

        :param filepath: The path of file in which the object has been dumped
        :type filepath: str

        :param fmt: Serialization format.
        :type fmt: str

        :raises AssertionError: if the deserialized object is not an instance
                                of the class.

        .. note:: Only deserialize files from trusted source, as both pickle
            and YAML formats can lead to arbitrary code execution.
        """
        instance = cls._from_path(filepath, fmt)
        assert isinstance(instance, cls)
        return instance

    @classmethod
    def _from_path(cls, filepath, fmt):
        yaml = cls._get_yaml('unsafe')
        filepath = str(filepath)
        if fmt is None:
            fmt = cls.DEFAULT_SERIALIZATION_FMT

        if fmt == 'yaml':
            kwargs = dict(mode='r', encoding=cls.YAML_ENCODING)
            loader = yaml.load
        elif fmt == 'pickle':
            kwargs = dict(mode='rb')
            loader = pickle.load
        else:
            raise ValueError(f'Unknown format "{fmt}"')

        with cls._set_relative_include_root(os.path.dirname(filepath)):
            with open(filepath, **kwargs) as fh:
                instance = loader(fh)

        return instance

    def __getstate__(self):
        """
        Filter the instance's attributes upon serialization.

        The following class attributes can be used to customize the serialized
        content:
            * :attr:`serialized_whitelist`: list of attribute names to
              serialize. All other attributes will be ignored and will not be
              saved/restored.

            * :attr:`serialized_blacklist`: list of attribute names to not
              serialize.  All other attributes will be saved/restored.

            * serialized_placeholders: Map of attribute names to placeholder
              values. These attributes will not be serialized, and the
              placeholder value will be used upon restoration.

            If both :attr:`serialized_whitelist` and
            :attr:`serialized_blacklist` are specified,
            :attr:`serialized_blacklist` is ignored.
        """

        dct = copy.copy(self.__dict__)
        if self.serialized_whitelist:
            dct = {attr: dct[attr] for attr in self.serialized_whitelist}

        elif self.serialized_blacklist:
            for attr in self.serialized_blacklist:
                dct.pop(attr, None)

        for attr, _ in self.serialized_placeholders.items():
            dct.pop(attr, None)

        return dct

    def __setstate__(self, dct):
        if self.serialized_placeholders:
            dct.update(copy.deepcopy(self.serialized_placeholders))
        self.__dict__ = dct

    def __copy__(self):
        """
        Make sure that copying the class still works as usual, without
        dropping some attributes by defining __copy__
        """
        try:
            return super().__copy__()
        except AttributeError:
            cls = self.__class__
            new = cls.__new__(cls)
            new.__dict__.update(self.__dict__)
            return new


def setup_logging(filepath='logging.conf', level=None):
    """
    Initialize logging used for all the LISA modules.

    :param filepath: the relative or absolute path of the logging
                     configuration to use. Relative path uses
                     :attr:`lisa.utils.LISA_HOME` as base folder.
    :type filepath: str

    :param level: Override the conf file and force logging level. Defaults to
        ``logging.INFO``.
    :type level: int or str
    """
    resolved_level = logging.INFO if level is None else level

    # Capture the warnings as log entries
    logging.captureWarnings(True)

    if level is not None:
        log_format = '[%(asctime)s][%(name)s] %(levelname)s  %(message)s'
        logging.basicConfig(level=resolved_level, format=log_format)
    else:
        # Load the specified logfile using an absolute path
        if not os.path.isabs(filepath):
            filepath = os.path.join(LISA_HOME, filepath)

        # Set the level first, so the config file can override with more details
        logging.getLogger().setLevel(resolved_level)

        if os.path.exists(filepath):
            logging.config.fileConfig(filepath)
            logging.info(f'Using LISA logging configuration: {filepath}')
        else:
            raise FileNotFoundError(f'Logging configuration file not found: {filepath}')


class ArtifactPath(str, Loggable, HideExekallID):
    """Path to a folder that can be used to store artifacts of a function.
    This must be a clean folder, already created on disk.
    """
    def __new__(cls, root, relative, *args, **kwargs):
        root = os.path.realpath(str(root))
        relative = str(relative)
        # we only support paths relative to the root parameter
        assert not os.path.isabs(relative)
        absolute = os.path.join(root, relative)

        # Use a resolved absolute path so it is more convenient for users to
        # manipulate
        path = os.path.realpath(absolute)

        path_str = super().__new__(cls, path, *args, **kwargs)
        # Record the actual root, so we can relocate the path later with an
        # updated root
        path_str.root = root
        path_str.relative = relative
        return path_str

    def __fspath__(self):
        return str(self)

    def __reduce__(self):
        # Serialize the path relatively to the root, so it can be relocated
        # easily
        relative = self.relative_to(self.root)
        return (type(self), (self.root, relative))

    def relative_to(self, path):
        return os.path.relpath(str(self), start=str(path))

    def with_root(self, root):
        # Get the path relative to the old root
        relative = self.relative_to(self.root)

        # Swap-in the new root and return a new instance
        return type(self)(root, relative)

    @classmethod
    def join(cls, path1, path2):
        """
        Join two paths together, similarly to :func:`os.path.join`.

        If ``path1`` is a :class:`ArtifactPath`, the result will also be one,
        and the root of ``path1`` will be used as the root of the new path.
        """
        if isinstance(path1, cls):
            joined = cls(
                root=path1.root,
                relative=os.path.join(path1.relative, str(path2))
            )
        else:
            joined = os.path.join(str(path1), str(path2))

        return joined


def value_range(start, stop, step=None, inclusive=False):
    """
    Equivalent to builtin :class:`range` function, but works for floats as well.

    :param start: First value to use.
    :type start: numbers.Number

    :param stop: Last value to use.
    :type stop: numbers.Number

    :param step: Increment. If ``None``, increment defaults to 1.
    :type step: numbers.Number

    :param inclusive: If ``True``, the ``stop`` value will be included (unlike
        the builtin :class:`range`)
    :type inclusive: bool

    .. note:: Unlike :class:`range`, it will raise :exc:`ValueError` if
        ``start > stop and step > 0``.
    """

    step = 1 if step is None else step

    if stop < start and step > 0:
        raise ValueError(f"step ({step}) > 0 but stop ({stop}) < start ({start})")

    if not step:
        raise ValueError(f"Step cannot be 0: {step}")

    ops = {
        (True, True): operator.le,
        (True, False): operator.lt,

        (False, True): operator.ge,
        (False, False): operator.gt,
    }
    op = ops[start <= stop, inclusive]
    comp = lambda x: op(x, stop)
    return itertools.takewhile(comp, itertools.count(start, step))


def filter_values(iterable, values):
    """
    Yield value from ``iterable`` unless they are in ``values``.
    """
    return itertools.filterfalse(
        (lambda x: x in values),
        iterable,
    )


def groupby(iterable, key=None, reverse=False):
    """
    Equivalent of :func:`itertools.groupby`, with a pre-sorting so it works as
    expected.

    :param iterable: Iterable to group.

    :param key: Forwarded to :func:`sorted`
    :param reverse: Forwarded to :func:`sorted`
    """
    # We need to sort before feeding to groupby, or it will fail to establish
    # the groups as expected.
    iterable = sorted(iterable, key=key, reverse=reverse)
    return itertools.groupby(iterable, key=key)


def grouper(iterable, n, fillvalue=None):
    """
    Collect data into fixed-length chunks or blocks
    """
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    # Since the same iterator is used, it will yield a new item every time zip
    # call next() on it
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


def group_by_value(mapping, key_sort=lambda x: x):
    """
    Group a mapping by its values

    :param mapping: Mapping to reverse
    :type mapping: collections.abc.Mapping

    :param key_sort: The ``key`` parameter to a :func:`sorted` call on the
      mapping keys
    :type key_sort: collections.abc.Callable

    :rtype: collections.OrderedDict

    The idea behind this method is to "reverse" a mapping, IOW to create a new
    mapping that has the passed mapping's values as keys. Since different keys
    can point to the same value, the new values will be lists of old keys.

    **Example:**

    >>> group_by_value({0: 42, 1: 43, 2: 42})
    OrderedDict([(42, [0, 2]), (43, [1])])
    """
    if not key_sort:
        # Just conserve the order
        def key_sort(_):
            return 0

    return OrderedDict(
        (val, sorted((k for k, v in key_group), key=key_sort))
        for val, key_group in groupby(mapping.items(), key=operator.itemgetter(1))
    )


def deduplicate(seq, keep_last=True, key=lambda x: x):
    """
    Deduplicate items in the given sequence and return a list.
    :param seq: Sequence to deduplicate
    :type Seq: collections.abc.Sequence

    :param key: Key function that will be used to determine duplication.  It
        takes one item at a time, returning a hashable key value
    :type key: collections.abc.Callable

    :param keep_last: If True, will keep the last occurence of each duplicated
        items. Otherwise, keep the first occurence.
    :type keep_last: bool
    """
    reorder = reversed if keep_last else (lambda seq: seq)

    out = []
    visited = set()
    for x in reorder(seq):
        k = key(x)
        if k not in visited:
            out.append(x)
            visited.add(k)

    return list(reorder(out))


def fold(f, xs, init=None):
    """
    Fold the given function over ``xs``, with ``init`` initial accumulator
    value.

    This is very similar to :func:`functools.reduce`, except that it is not
    assumed that the function returns values of the same type as the item type.

    This means that this function enforces non-empty input.
    """

    first, *xs = xs
    return functools.reduce(
        f,
        xs,
        f(init, first),
    )

def take(n, iterable):
    """
    Yield the first ``n`` items of an iterator, if ``n`` positive, or last
    items otherwise.

    Yield nothing if the iterator is empty.
    """
    if not n:
        return

    if n > 0:
        yield from itertools.islice(iterable, n)
    else:
        # Inspired from:
        # https://docs.python.org/3/library/itertools.html#itertools-recipes
        n = abs(n)
        yield from iter(collections.deque(iterable, maxlen=n))


def consume(n, iterator):
    """
    Advance the iterator n-steps ahead. If ``n`` is None, consume entirely.
    """
    # Inspired from:
    # https://docs.python.org/3/library/itertools.html#itertools-recipes

    # Use functions that consume iterators at C speed.
    if n is None:
        # feed the entire iterator into a zero-length deque
        collections.deque(iterator, maxlen=0)
    else:
        # advance to the empty slice starting at position n
        next(itertools.islice(iterator, n, n), None)


def get_nested_key(mapping, key_path, getitem=operator.getitem):
    """
    Get a key in a nested mapping

    :param mapping: The mapping to lookup in
    :type mapping: collections.abc.Mapping

    :param key_path: Path to the key in the mapping, in the form of a list of
        keys.
    :type key_path: list

    :param getitem: Function used to get items on the mapping. Defaults to
        :func:`operator.getitem`.
    :type getitem: collections.abc.Callable
    """
    for key in key_path:
        mapping = getitem(mapping, key)

    return mapping


def set_nested_key(mapping, key_path, val, level=None):
    """
    Set a key in a nested mapping

    :param mapping: The mapping to update
    :type mapping: collections.abc.MutableMapping

    :param key_path: Path to the key in the mapping, in the form of a list of
        keys.
    :type key_path: list

    :param level: Factory used when creating a level is needed. By default,
        ``type(mapping)`` will be called without any parameter.
    :type level: collections.abc.Callable
    """
    assert key_path

    if level is None:
        # This should work for dict and most basic structures
        level = type(mapping)

    for key in key_path[:-1]:
        try:
            mapping = mapping[key]
        except KeyError:
            new_level = level()
            mapping[key] = new_level
            mapping = new_level

    mapping[key_path[-1]] = val


def get_call_site(levels=0, exclude_caller_module=False):
    """
    Get the location of the source that called that function.

    :returns: (caller, filename, lineno) tuples. Any component can be None if
        nothing was found. Caller is a string containing the function name.

    :param levels: How many levels to look at in the stack
    :type levels: int

    :param exclude_caller_module: Return the first function in the stack that
        is not defined in the same module as the direct caller of
        :func:`get_call_site`.

    .. warning:: That function will exclude all source files that are not part
        of the `lisa` package. It will also exclude functions of
        :mod:`lisa.utils` module.
    """

    try:
        # context=0 speeds up a lot the stack retrieval, since it avoids
        # reading the source files
        stack = inspect.stack(context=0)
    # Getting the stack can sometimes fail under IPython for some reason:
    # https://github.com/ipython/ipython/issues/1456/
    except IndexError:
        return (None, None, None)

    # Exclude all functions from lisa.utils
    excluded_files = {
        __file__,
    }
    if exclude_caller_module:
        excluded_files.add(stack[1].filename)

    caller = None
    filename = None
    lineno = None
    for frame in stack[levels + 1:]:
        caller = frame.function
        filename = frame.filename
        lineno = frame.lineno
        # exclude all non-lisa sources
        if not any(
            filename.startswith(path)
            for path in lisa.__path__
        ) or filename in excluded_files:
            continue
        else:
            break

    return (caller, filename, lineno)


def is_running_sphinx():
    """
    Returns True if the module is imported when Sphinx is running, False
    otherwise.
    """
    return 'sphinx' in sys.modules


def is_running_ipython():
    """
    Returns True if running in IPython console or Jupyter notebook, False
    otherwise.
    """
    try:
        __IPYTHON__
    except NameError:
        return False
    else:
        return True


def non_recursive_property(f):
    """
    Create a property that raises an :exc:`AttributeError` if it is re-entered.

    .. note:: This only guards against single-thread accesses, it is not
        threadsafe.
    """

    # WeakKeyDictionary ensures that instances will not be held alive just for
    # the guards. Since there is one guard_map per property, we only need to
    # index on the instances
    guard_map = WeakKeyDictionary()

    def _get(self):
        return guard_map.get(self, False)

    def _set(self, val):
        guard_map[self] = val

    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
        if _get(self):
            raise AttributeError(f'Recursive access to property "{self.__class__.__qualname__}.{f.__name__}" while computing its value')

        try:
            _set(self, True)
            return f(self, *args, **kwargs)
        finally:
            _set(self, False)

    return property(wrapper)


def get_short_doc(obj):
    """
    Get the short documentation paragraph at the beginning of docstrings.
    """
    docstring = inspect.getdoc(obj)
    if docstring:
        docstring = split_paragraphs(docstring)[0]
        docstring = ' '.join(docstring.splitlines())
        if not docstring.endswith('.'):
            docstring += '.'
    else:
        docstring = ''
    return docstring


def optional_kwargs(func):
    """
    Decorator used to allow another decorator to both take keyword parameters
    when called, and none when not called::

        @optional_kwargs
        def decorator(func, xxx=42):
            ...

        # Both of these work:

        @decorator
        def foo(...):
           ...

        @decorator(xxx=42)
        def foo(...):
           ...

    .. note:: This only works for keyword parameters.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not kwargs and len(args) == 1 and callable(args[0]):
            return func(args[0])
        else:
            if args:
                raise TypeError(f'Positional parameters are not allowed when applying {func.__qualname__} decorator, please use keyword arguments')
            return functools.partial(func, **kwargs)

    return wrapper


def update_wrapper_doc(func, added_by=None, sig_from=None, description=None, remove_params=None, include_kwargs=False):
    """
    Equivalent to :func:`functools.wraps` that updates the signature by taking
    into account the wrapper's extra *keyword-only* parameters and the given
    description.

    :param func: callable to decorate
    :type func: collections.abc.Callable

    :param added_by: Add some kind of reference to give a sense of where the
        new behaviour of the wraps function comes from.
    :type added_by: collections.abc.Callable or str or None

    :param sig_from: By default, the signature containing the added parameters
        will be taken from ``func``. This allows overriding that, in case ``func``
        is just a wrapper around something else.
    :type sig_from: collections.abc.Callable

    :param description: Extra description output in the docstring.
    :type description: str or None

    :param remove_params: Set of parameter names of ``func`` to not include in
        the decorated function signature. This can be used to hide parameters
        that are only used as part of a decorated/decorator protocol, and not
        exposed in the final decorated function.
    :type remove_params: list(str) or None

    :param include_kwargs: If `True`, variable keyword parameter (``**kwargs``)
        of the decorator is kept in the signature. It is usually removed, since
        it's mostly used to transparently forward arguments to the inner
        ``func``, but can also serve other purposes.
    :type include_kwargs: bool

    .. note:: :func:`functools.wraps` is applied by this decorator, which will
        not work if you applied it yourself.
    """

    if description:
        description = f'\n{description}\n'

    remove_params = remove_params if remove_params else set()

    def decorator(f):
        wrapper_sig = inspect.signature(f if sig_from is None else sig_from)
        f = functools.wraps(func)(f)
        f_sig = inspect.signature(f)

        added_params = [
            desc
            for name, desc in wrapper_sig.parameters.items()
            if (
                desc.kind == inspect.Parameter.KEYWORD_ONLY
                or (
                    include_kwargs
                    and desc.kind == inspect.Parameter.VAR_KEYWORD
                )
            )
        ]
        added_names = {desc.name for desc in added_params}

        if include_kwargs:
            f_var_keyword_params = []
        else:
            f_var_keyword_params = [
                desc
                for name, desc in f_sig.parameters.items()
                if (
                    desc.kind == inspect.Parameter.VAR_KEYWORD
                    and name not in remove_params
                )
            ]

        f_params = [
            desc
            for name, desc in f_sig.parameters.items()
            if (
                desc.name not in added_names
                and desc not in f_var_keyword_params
                and name not in remove_params
            )
        ]

        f.__signature__ = f_sig.replace(
            # added_params are keyword-only, so they need to go before the var
            # keyword param if there is any
            parameters=f_params + added_params + f_var_keyword_params,
        )

        if added_by:
            if callable(added_by):
                added_by_ = get_sphinx_name(added_by, style='rst')
            else:
                added_by_ = added_by

            added_by_ = f'**Added by** {added_by_}:\n'
        else:
            added_by_ = ''

        # Replace the one-liner f description
        extra_doc = f"\n\n{added_by_}{(description if description else '')}"

        f_doc = inspect.getdoc(f) or ''
        f.__doc__ = f_doc + extra_doc

        return f
    return decorator


DEPRECATED_MAP = {}
"""
Global dictionary of deprecated classes, functions and so on.
"""


def deprecate(msg=None, replaced_by=None, deprecated_in=None, removed_in=None, parameter=None):
    """
    Mark a class, method, function etc as deprecated and update its docstring.

    :param msg: Message to tell more about this deprecation.
    :type msg: str or None

    :param replaced_by: Other object the deprecated object is replaced by.
    :type replaced_by: object

    :param deprecated_in: Version in which the object was flagged as deprecated.
    :type deprecated_in: str

    :param removed_in: Version in which the deprecated object will be removed.
    :type removed_in: str

    :param parameter: If not ``None``, the deprecation will only apply to the
        usage of the given parameter. The relevant ``:param:`` block in the
        docstring will be updated, and the deprecation warning will be emitted
        anytime a caller gives a value to that parameter (default or not).
    :type parameter: str or None

    .. note:: In order to decorate all the accessors of properties, apply the
        decorator once the property is fully built::

            class C:
                @property
                def foo(self):
                    pass

                @foo.setter
                def foo(self, val):
                    pass

                # Once all getters/setter/deleters are set, apply the decorator
                foo = deprecate()(foo)
    """
    def get_meth_stacklevel(func_name):
        # Special methods are usually called from another module, so
        # make sure the warning filters set on lisa will pick these up.
        if func_name.startswith('__') and func_name.endswith('__'):
            return 1
        else:
            return 2

    if removed_in:
        removed_in = parse_version(removed_in)
    current_version = lisa.version.version_tuple

    def make_msg(deprecated_obj, parameter=None, style=None, show_doc_url=True, indent=None):
        if replaced_by is not None:
            doc_url = ''
            if show_doc_url:
                with contextlib.suppress(Exception):
                    doc_url = f' (see: {get_doc_url(replaced_by)})'

            replacement_msg = f', use {get_sphinx_name(replaced_by, style=style)} instead{doc_url}'
        else:
            replacement_msg = ''

        if removed_in:
            removal_msg = f' and will be removed in version {format_version(removed_in)}'
        else:
            removal_msg = ''

        name = get_sphinx_name(deprecated_obj, style=style, abbrev=True)
        if parameter:
            if style == 'rst':
                parameter = f'``{parameter}``'
            name = f'{parameter} parameter of {name}'

        if msg is None:
            _msg = ''
        else:
            _msg = textwrap.dedent(msg).strip()
            if indent:
                _msg = _msg.replace('\n', '\n' + indent)

        return '{name} is deprecated{remove}{replace}{msg}'.format(
            name=name,
            replace=replacement_msg,
            remove=removal_msg,
            msg=': ' +  _msg if _msg else '',
        )

    def decorator(obj):
        obj_name = get_sphinx_name(obj)

        if removed_in and current_version >= removed_in:
            raise DeprecationWarning(f'{obj_name} was marked as being removed in version {format_version(removed_in)} but is still present in current version {format_version(current_version)}')

        # stacklevel != 1 breaks the filtering for warnings emitted by APIs
        # called from external modules, like __init_subclass__ that is called
        # from other modules like abc.py
        if parameter:
            register_deprecated_map = False
            def wrap_func(func, stacklevel=1):
                sig = inspect.signature(func)
                @functools.wraps(func)
                def wrapper(*args, **kwargs):
                    kwargs = sig.bind(*args, **kwargs).arguments
                    if parameter in kwargs:
                        warnings.warn(make_msg(obj, parameter), DeprecationWarning, stacklevel=stacklevel)
                    return func(**kwargs)
                return wrapper
        else:
            register_deprecated_map = True
            def wrap_func(func, stacklevel=1):
                @functools.wraps(func)
                def wrapper(*args, **kwargs):
                    warnings.warn(make_msg(obj), DeprecationWarning, stacklevel=stacklevel)
                    return func(*args, **kwargs)
                return wrapper

        # For classes, wrap __new__ and update docstring
        if isinstance(obj, type):
            # Warn on instance creation
            obj.__init__ = wrap_func(obj.__init__)
            # Will show the warning when the class is subclassed
            # in Python >= 3.6 . Earlier versions of Python don't have
            # object.__init_subclass__
            if hasattr(obj, '__init_subclass__'):
                obj.__init_subclass__ = wrap_func(obj.__init_subclass__)
            return_obj = obj
            update_doc_of = obj

        elif isinstance(obj, property):
            # Since we cannot update the property itself, replace it with a new
            # one that uses a wrapped getter. This should be safe as properties
            # seems to be immutable, so there is no risk of somebody
            # monkey-patching the object and us throwing away the extra
            # attributes.
            # Note that this will only wrap accessors that are visible at the
            # time the decorator is applied.
            obj = property(
                fget=wrap_func(obj.fget, stacklevel=2),
                fset=wrap_func(obj.fset, stacklevel=2),
                fdel=wrap_func(obj.fdel, stacklevel=2),
            )
            return_obj = obj
            update_doc_of = obj

        elif isinstance(obj, (staticmethod, classmethod)):
            func = obj.__func__
            stacklevel = get_meth_stacklevel(func.__name__)
            func = wrap_func(func, stacklevel=stacklevel)
            # Build a new staticmethod/classmethod with the updated function
            return_obj = obj.__class__(func)
            # Updating the __doc__ of the staticmethod/classmethod itself will
            # have no effect, so update the doc of the underlying function
            update_doc_of = func

        # For other callables, emit the warning when called
        else:
            stacklevel = get_meth_stacklevel(obj.__name__)
            return_obj = wrap_func(obj, stacklevel=stacklevel)
            update_doc_of = return_obj

        extra_doc = textwrap.dedent(
            """
        .. attention::

            .. deprecated:: {deprecated_in}

            {msg}
        """.format(
                deprecated_in=deprecated_in if deprecated_in else '<unknown>',
                # The documentation already creates references to the replacement,
                # so we can avoid downloading the inventory for nothing.
                msg=make_msg(obj, parameter, style='rst', show_doc_url=False, indent=' ' * 12),
            )).strip()
        doc = inspect.getdoc(update_doc_of) or ''

        # Update the description of the parameter in the right spot in the docstring
        if parameter:

            # Split into chunks of restructured text at boundaries such as
            # ":param foo: ..." or ":type foo: ..."
            blocks = []
            curr_block = []
            for line in doc.splitlines(keepends=True):
                if re.match(r'\s*:', line):
                    curr_block = []
                    blocks.append(curr_block)

                curr_block.append(line)

            # Add the extra bits in the right block and join lines of the block
            def update_block(block):
                if re.match(rf':param\s+{re.escape(parameter)}', block[0]):
                    if len(block) > 1:
                        indentation = re.match(r'^(\s*)', block[-1]).group(0)
                    else:
                        indentation = ' ' * 4
                    block.append('\n' + textwrap.indent(extra_doc, indentation) + '\n')
                return ''.join(block)

            doc = ''.join(map(update_block, blocks))

        # Otherwise just append the extra bits at the end of the docstring
        else:
            doc += '\n\n' + extra_doc

        update_doc_of.__doc__ = doc

        # Register in the mapping only once we know what is the returned
        # object, so that what the rest of the world will see is consistent
        # with the 'obj' key
        if register_deprecated_map:
            # Make sure we don't accidentally override an existing entry
            assert obj_name not in DEPRECATED_MAP
            DEPRECATED_MAP[obj_name] = {
                'obj': return_obj,
                'replaced_by': replaced_by,
                'msg': msg,
                'removed_in': removed_in,
                'deprecated_in': deprecated_in,
            }

        return return_obj

    return decorator


def get_doc_url(obj):
    """
    Return an URL to the documentation about the given object.
    """

    # If it does not have a __qualname__, we are probably more interested in
    # its class
    if not hasattr(obj, '__qualname__'):
        obj = obj.__class__

    obj_name = f'{inspect.getmodule(obj).__name__}.{obj.__qualname__}'

    return _get_doc_url(obj_name)


# Make sure to cache (almost) all the queries with a strong reference over
# `obj_name` values
@functools.lru_cache(maxsize=4096)
def _get_doc_url(obj_name):
    doc_base_url = 'https://lisa-linux-integrated-system-analysis.readthedocs.io/en/master/'
    # Use the inventory built by RTD
    inv_url = urllib.parse.urljoin(doc_base_url, 'objects.inv')

    inv = sphobjinv.Inventory(url=inv_url)

    for inv_obj in inv.objects:
        if inv_obj.name == obj_name and inv_obj.domain == "py":
            doc_page = inv_obj.uri.replace('$', inv_obj.name)
            doc_url = urllib.parse.urljoin(doc_base_url, doc_page)
            return doc_url

    raise ValueError(f'Could not find the doc of: {obj_name}')


def show_doc(obj, iframe=False):
    """
    Show the online LISA documentation about the given object.

    :param obj: Object to show the doc of. It can be anything, including
        instances.
    :type obj: object

    :param iframe: If ``True``, uses an IFrame, otherwise opens a web browser.
    :type iframe: bool
    """
    doc_url = get_doc_url(obj)

    if iframe:
        print(doc_url)
        return IFrame(src=doc_url, width="100%", height="600em")
    else:
        webbrowser.open(doc_url)
        return None


def split_paragraphs(string):
    """
    Split `string` into a list of paragraphs.

    A paragraph is delimited by empty lines, or lines containing only
    whitespace characters.
    """
    para_list = []
    curr_para = []
    for line in string.splitlines(keepends=True):
        if line.strip():
            curr_para.append(line)
        else:
            para_list.append(''.join(curr_para))
            curr_para = []

    if curr_para:
        para_list.append(''.join(curr_para))

    return para_list


mimetypes.add_type('text/rst', '.rst')


def guess_format(path):
    """
    Guess the file format from a `path`, using the mime types database.
    """
    if path is None:
        return None

    mime_type = mimetypes.guess_type(path, strict=False)[0]
    guessed_format = mime_type.split('/')[1].split('.', 1)[-1].split('+')[0]
    return guessed_format


@contextlib.contextmanager
def nullcontext(enter_result=None):
    """
    Backport of Python 3.7 ``contextlib.nullcontext``

    This context manager does nothing, so it can be used as a default
    placeholder for code that needs to select at runtime what context manager
    to use.

    :param enter_result: Object that will be bound to the target of the with
        statement, or `None` if nothing is specified.
    :type enter_result: object
    """
    yield enter_result


class ExekallTaggable:
    """
    Allows tagging the objects produced in exekall expressions ID.

    .. seealso:: :ref:`exekall expression ID<exekall-expression-id>`
    """

    @abc.abstractmethod
    def get_tags(self):
        """
        :return: Dictionary of tags and tag values
        :rtype: dict(str, object)
        """
        return {}


def annotations_from_signature(sig):
    """
    Build a PEP484 ``__annotations__`` dictionary from a :class:`inspect.Signature`.
    """
    annotations = {
        name: param_spec.annotation
        for name, param_spec in sig.parameters.items()
        if param_spec.annotation != inspect.Parameter.empty
    }

    if sig.return_annotation != inspect.Signature.empty:
        annotations['return'] = sig.return_annotation

    return annotations


def namedtuple(*args, module, **kwargs):
    """
    Same as :func:`collections.namedtuple`, with
    :class:`collections.abc.Mapping` behaviour.

    .. warning:: Iterating over instances will yield the field names rather the
        values, unlike regular :func:`collections.namedtuple`.

    :param module: Name of the module the type is defined in.
    :type module: str
    """
    assert isinstance(module, str)

    type_ = collections.namedtuple(*args, **kwargs)
    # Make sure this type also has a sensible __module__, since it's going to
    # appear as a base class. Otherwise, Sphinx's autodoc will choke on it.
    type_.__module__ = module

    class Augmented(Mapping):
        # We need to record inner tuple type here so that we have a stable name
        # for the class, otherwise pickle will choke on it
        _type = type_

        # Keep an efficient representation to avoid adding too much overhead on
        # top of the inner tuple
        __slots__ = ['_tuple']

        def __init__(self, *args, **kwargs):
            # This inner tuple attribute is read-only, DO NOT UPDATE IT OR IT
            # WILL BREAK __hash__
            self._tuple = type_(*args, **kwargs)

        def __getattr__(self, attr):
            # Avoid infinite loop when deserializing instances
            if attr in self.__slots__:
                raise AttributeError

            return getattr(self._tuple, attr)

        def __hash__(self):
            return hash(self._tuple)

        def __getitem__(self, key):
            return self._tuple._asdict()[key]

        def __iter__(self):
            return iter(self._tuple._fields)

        def __len__(self):
            return len(self._tuple._fields)

    Augmented.__qualname__ = type_.__qualname__
    Augmented.__name__ = type_.__name__
    Augmented.__doc__ = type_.__doc__
    Augmented.__module__ = module

    # Fixup the inner namedtuple, so it can be pickled
    Augmented._type.__name__ = '_type'
    Augmented._type.__qualname__ = f'{Augmented.__qualname__}.{Augmented._type.__name__}'
    return Augmented

class _TimeMeasure:
    def __init__(self, start, stop):
        self.start = start
        self.stop = stop
        self.nested_delta = 0

    @property
    def delta(self):
        return self.stop - self.start

    @property
    def exclusive_delta(self):
        return self.stop - self.start - self.nested_delta

_measure_time_stack = threading.local()

@contextlib.contextmanager
def measure_time(clock=time.monotonic):
    """
    Context manager to measure time in seconds.

    :param clock: Clock to use.
    :type clock: collections.abc.Callable

    **Example**::

        with measure_time() as measure:
            ...
        print(measure.start, measure.stop, measure.exclusive_delta, measure.exclusive_delta)

    .. note:: The ``exclusive_delta`` discount the time spent in nested
        ``measure_time`` context managers.
    """
    try:
        stack = _measure_time_stack.stack
    except AttributeError:
        stack = []
        _measure_time_stack.stack = stack

    measure = _TimeMeasure(0, 0)
    stack.append(measure)

    start = clock()
    try:
        yield measure
    finally:
        stop = clock()
        measure.start = start
        measure.stop = stop
        stack.pop()
        try:
            parent_measure = stack[-1]
        except IndexError:
            pass
        else:
            parent_measure.nested_delta += measure.delta


def checksum(file_, method):
    """
    Compute a checksum on a given file-like object.

    :param file_: File-like object, as returned by ``open()`` for example.
    :type file_: io.IOBase

    :param method: Checksum to use. Can be any of ``md5``, ``sha256``,
        ``crc32``.
    :type method: str

    The file is read block by block to avoid clogging the memory with a huge
    read.
    """
    if method in ('md5', 'sha256'):
        h = getattr(hashlib, method)()
        update = h.update
        result = h.hexdigest
        chunk_size = h.block_size
    elif method == 'crc32':
        crc32_state = 0
        def update(data):
            nonlocal crc32_state
            crc32_state = zlib.crc32(data, crc32_state) & 0xffffffff
        result = lambda: hex(crc32_state)
        chunk_size = 1 * 1024 * 1024
    else:
        raise ValueError(f'Unsupported method: {method}')

    while True:
        chunk = file_.read(chunk_size)
        if not chunk:
            break
        update(chunk)

    return result()



def get_sphinx_role(obj):
    """
    Return the reStructuredText Sphinx role of a given object.
    """
    if isinstance(obj, type):
        return 'class'
    elif callable(obj):
        if '<locals>' in obj.__qualname__:
            return 'code'
        elif '.' in obj.__qualname__:
            return 'meth'
        else:
            return 'func'
    else:
        return 'code'

def get_sphinx_name(obj, style=None, abbrev=False):
    """
    Get a Sphinx-friendly name of an object.

    :param obj: The object to take the name from
    :type obj: object or type

    :param style: If ``rst``, a reStructuredText reference will be returned.
        Otherwise a bare name is returned.
    :type style: str or None

    :param abbrev: If ``True``, a short name will be used with ``style='rst'``.
    :type abbrev: bool
    """
    if isinstance(obj, (staticmethod, classmethod)):
        obj = obj.__func__
    elif isinstance(obj, property):
        obj = obj.fget

    try:
        mod = obj.__module__ + '.'
    except AttributeError:
        mod = ''

    try:
        qualname = obj.__qualname__
    except AttributeError:
        qualname = str(obj)

    name = mod + qualname

    if style == 'rst':
        return ':{}:`{}{}{}`'.format(
            get_sphinx_role(obj),
            '~' if abbrev else '',
            mod, qualname
        )
    else:
        return name



def newtype(cls, name, doc=None, module=None):
    """
    Make a new class inheriting from ``cls`` with the given ``name``.

    :param cls: Class to inherit from.
    :type cls: type

    :param name: Qualified name of the new type.
    :type name: str

    :param doc: Content of docstring to assign to the new type.
    :type doc: str or None

    :param module: Module name to assign to ``__module__`` attribute of the new
        type. By default, it's inferred from the caller of :func:`newtype`.
    :type module: str or None

    The instances of ``cls`` class will be recognized as instances of the new
    type as well using ``isinstance``.
    """
    class Meta(type(cls)):
        def __instancecheck__(self, x):
            return isinstance(x, cls)

    class New(cls, metaclass=Meta): # pylint: disable=invalid-metaclass
        pass

    New.__name__ = name.split('.')[-1]
    New.__qualname__ = name

    if module is None:
        try:
            module = sys._getframe(1).f_globals.get('__name__', '__main__')
        except Exception: # pylint: disable=broad-except
            module = cls.__module__
    New.__module__ = module
    New.__doc__ = doc

    return New


_SPHINX_NITPICK_IGNORE = set()
def sphinx_register_nitpick_ignore(x):
    """
    Register an object with a name that cannot be resolved and therefore cross
    referenced by Sphinx.

    .. seealso:: https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-nitpick_ignore
    """
    _SPHINX_NITPICK_IGNORE.add(x)

def sphinx_nitpick_ignore():
    """
    Set of objects to ignore without warning when cross referencing in Sphinx.
    """
    # Make sure the set is populated
    import_all_submodules(lisa)
    return _SPHINX_NITPICK_IGNORE

class FrozenDict(Mapping):
    """
    Read-only mapping that is therefore hashable.

    .. note:: The content of the iterable passed to the constructor is
        deepcopied to ensure non-mutability.

    .. note:: Hashability allows to use it as a key in other mappings.
    """
    def __init__(self, x):
        self._dct = copy.deepcopy(dict(x))
        # We cannot use memoized() since it would create an infinite loop
        self._hash = hash(tuple(sorted(self._dct.items())))

    def __getitem__(self, key):
        return self._dct[key]

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self._dct == other._dct
        else:
            return False

    def __iter__(self):
        return iter(self._dct)

    def __len__(self):
        return len(self._dct)

    def __str__(self):
        return str(self._dct)

    def __repr__(self):
        return repr(self._dct)


# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
